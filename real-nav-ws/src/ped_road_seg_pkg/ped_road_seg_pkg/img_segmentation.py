"""img_segmentation_node

- カメラ画像をsubscribeして、走行可能領域と歩行者マスクを推定する
- 推定結果を色分けした画像としてpublishする
- publish前に行動モデルの入力解像度にリサイズする
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import os
from ament_index_python.packages import get_package_share_directory

import torch
import segmentation_models_pytorch as smp
import albumentations as albu
from ultralytics import YOLO

# --- 前処理関数 ---
def get_validation_augmentation():
    # 推論時は強い拡張をせず、入力サイズだけ整える
    # パディング: 画像の外周に余白ピクセルを追加して最小サイズを満たす（内容は拡大縮小しない）
    test_transform = [albu.PadIfNeeded(480, 640)]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    # 画像配列の並び替え: (H,W,C) -> (C,H,W)
    # H:高さ, W:幅, C:色チャンネル（RGBなら3）
    # 多くのPyTorchモデルはチャンネル先頭（CHW）を前提にしている
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    # encoder / encoder_weights に対応した前処理をまとめる
    # - 正規化: 学習時と同じ入力分布に揃える（例: ImageNet の mean/std で標準化）
    # - テンソル化: PyTorch に渡しやすい形 (C,H,W) へ変換
    # ※ preprocessing_fn は値域変換（0-255 -> 0-1）+ mean/std 正規化を行う（encoder/weightsに依存）
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
# ------------------------------------

class ImgSegmentationNode(Node):
    def __init__(self):
        super().__init__('img_segmentation_node')

        # Parameters（呼び出し時に必要に応じて上書きする）
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('output_topic', '/gb_img')
        self.declare_parameter('queue_size', 10)
        self.declare_parameter('yolo_confidence', 0.55)
        self.declare_parameter('yolo_imgsz', 512)
        self.declare_parameter('out_width', 112)
        self.declare_parameter('out_height', 84)

        self.image_topic = self.get_parameter('image_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.queue_size = int(self.get_parameter('queue_size').value)
        self.confidence = float(self.get_parameter('yolo_confidence').value)
        self.yolo_imgsz = int(self.get_parameter('yolo_imgsz').value)
        self.out_width = int(self.get_parameter('out_width').value)
        self.out_height = int(self.get_parameter('out_height').value)

        self.publisher_ = self.create_publisher(Image, self.output_topic, self.queue_size)
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            self.queue_size)
        
        self.bridge = CvBridge()
        self.segmentation_buffer = None

        try:
            # モデルは package の share/resource から読み込む（必要ならパラメータで上書き）
            share_dir = get_package_share_directory('ped_road_seg_pkg')
            default_model_path = os.path.join(share_dir, 'resource', 'best_model_house2.pth')
            default_yolo_path = os.path.join(share_dir, 'resource', 'yolo26s-seg_pedflow2cls.pt')

            self.declare_parameter('model_path', default_model_path)
            self.declare_parameter('yolo_path', default_yolo_path)

            model_path = self.get_parameter('model_path').value
            yolo_path = self.get_parameter('yolo_path').value
            
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.get_logger().info(f"Device: {self.device}")
            
            # 走行可能領域セグメンテーションモデル
            self.model = torch.load(model_path, map_location=torch.device(self.device), weights_only=False)
            self.model.eval()
            self.get_logger().info(f"Drivable area segmentation model loaded from {model_path} onto {self.device}")

            # YOLOセグメンテーションモデル（歩行者seg + 向き推定）
            self.yolo_model = YOLO(yolo_path).to(self.device)
            if self.device == 'cuda':
                self.yolo_model = self.yolo_model.half()
                self.get_logger().info("YOLO half precision enabled")
            self.get_logger().info(f"YOLO model loaded from {yolo_path}")

            # smp: segmentation_models_pytorch（セグメンテーションモデル/encoderの前処理ユーティリティ）
            # 学習時に使った encoder の前処理（正規化）を推論時にも揃えて、精度劣化を防ぐ
            encoder = 'resnet50'
            encoder_weights = 'imagenet'
            self.preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
            self.augmentation = get_validation_augmentation()
            self.preprocessing = get_preprocessing(self.preprocessing_fn)

        except Exception as e:
            self.get_logger().error(f"Failed to load model or setup preprocessing: {e}")
            raise e

    def image_callback(self, msg):
        # 画像受信 -> (走行可能領域seg + 歩行者seg) -> 色分け画像を publish
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        h, w = frame.shape[:2]

        if self.segmentation_buffer is None or self.segmentation_buffer.shape[:2] != (h, w):
            # 毎フレーム確保しないためのバッファ（入力解像度が変わったら作り直す）
            self.segmentation_buffer = np.empty((h, w, 3), dtype=np.uint8)

        # === 1. 走行可能領域セグメンテーション ===
        try:
            # OpenCV画像はBGRなのでRGBへ変換（学習時の前提に合わせる）
            # PadIfNeededで最小サイズを満たす（足りない分だけ外周に余白を追加）
            # -> 正規化（例: ImageNet mean/std）-> (C,H,W) へ変換
            # その後、PyTorchの入力 (N,C,H,W) にして推論し、しきい値で走行可能領域マスクを作る
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sample = self.augmentation(image=image)
            image_aug = sample['image']
            sample = self.preprocessing(image=image_aug)
            image_preproc = sample['image']

            x_tensor = torch.from_numpy(image_preproc).to(self.device).unsqueeze(0)

            with torch.no_grad():
                pr_mask_tensor = self.model(x_tensor)

            pr_mask_np = pr_mask_tensor.squeeze().cpu().numpy()
            drivable_mask = (pr_mask_np > 0.5).astype(np.uint8)  # ２値化

            # PadIfNeeded により推論サイズが拡張された場合は元の解像度に戻す
            if drivable_mask.shape[:2] != (h, w):
                pad_h = drivable_mask.shape[0] - h
                pad_w = drivable_mask.shape[1] - w
                # 既定の center padding を前提に中央を切り出す
                start_y = max(0, pad_h // 2)
                start_x = max(0, pad_w // 2)
                drivable_mask = drivable_mask[start_y:start_y + h, start_x:start_x + w]
        except Exception as e:
            self.get_logger().error(f"Drivable area segmentation failed: {e}")
            return

        # === 2. YOLOで歩行者seg（同方向/逆方向の2クラス） ===
        same_dir_mask = np.zeros((h, w), dtype=np.uint8)
        ops_dir_mask = np.zeros((h, w), dtype=np.uint8)

        try:
            with torch.inference_mode():
                yolo_results = self.yolo_model.predict(
                    frame,
                    imgsz=self.yolo_imgsz,
                    conf=self.confidence,
                    retina_masks=True,
                    stream=False,
                )[0]

            if yolo_results.boxes is not None and yolo_results.masks is not None:
                # YOLOのマスクは入力imgsz由来の解像度なので、元画像サイズに合わせて合成
                masks = (yolo_results.masks.data > 0.5).cpu().numpy().astype(np.uint8)
                class_ids = yolo_results.boxes.cls.cpu().numpy().astype(int)

                for idx, mask_pred in enumerate(masks):
                    # マスクを元画像サイズにリサイズ
                    if mask_pred.shape[:2] != (h, w):
                        mask_pred = cv2.resize(mask_pred, (w, h), interpolation=cv2.INTER_NEAREST)

                    cls_id = class_ids[idx] if idx < len(class_ids) else None
                    if cls_id == 0:  # 同方向
                        same_dir_mask = np.maximum(same_dir_mask, mask_pred)
                    elif cls_id == 1:  # 同方向以外
                        ops_dir_mask = np.maximum(ops_dir_mask, mask_pred)

        except Exception as e:
            self.get_logger().warn(f"YOLO inference failed: {e}")

        # === 3. 四値化画像作成 ===
        segmentation = self.segmentation_buffer

        # 色定義(BGR)
        road_color = (0, 255, 0)          # 緑（走行可能領域）
        same_dir_color = (255, 0, 0)      # 青（同方向歩行者）
        ops_dir_color = (0, 0, 255)       # 赤（同方向以外歩行者）
        other_color = (255, 255, 0)       # シアン（その他）

        # 優先順位(低→高): その他 < 走行可能領域 < 同方向 < 逆方向（歩行者が最優先で上書き）
        segmentation[:] = other_color
        segmentation[drivable_mask == 1] = road_color
        segmentation[same_dir_mask == 1] = same_dir_color
        segmentation[ops_dir_mask == 1] = ops_dir_color

        # 行動モデルに合わせた解像度にリサイズ
        img_out = cv2.resize(
            segmentation,
            (self.out_width, self.out_height),
            interpolation=cv2.INTER_NEAREST,
        )

        try:
            out_msg = self.bridge.cv2_to_imgmsg(img_out, "bgr8")
            self.publisher_.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f"Publish failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    img_segmentation_node = ImgSegmentationNode()
    rclpy.spin(img_segmentation_node)
    img_segmentation_node.destroy_node()
    rclpy.shutdown()
