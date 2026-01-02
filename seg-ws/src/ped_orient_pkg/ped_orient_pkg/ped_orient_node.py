#!/usr/bin/env python3
"""
ROS2ノード: YOLOv8-seg + MEBOWで歩行者の向きを推定
ダウンスケール版（デフォルト112x84）の完全セグメンテーション画像を配信します。
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO

from ament_index_python.packages import get_package_share_path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ---- パス設定 ----
PKG_NAME = "ped_orient_pkg"
PKG_DIR = Path(__file__).resolve().parent
# パッケージ共有フォルダ取得。インストール時はshare、開発時はパッケージ直下を利用
try:
    SHARE_DIR = get_package_share_path(PKG_NAME)
except Exception:
    SHARE_DIR = PKG_DIR  # インストール不備時のフォールバック

_mebow_root_candidate = SHARE_DIR / "MEBOW"
_yolo_weights_candidate = SHARE_DIR / "yolov8n-seg.pt"

MEBOW_ROOT = _mebow_root_candidate if _mebow_root_candidate.exists() else PKG_DIR / "MEBOW"
YOLO_WEIGHTS = _yolo_weights_candidate if _yolo_weights_candidate.exists() else PKG_DIR / "yolov8n-seg.pt"
MEBOW_LIB_PATH = MEBOW_ROOT / "lib"
# MEBOWのlibをimport可能にする
if str(MEBOW_LIB_PATH) not in sys.path:
    sys.path.insert(0, str(MEBOW_LIB_PATH))

try:
    from config import cfg as mebow_cfg
    from config import update_config as mebow_update_config
    import models as mebow_models
except ImportError as e:  # pragma: no cover - インポート時のガード
    # MEBOWの配置ミス時は明示的に失敗させる
    raise ImportError(
        f"MEBOWモジュールのインポートに失敗しました ({MEBOW_LIB_PATH})。"
        "パッケージ内にMEBOWディレクトリが存在することを確認してください。"
    ) from e

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COLOR = {"front": (0, 0, 255), "back": (255, 0, 0)}  # front: red (BGR), back: blue (BGR)
BACKGROUND_COLOR = (255, 255, 0)  # cyan (BGR) for non-person regions - RGB: (0, 255, 255)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
mebow_transform = transforms.Compose([transforms.ToTensor(), normalize])


def quantize_angle(angle_deg: float) -> str:
    """連続的な角度を2方向に量子化
    - 45°〜225°をfront（赤）扱い
    - それ以外をback（青）扱い
    """
    if 45 <= angle_deg < 225:
        return "front"
    else:
        return "back"


class PedOrientNode(Node):
    def __init__(self):
        super().__init__("ped_orient_node")
        self.bridge = CvBridge()
        self.processing = False  # 単純な排他制御（フレーム落ち防止）

        # パラメータ設定: 入力トピックのみ。出力とサイズはシンプルさのため固定
        self.declare_parameter("input_image_topic", "/img")
        self.declare_parameter("invert_orientation", False)
        self.input_topic = str(self.get_parameter("input_image_topic").value)
        self.invert_orientation = bool(self.get_parameter("invert_orientation").value)
        self.output_topic = "/ped_orient/segmentation"
        self.target_size = (112, 84)
        self.confidence = 0.7

        self.get_logger().info(f"Using device: {DEVICE}")
        # MEBOWモデルのロード（HRNetベース）。入力サイズは設定から取得
        self.mebow_model = self._load_mebow()
        width, height = mebow_cfg.MODEL.IMAGE_SIZE
        self.mebow_input_size = (int(width), int(height))  # cv2.resizeは(w,h)
        # YOLOv8セグメンテーションモデルのロード（人物クラスのみ利用）
        self.seg_model = YOLO(str(YOLO_WEIGHTS)).to(DEVICE)
        if DEVICE == "cuda":
            self.seg_model = self.seg_model.half()  # GPU時は半精度で高速化

        # ROSインターフェース
        self.sub_img = self.create_subscription(Image, self.input_topic, self.on_image, qos_profile_sensor_data)
        self.pub_segmentation = self.create_publisher(Image, self.output_topic, 10)

        self.get_logger().info(
            f"Subscribed to {self.input_topic}; publishing segmentation->{self.output_topic}"
        )

    def _load_mebow(self):
        # MEBOW設定と重みの適用
        cfg_path = MEBOW_ROOT / "experiments/coco/segm-4_lr1e-3.yaml"
        model_path = MEBOW_ROOT / "models/model_hboe.pth"
        args = SimpleNamespace(cfg=str(cfg_path), opts=[], modelDir="", logDir="", dataDir="")
        mebow_update_config(mebow_cfg, args)  # YAMLの読み込みとcfgへの反映
        model = mebow_models.pose_hrnet.get_pose_net(mebow_cfg, is_train=False)
        if not model_path.exists():
            raise FileNotFoundError(f"MEBOW model not found at {model_path}")
        self.get_logger().info(f"Loading MEBOW weights: {model_path}")
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state, strict=False)  # 互換性のためstrict=False
        return model.to(DEVICE).eval()

    def on_image(self, msg: Image):
        # 画像1枚に対する推論パイプライン
        if self.processing:
            return
        self.processing = True
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 1) YOLOで人物セグメンテーション取得（マスク+バウンディングボックス）
            try:
                with torch.inference_mode():
                    seg_res = self.seg_model.predict(
                        frame, imgsz=640, conf=self.confidence, classes=[0], retina_masks=True, stream=False
                    )[0]
            except Exception as e:
                self.get_logger().error(f"YOLO推論に失敗: {e}")
                return
            if seg_res.boxes is None or seg_res.masks is None:
                return
            b_seg = seg_res.boxes.xyxy.cpu().numpy()
            masks = (seg_res.masks.data > 0.5).cpu().numpy().astype(np.uint8)
            if masks.size == 0:
                return

            # 2) 出力画像（背景はシアン）を初期化
            segmentation = np.full_like(frame, BACKGROUND_COLOR)

            # 3) 各人物領域をMEBOW入力サイズへリサイズ→バッチ推論
            person_inputs = []
            valid_masks = []
            valid_boxes = []
            for bs, mask_pred in zip(b_seg, masks):
                x1, y1, x2, y2 = map(int, bs)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x1 >= x2 or y1 >= y2:
                    continue
                person_img = frame_rgb[y1:y2, x1:x2]
                person_img_resized = cv2.resize(person_img, self.mebow_input_size, interpolation=cv2.INTER_LINEAR)
                person_inputs.append(mebow_transform(person_img_resized))
                valid_masks.append(mask_pred)
                valid_boxes.append((x1, y1, x2, y2))

            if not person_inputs:
                return

            batch_input_tensor = torch.stack(person_inputs).to(DEVICE)
            with torch.no_grad():
                _, batch_hoe_output = self.mebow_model(batch_input_tensor)  # HRNet出力(2D) + 方向分類(72bin)

            # 4) 方向をfront/backへ量子化し、マスク領域へ着色
            for i, hoe_output in enumerate(batch_hoe_output):
                x1, y1, x2, y2 = valid_boxes[i]
                mask_pred = valid_masks[i]

                pred_idx = torch.argmax(hoe_output)  # 5°刻みのbin
                angle_pred = pred_idx.item() * 5
                direction = quantize_angle(angle_pred)  # front/backへ簡易マッピング
                if self.invert_orientation:
                    direction = "front" if direction == "back" else "back"
                col = COLOR[direction]

                # マスク解像度を元画像へ合わせて塗りつぶし
                if mask_pred.shape[:2] != (h, w):
                    mask_pred = cv2.resize(mask_pred, (w, h), interpolation=cv2.INTER_NEAREST)
                segmentation[mask_pred == 1] = col

            # 5) 希望サイズへダウンスケールし、BGR8でpublish
            try:
                resized = cv2.resize(segmentation, self.target_size, interpolation=cv2.INTER_AREA)
                msg_seg = self.bridge.cv2_to_imgmsg(resized, encoding="bgr8")
                msg_seg.header = msg.header
                self.pub_segmentation.publish(msg_seg)
            except Exception as e:
                self.get_logger().error(f"セグメンテーション画像の配信に失敗: {e}")
        except Exception as e:
            self.get_logger().error(f"画像処理中に予期しない例外: {e}")
        finally:
            self.processing = False


def main(args=None):
    rclpy.init(args=args)
    node = PedOrientNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
