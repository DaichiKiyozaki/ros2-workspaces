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
from ultralytics import YOLO
import torchvision.transforms as transforms

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ---- パス設定 ----
PKG_DIR = Path(__file__).resolve().parent
MEBOW_ROOT = PKG_DIR / "MEBOW"
MEBOW_LIB_PATH = MEBOW_ROOT / "lib"
YOLO_WEIGHTS = PKG_DIR / "yolov8n-seg.pt"

if str(MEBOW_LIB_PATH) not in sys.path:
    sys.path.insert(0, str(MEBOW_LIB_PATH))

try:
    from config import cfg as mebow_cfg
    from config import update_config as mebow_update_config
    import models as mebow_models
except ImportError as e:  # pragma: no cover - インポート時のガード
    raise ImportError(
        f"MEBOWモジュールのインポートに失敗しました ({MEBOW_LIB_PATH})。"
        "パッケージ内にMEBOWディレクトリが存在することを確認してください。"
    ) from e

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIR2 = ["front", "back"]
COLOR = {"front": (0, 0, 255), "back": (255, 0, 0)}  # front: red (BGR), back: blue (BGR)
BACKGROUND_COLOR = (255, 255, 0)  # cyan (BGR) for non-person regions - RGB: (0, 255, 255)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
mebow_transform = transforms.Compose([transforms.ToTensor(), normalize])


def quantize_angle(angle_deg: float) -> str:
    """連続的な角度を2方向に量子化"""
    if 45 <= angle_deg < 225:
        return "front"
    else:
        return "back"


class PedOrientNode(Node):
    def __init__(self):
        super().__init__("ped_orient_node")
        self.bridge = CvBridge()
        self.processing = False

        # パラメータ設定: 入力トピックのみ。出力とサイズはシンプルさのため固定
        self.declare_parameter("input_image_topic", "/img")
        self.input_topic = str(self.get_parameter("input_image_topic").value)
        self.output_topic = "/ped_orient/segmentation"
        self.target_size = (112, 84)
        self.confidence = 0.7

        self.get_logger().info(f"Using device: {DEVICE}")
        self.mebow_model = self._load_mebow()
        self.mebow_input_size = (192, 256)

        self.seg_model = YOLO(str(YOLO_WEIGHTS)).to(DEVICE)
        if DEVICE == "cuda":
            self.seg_model = self.seg_model.half()
            torch.cuda.Stream()  # ウォームアップ

        # ROSインターフェース
        self.sub_img = self.create_subscription(Image, self.input_topic, self.on_image, qos_profile_sensor_data)
        self.pub_segmentation = self.create_publisher(Image, self.output_topic, 10)

        self.get_logger().info(
            f"Subscribed to {self.input_topic}; publishing segmentation->{self.output_topic}"
        )

    def _load_mebow(self):
        cfg_path = MEBOW_ROOT / "experiments/coco/segm-4_lr1e-3.yaml"
        model_path = MEBOW_ROOT / "models/model_hboe.pth"
        args = SimpleNamespace(cfg=str(cfg_path), opts=[], modelDir="", logDir="", dataDir="")
        mebow_update_config(mebow_cfg, args)
        model = mebow_models.pose_hrnet.get_pose_net(mebow_cfg, is_train=False)
        if not model_path.exists():
            raise FileNotFoundError(f"MEBOW model not found at {model_path}")
        self.get_logger().info(f"Loading MEBOW weights: {model_path}")
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        return model.to(DEVICE).eval()

    def on_image(self, msg: Image):
        if self.processing:
            return
        self.processing = True
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:  # pragma: no cover - 実行時のガード
            self.get_logger().error(f"画像変換に失敗: {e}")
            self.processing = False
            return

        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            with torch.inference_mode():
                seg_res = self.seg_model.predict(
                    frame, imgsz=640, conf=self.confidence, classes=[0], stream=False
                )[0]
        except Exception as e:
            self.get_logger().error(f"YOLO推論に失敗: {e}")
            self.processing = False
            return

        if seg_res.boxes is None or seg_res.masks is None:
            self.processing = False
            return

        b_seg = seg_res.boxes.xyxy.cpu().numpy()
        masks = seg_res.masks.data.byte().cpu().numpy()
        if masks.size == 0:
            self.processing = False
            return

        # 完全セグメンテーション画像を作成（人以外の領域はシアン）
        segmentation = np.full_like(frame, BACKGROUND_COLOR)

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
            self.processing = False
            return

        batch_input_tensor = torch.stack(person_inputs).to(DEVICE)
        with torch.no_grad():
            _, batch_hoe_output = self.mebow_model(batch_input_tensor)

        for i, hoe_output in enumerate(batch_hoe_output):
            x1, y1, x2, y2 = valid_boxes[i]
            mask_pred = valid_masks[i]

            pred_idx = torch.argmax(hoe_output)
            angle_pred = pred_idx.item() * 5
            direction = quantize_angle(angle_pred)
            col = COLOR[direction]

            # セグメンテーション画像に色を塗る（透過なし）
            mask_resized = cv2.resize(mask_pred, (w, h), interpolation=cv2.INTER_NEAREST)
            segmentation[mask_resized == 1] = col

        # ダウンサイズして配信
        try:
            resized = cv2.resize(segmentation, self.target_size, interpolation=cv2.INTER_AREA)
            msg_seg = self.bridge.cv2_to_imgmsg(resized, encoding="bgr8")
            msg_seg.header = msg.header
            self.pub_segmentation.publish(msg_seg)
        except Exception as e:
            self.get_logger().error(f"セグメンテーション画像の配信に失敗: {e}")

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
