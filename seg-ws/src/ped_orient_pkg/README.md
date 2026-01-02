# ped_orient_pkg

YOLOv8セグメンテーション + MEBOWによる歩行者方向推定

## 概要

カメラ画像から歩行者を検出し、その向き（front・backの2方向）を推定して完全セグメンテーション画像として出力するROS2ノード。

- **YOLOv8-seg**: 人物検出・セグメンテーション
    -
- **MEBOW**: 人物向き推定モデル
  - <https://github.com/ChenyanWu/MEBOW>
- **出力**: ダウンサイズ版（112×84）の完全セグメンテーション画像

### 色と向きの対応

- **Front (前)**: <span style="color:red">**Red (赤)**</span> - BGR: (0, 0, 255)
- **Back (後)**: <span style="color:blue">**Blue (青)**</span> - BGR: (255, 0, 0)
- **背景 (人以外の領域)**: **Cyan (シアン)** - RGB: (0, 255, 255) / BGR: (255, 255, 0)

## 環境構築

### 必要なファイル

```
ped_orient_pkg/
├── MEBOW/                       # MEBOWモデル（別途配置）
│   ├── lib/
│   ├── experiments/coco/segm-4_lr1e-3.yaml
│   └── models/model_hboe.pth
└── yolov8n-seg.pt              # YOLOv8モデル
```

### インストール

```bash
# 仮想環境を作成・有効化（まだの場合）
cd /home/daichi-kiyozaki/ros_pj/seg-ws
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# 依存パッケージをインストール
pip install -r src/ped_orient_pkg/requirements.txt

# ビルド
colcon build --packages-select ped_orient_pkg --symlink-install
```

## 使い方

### 生画像をpublish

- 使用するカメラを指定する必要あり

```bash
ros2 run v4l2_camera v4l2_camera_node \
  --ros-args \
    -p video_device:="/dev/video0" \
    -p image_size:="[640,480]" \
    -p pixel_format:="YUYV"
```


### ノード起動

```bash
cd /home/daichi-kiyozaki/ros_pj/seg-ws
source .venv/bin/activate
source install/setup.bash
export PYTHONPATH="${PWD}/.venv/lib/python3.12/site-packages:${PYTHONPATH}"
ros2 run ped_orient_pkg ped_orient_node
```

### パラメータ

- `input_image_topic` (デフォルト: `/img`) - 入力画像トピック

例：
```bash
ros2 run ped_orient_pkg ped_orient_node --ros-args -p input_image_topic:=/image_raw
```

### トピック

**subscribe:**
- `<input_image_topic>` (`sensor_msgs/Image`) - 入力カメラ画像

**publish:**
- `/ped_orient/segmentation` (`sensor_msgs/Image`) - ダウンサイズ版（112×84）の完全セグメンテーション画像
