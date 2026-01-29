# ped_road_seg_pkg

## 概要

カメラ画像から「走行可能領域（床）」と「歩行者（同方向/同方向以外）」を検出し、4値化したセグメンテーション画像（112×84, BGR8）を出力する。

出力（/gb_img）の色（BGR）は以下。

| クラス | 色(BGR) |
| --- | --- |
| 走行可能領域（床） | (0, 255, 0) <span style="display:inline-block;width:0.9em;height:0.9em;background:#00ff00;border:1px solid #999;vertical-align:middle"></span> |
| 同方向歩行者（class 0 / same-dir） | (255, 0, 0) <span style="display:inline-block;width:0.9em;height:0.9em;background:#0000ff;border:1px solid #999;vertical-align:middle"></span> |
| 同方向以外歩行者（class 1 / ops-dir） | (0, 0, 255) <span style="display:inline-block;width:0.9em;height:0.9em;background:#ff0000;border:1px solid #999;vertical-align:middle"></span> |
| その他 | (255, 255, 0) <span style="display:inline-block;width:0.9em;height:0.9em;background:#00ffff;border:1px solid #999;vertical-align:middle"></span> |

## 開発環境

- ROS 2
  - 開発環境：jazzy
- Python
  - 開発環境：Python 3.12.3
- CUDA推奨
- Python依存は [src/ped_road_seg_pkg/requirements.txt](src/ped_road_seg_pkg/requirements.txt) を参照

モデルファイル（要配置）：

- `resource/best_model_house2.pth`
- `resource/yolo26s-seg_pedflow2cls.pt`

モデルは `src/ped_road_seg_pkg/resource/` に配置する。

## 使用方法

### 1) セットアップ & ビルド

```bash
cd /home/daichi-kiyozaki/ros_pj/real-nav-ws

python3 -m venv .venv --system-site-packages
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r src/ped_road_seg_pkg/requirements.txt

python -m colcon build --packages-select ped_road_seg_pkg --symlink-install
source install/setup.bash
```

### 2) 起動

カメラ（例）：

```bash
ros2 run v4l2_camera v4l2_camera_node \
  --ros-args \
    -p video_device:="/dev/video0" \
    -p image_size:="[640,480]" \
    -p pixel_format:="YUYV"
```

セグメンテーション：

```bash
cd /home/daichi-kiyozaki/ros_pj/real-nav-ws
source .venv/bin/activate
source install/setup.bash
ros2 run ped_road_seg_pkg img_segmentation_node
```

トピック：

- Subscribe: `/image_raw`（sensor_msgs/Image, BGR8, 640×480想定）
- Publish: `/gb_img`（sensor_msgs/Image, BGR8, 112×84）

## 処理フロー

1. `/image_raw` を受信
2. 床セグメンテーション（PyTorchモデルで床マスク生成）
3. 歩行者seg（YOLO-segで class 0/1 のマスク生成、複数人はクラス別に統合）
4. 4値化画像を作成（優先度: その他 → 床 → class 0 → class 1）
5. 112×84 にリサイズして `/gb_img` を publish
