# real-nav-ws

行動モデルを実機に適用して目的地まで自律走行させるためのワークスペース

## ped_road_seg_pkg

### 概要

- カメラ画像から「走行可能領域（床）」と「歩行者（同方向/同方向以外）」を検出し、4値化セグメンテーション画像を出力する。
- 学習環境を4値化し、実環境でも本パッケージで4値化した画像を行動モデルに入力することで、sim2realの視覚ギャップを軽減する。

使用モデル：

- セマンティックセグメンテーション（走行可能領域）
  - 画像をピクセル単位で分類し、走行可能領域（床）マスクを推定する。
  - 実装上は PyTorch のセマンティックセグメンテーションモデル（`best_model_house2.pth`）で床マスクを生成する。
- YOLO-seg（歩行者 + 向きカテゴリ）
  - [yolo26s-seg](https://docs.ultralytics.com/ja/models/yolo26/)ベースでファインチューニングした2クラスのインスタンスセグメンテーションモデル（`yolo26s-seg_pedflow2cls.pt`）。
  - 歩行者領域を検出し、ピクセル単位のマスクを推定する。
  - クラス定義（実装の `class 0/1`）
    - class 0: 同方向歩行者（カメラとの相対角度が 45° 以内）
    - class 1: 同方向以外歩行者

出力（`/gb_img`）の色（BGR）は以下。

| クラス | 色(BGR) |
| --- | --- |
| 走行可能領域（床） | 緑（0, 255, 0） |
| 同方向歩行者 | 青（255, 0, 0） |
| 同方向以外歩行者 | 赤（0, 0, 255） |
| その他 | シアン（255, 255, 0） |

### 開発環境

- ROS2 jazzy
- Python 3.12.3
- CUDA Version: 13.0   
- Python依存は [src/ped_road_seg_pkg/requirements.txt](src/ped_road_seg_pkg/requirements.txt) を参照

モデルファイル（要配置）：

- `resource/best_model_house2.pth`
- `resource/yolo26s-seg_pedflow2cls.pt`

モデルは `src/ped_road_seg_pkg/resource/` に配置する。

### 使用方法

#### 1) セットアップ & ビルド

```bash
cd ~/ros2-workspaces/real-nav-ws

python3 -m venv .venv --system-site-packages
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r src/ped_road_seg_pkg/requirements.txt

python -m colcon build --packages-select ped_road_seg_pkg --symlink-install
source install/setup.bash
```

#### 2) 起動

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
cd ~/ros2-workspaces/real-nav-ws
source .venv/bin/activate
source install/setup.bash
ros2 run ped_road_seg_pkg img_segmentation_node
```

トピック：

- Subscribe: `/image_raw`（sensor_msgs/Image, BGR8, 640×480想定）
- Publish: `/gb_img`（sensor_msgs/Image, BGR8, 112×84）

### 処理フロー

1. `/image_raw` を受信
2. 床セグメンテーション（PyTorchモデルで床マスク生成）
3. 歩行者seg（YOLO-segで class 0/1 のマスク生成、複数人はクラス別に統合）
4. 4値化画像を作成（優先度: その他 → 床 → class 0 → class 1）
5. 行動モデルの入力画像サイズにリサイズして `/gb_img` を publish
