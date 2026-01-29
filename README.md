# Ros2_WorkSpaces

開発した ROS 2 ワークスペース/パラメータ類をまとめて管理するリポジトリ

## 構成

- `unity-inference-ws/` : Unity 環境で ROS2-For-Unity を使用して推論を行うワークスペース
- `real-nav-ws/` : 実機へモデルを適用して Navigation するためのワークスペース
- `slam-params/` : nav2 / SLAM Toolbox 等のパラメータファイル置き場

## real-nav-ws

### 関連パッケージ

- `ped_road_seg_pkg`
  - セマンティックセグメンテーション + fine-tuning した YOLO-seg を用いて、推論結果を 4値に整理

### 4値化のクラス・色

| クラス | 色(BGR) |
| --- | --- |
| 走行可能領域（床） | (0, 255, 0) <span style="display:inline-block;width:0.9em;height:0.9em;background:#00ff00;border:1px solid #999;vertical-align:middle"></span> |
| 同方向歩行者（class 0 / same-dir） | (255, 0, 0) <span style="display:inline-block;width:0.9em;height:0.9em;background:#0000ff;border:1px solid #999;vertical-align:middle"></span> |
| 同方向以外歩行者（class 1 / ops-dir） | (0, 0, 255) <span style="display:inline-block;width:0.9em;height:0.9em;background:#ff0000;border:1px solid #999;vertical-align:middle"></span> |
| その他 | (255, 255, 0) <span style="display:inline-block;width:0.9em;height:0.9em;background:#00ffff;border:1px solid #999;vertical-align:middle"></span> |

## unity-inference-ws

Unity 環境で ROS2-For-Unity を使用し、推論結果を ROS 2 側へ流すためのワークスペース。

## slam-params

nav2 / map_server / AMCL / SLAM Toolbox などのパラメータファイルを配置する。
