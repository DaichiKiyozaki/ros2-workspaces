# Unity-inference-ws

## 概要
Unity上のエージェントをROS 2ノードで推論・制御するためのワークスペースです。
`ros2-for-unity` を使用してUnityとROS 2間で通信を行い、ROS 2ノード上でONNXモデルを用いた推論を実行します。

## 開発環境
- Ros2 jazzy
- python 3.12

## セットアップ

1. ワークスペースのビルド
   ```bash
   cd ~/ros_pj/unity-inference-ws
   colcon build --symlink-install
   ```

2. オーバーレイ
   ```bash
   source install/setup.bash
   ```

## 使い方

### AMCL + Map Server の起動
`unity_amcl.launch.py` で地図と自己位置推定を起動します。`/amcl_pose` を発行します。

```bash
ros2 launch model_in_ros2node_pkg unity_amcl.launch.py
```

主な引数:
- `use_sim_time` (default: `true`)
- `map`
- `rviz` (default: `true`)
- `rviz_config`
- `global_frame_id` (default: `map`)
- `odom_frame_id` (default: `odom`)
- `base_frame_id` (default: `base_link`)
- `scan_topic` (default: `/scan`)

### ノードの起動
推論エージェントノードを起動します。

```bash
ros2 run model_in_ros2node_pkg agent_node
```

### 通信仕様 (ROS Topics)

#### Subscribed Topics (入力)
| トピック名 | 型 | 説明 |
| --- | --- | --- |
| `/unity/camera/image_raw` | `sensor_msgs/Image` | エージェントの視覚情報 (RGB)。ノード内で **112x84** にリサイズされます。 |
| `/goal_pose` | `geometry_msgs/PoseStamped` | RViz2 の 2D Nav Goal。ゴール位置として使用します。 |
| `/amcl_pose` | `geometry_msgs/PoseWithCovarianceStamped` | AMCL 推定の自己位置。ゴール相対角の計算に使用します。 |

#### Published Topics (出力)
| トピック名 | 型 | 説明 |
| --- | --- | --- |
| `/agent/cmd` | `std_msgs/Float32MultiArray` | 推論された行動コマンド (連続値)。 |
| `/debug/stacked_image` | `sensor_msgs/Image` | デバッグ用。スタックしたフレームを横並びで可視化します。 |

## Unity側の設定
Unity側では `ros2-for-unity` を使用して、以下のデータをPublishしてください。

1. **カメラ画像**: RGB形式。解像度は **112x84** を推奨します。
2. **自己位置とゴール情報**: `/amcl_pose` と `/goal_pose` は ROS 側（AMCL/RViz2 など）で用意してください。
3. **TF/座標系**: `map`/`odom`/`base_link` が一貫していることを確認してください。

## パラメータ
| 名前 | デフォルト | 説明 |
| --- | --- | --- |
| `debug` | `true` | デバッグログと `/debug/stacked_image` の publish を有効化します。 |
| `log_period_sec` | `1.0` | デバッグログの周期 (秒) です。 |

## ディレクトリ構造
- `src/model_in_ros2node_pkg`: 推論ノードのパッケージ
  - `model_in_ros2node_pkg/agent_node.py`: 推論ロジックのメインスクリプト

