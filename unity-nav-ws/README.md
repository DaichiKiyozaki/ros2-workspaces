# unity-nav-ws

## 概要
- Unity 上のエージェントを ROS2 ノードで推論・制御するワークスペース
- [ros2-for-unity](https://github.com/RobotecAI/ros2-for-unity) により Unity と ROS2 間で通信
- ROS2 ノード上で ONNX モデル推論を実行
- 自己位置推定にamclを用いるため、環境地図を事前に用意する必要がある

## セットアップ

1. 環境地図とrvizの設定ファイルを追加
   - 環境地図
      - 配置先： `unity-nav-ws/src/model_in_ros2node_pkg/map`
      - .yamlと.pgmのセット
      - 推論環境の地図は launch 呼び出し時に指定、または launch ファイルのデフォルト値を変更
   - rviz設定ファイル
      - 配置先： `unity-nav-ws/src/model_in_ros2node_pkg/rviz`

2. 行動モデル (ONNX) を追加
   - 配置先： `unity-nav-ws/src/model_in_ros2node_pkg/models`
   - 例: `unity-nav-ws/src/model_in_ros2node_pkg/models/balance.onnx`
   - `agent_node` は `share/model_in_ros2node_pkg/models` から読み込むため、配置後にビルドが必要

3. ワークスペースのビルド
   ```bash
   cd ~/ros2-workspaces/unity-nav-ws
   colcon build --symlink-install
   ```

4. オーバーレイ
   ```bash
   source install/setup.bash
   ```

## 使い方

### 1. Unityの再生

### 2. AMCL + Map Server の起動
`unity_amcl.launch.py` で地図と自己位置推定を起動し、`/amcl_pose` を publish する。

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

### 3. rviz2で初期位置・目標位置を設定

- 初期位置：2D Pose Estimate
- 目標位置：2D Goal Pose

### 4. ノードの起動
推論エージェントノードを起動する。

```bash
ros2 run model_in_ros2node_pkg agent_node
```

モデルに応じた主な起動パラメータ例:

```bash
ros2 run model_in_ros2node_pkg agent_node --ros-args \
   -p model_file_name:=balance.onnx \
   -p stack_size:=5 \
   -p action_output_name:=continuous_actions
```

### 通信仕様 (ROS Topics)

#### Subscribed Topics (入力)
| トピック名 | 型 | 説明 |
| --- | --- | --- |
| `/unity/camera/image_raw` | `sensor_msgs/Image` | エージェントの視覚情報 (RGB)。ノード内で `img_width` x `img_height` にリサイズ（デフォルト: 112x84）。 |
| `/goal_pose` | `geometry_msgs/PoseStamped` | RViz2 の 2D Nav Goal。ゴール位置として使用。 |
| `/amcl_pose` | `geometry_msgs/PoseWithCovarianceStamped` | AMCL 推定の自己位置。ゴール相対角の計算に使用。 |

#### Published Topics (出力)
| トピック名 | 型 | 説明 |
| --- | --- | --- |
| `/agent/cmd` | `std_msgs/Float32MultiArray` | 推論された行動コマンド (continuous)。 |
| `/debug/stacked_image` | `sensor_msgs/Image` | デバッグ用。スタックしたフレームを横並びで可視化。 |

## Unity側の設定
Unity 側は `ros2-for-unity` を使用し、以下を publish 対象とする。

1. **カメラ画像**: RGB 形式。解像度は **112x84** を推奨。
2. **自己位置とゴール情報**: `/amcl_pose` と `/goal_pose` は ROS 側（AMCL/RViz2 など）で用意。
3. **TF/座標系**: 整合性が取れた`map`/`odom`/`base_link` をpublishする必要がある。

## パラメータ
| 名前 | デフォルト | 説明 |
| --- | --- | --- |
| `model_file_name` | `balance.onnx` | `share/model_in_ros2node_pkg/models` 配下から読み込む ONNX ファイル名。モデル差し替え時に指定。 |
| `action_output_name` | `""` | アクション出力に使う ONNX output 名。空の場合は `continuous_actions` → `deterministic_continuous_actions` の順で自動選択。※discrete系outputは未対応。 |
| `img_width` | `112` | 入力画像のリサイズ幅。モデルの入力形状に合わせる必要がある。 |
| `img_height` | `84` | 入力画像のリサイズ高さ。モデルの入力形状に合わせる必要がある。 |
| `stack_size` | `5` | 連続フレームのスタック数。入力が NCHW の場合はチャネル数が `3*stack_size` になるため、モデルに合わせて設定する。 |
| `vec_obs_dim` | `2` | ベクトル観測の次元数。基本は `[angle_deg, distance_m]`（2次元）で、不足は 0 埋め・超過は切り捨て。モデルの入力形状に合わせて設定する。 |
| `debug` | `true` | デバッグログと `/debug/stacked_image` の publish を有効化。 |
| `log_period_sec` | `1.0` | デバッグログの周期 (秒)。 |
