# Unity-inference-ws

## 概要
Unity上のエージェントをROS 2ノードで推論・制御するためのワークスペースです。
`ros2-for-unity` を使用してUnityとROS 2間で通信を行い、ROS 2ノード上でONNXモデルを用いた推論を実行します。

## 必要要件
- ROS 2 (Humble 推奨)
- Unity (ros2-for-unity 導入済み)
- Python ライブラリ:
  - `onnxruntime`
  - `numpy`
  - `opencv-python`

## セットアップ

1. ワークスペースのビルド
   ```bash
   cd ~/ros_pj/unity-inference_ws
   colcon build --symlink-install
   ```

2. オーバーレイ
   ```bash
   source install/setup.bash
   ```

## 使い方

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
| `/agent/vector_obs` | `std_msgs/Float32MultiArray` | ベクトル観測情報。現在は **[距離, 角度]** の2次元データを期待しています。 |

#### Published Topics (出力)
| トピック名 | 型 | 説明 |
| --- | --- | --- |
| `/agent/cmd` | `std_msgs/Float32MultiArray` | 推論された行動コマンド (連続値)。 |

## Unity側の設定
Unity側では `ros2-for-unity` を使用して、以下のデータをPublishしてください。

1. **カメラ画像**: RGB形式。解像度は **112x84** を推奨します。
2. **ベクトル観測**: ターゲットまでの距離と角度を含む `float` 配列。

## ディレクトリ構造
- `src/model_in_ros2node_pkg`: 推論ノードのパッケージ
  - `models/`: ONNXモデルファイル (`.onnx`) を配置
  - `model_in_ros2node_pkg/agent_node.py`: 推論ロジックのメインスクリプト

