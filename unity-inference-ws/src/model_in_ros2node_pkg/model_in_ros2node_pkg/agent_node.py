import rclpy
from rclpy.node import Node
import onnxruntime as ort
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
from ament_index_python.packages import get_package_share_directory
import os

class AgentNode(Node):
    # Constants
    IMG_WIDTH = 112
    IMG_HEIGHT = 84
    VEC_OBS_DIM = 2  # [distance, angle]
    MODEL_FILE_NAME = '1stack.onnx'

    def __init__(self):
        super().__init__('agent_node')

        self.bridge = CvBridge()
        self.goal = np.zeros(2)  # x,z

        # Debug/monitor parameters
        self.declare_parameter('debug', True)
        self.declare_parameter('log_period_sec', 1.0)
        self.debug = self.get_parameter('debug').value
        self.log_period_sec = float(self.get_parameter('log_period_sec').value)
        self._infer_count = 0
        self._last_log_time = self.get_clock().now()
        self._last_action = None

        self.sub_cam = self.create_subscription(Image, '/unity/camera/image_raw', self.cb_cam, 10)

        # self.sub_goal = self.create_subscription(Point, '/agent/goal', self.cb_goal, 10)

        # Optional: vector observations directly from Unity (2 floats expected)
        self.vec_obs = None  # shape (1,7) or None
        self.sub_vec = self.create_subscription(Float32MultiArray, '/agent/vector_obs', self.cb_vec, 10)

        self.pub_act = self.create_publisher(Float32MultiArray, '/agent/cmd', 10)
        
        share_dir = get_package_share_directory('model_in_ros2node_pkg')
        model_path = os.path.join(share_dir, 'models',  self.MODEL_FILE_NAME)

        self.session = ort.InferenceSession(model_path,
                                            providers=['CPUExecutionProvider'])
        # すべての入力名・形状を取得しログに出す（複数入力対応）
        self.inputs = self.session.get_inputs()
        self.input_names = [i.name for i in self.inputs]
        self.input_shapes = [i.shape for i in self.inputs]
        self.get_logger().info(f"ONNX inputs: {list(zip(self.input_names, self.input_shapes))}")

        # 出力情報も記録・ログ出し
        self.outputs = self.session.get_outputs()
        self.output_names = [o.name for o in self.outputs]
        self.output_shapes = [o.shape for o in self.outputs]
        self.get_logger().info(f"ONNX outputs: {list(zip(self.output_names, self.output_shapes))}")

    # def cb_goal(self, msg):
    #     self.goal = np.array([msg.x, msg.z], dtype=np.float32)

    def cb_vec(self, msg: Float32MultiArray):
        try:
            arr = np.array(msg.data, dtype=np.float32).reshape(-1)
            
            # 次元数チェックと調整
            if arr.size < self.VEC_OBS_DIM:
                arr = np.pad(arr, (0, self.VEC_OBS_DIM - arr.size), constant_values=0.0)
            elif arr.size > self.VEC_OBS_DIM:
                arr = arr[:self.VEC_OBS_DIM]
                
            self.vec_obs = arr.reshape(1, self.VEC_OBS_DIM)
        except Exception as e:
            self.get_logger().warn(f"Failed to parse /agent/vector_obs: {e}")

    def cb_cam(self, msg):
        # Option A: カラー（RGB, 3ch）で受け取り、モデル期待の (1,3,H,W) に合わせる
        try:
            img_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')  # (H,W,3), uint8
        except Exception as e:
            self.get_logger().error(f'cv_bridge convert failed: {e}')
            return

        # 学習時の解像度に揃える
        img_rgb = cv2.resize(img_rgb, (self.IMG_WIDTH, self.IMG_HEIGHT), interpolation=cv2.INTER_AREA)

        # 必要なら上下反転（Unity 由来の上下が逆な場合）
        # img_rgb = cv2.flip(img_rgb, 0)

        # [0,1] 正規化
        img_f = img_rgb.astype(np.float32) / 255.0             # (H,W,3)

        # 画像テンソル（NCHW / NHWC の両方を用意）
        img_nchw = np.transpose(img_f, (2, 0, 1))              # (3,H,W)
        img_nchw = np.expand_dims(img_nchw, axis=0)            # (1,3,H,W)
        img_nhwc = np.expand_dims(img_f, axis=0)               # (1,H,W,3)

        # ベクトル観測
        # 優先: Unity から送られるベクトル（距離・角度など）
        if self.vec_obs is not None:
            vec = self.vec_obs.astype(np.float32)              # (1, VEC_OBS_DIM)
        else:
            # フォールバック: データが来ていない場合はゼロ埋め
            vec = np.zeros((1, self.VEC_OBS_DIM), dtype=np.float32)

        # 入力マップを構築（形状から画像orベクトルを自動判別）
        feed = {}
        for name, shape in zip(self.input_names, self.input_shapes):
            # shape が [None, ...] のように可変でも次元数で判定
            if len(shape) == 4:
                # NCHW 期待: shape[1] が 1 or 3 のことが多い / それ以外は NHWC とみなす
                ch = shape[1]
                if isinstance(ch, int) and ch in (1, 3):
                    feed[name] = img_nchw
                else:
                    feed[name] = img_nhwc
            else:
                # 2D ベクトル想定。終端次元に合わせてパディング/トリム
                target_dim = shape[-1] if isinstance(shape[-1], int) else vec.shape[1]
                v = vec
                if isinstance(target_dim, int) and v.shape[1] != target_dim:
                    if v.shape[1] < target_dim:
                        v = np.pad(v, ((0,0),(0, target_dim - v.shape[1])), constant_values=0.0)
                    else:
                        v = v[:, :target_dim]
                feed[name] = v

        try:
            outputs_list = self.session.run(None, feed)
            name_to_out = {name: np.array(val) for name, val in zip(self.output_names, outputs_list)}

            action = None
            chosen_name = None

            # 1) 連続アクションらしい出力名を優先
            for key in name_to_out.keys():
                key_l = key.lower()
                if 'continuous' in key_l and 'action' in key_l:
                    arr = name_to_out[key].astype(np.float32)
                    if arr.ndim >= 2:
                        action = arr.reshape(arr.shape[0], -1)[0]
                        chosen_name = key
                        break

            # 2) 次に 'action' を含み、かつ最終次元が小さめ（2〜16）
            if action is None:
                for key, arr in name_to_out.items():
                    key_l = key.lower()
                    a = np.array(arr)
                    if 'action' in key_l and a.ndim >= 2:
                        last = a.shape[-1]
                        if isinstance(last, int) and 2 <= last <= 16:
                            action = a.astype(np.float32).reshape(a.shape[0], -1)[0]
                            chosen_name = key
                            break

            # 3) 最終手段: 形状から推測（最終次元が2〜8）
            if action is None:
                for key, arr in name_to_out.items():
                    a = np.array(arr)
                    if a.ndim >= 2 and isinstance(a.shape[-1], int) and 2 <= a.shape[-1] <= 8:
                        action = a.astype(np.float32).reshape(a.shape[0], -1)[0]
                        chosen_name = key
                        break

            # 4) それでも見つからない場合、スカラー/離散を2次元行動にデコード
            if action is None:
                # 先頭出力を使用
                first_name = self.output_names[0] if self.output_names else 'out0'
                arr = name_to_out.get(first_name, outputs_list[0])
                val = float(np.array(arr).reshape(-1)[0])
                idx = int(round(val))
                # デフォルトの離散→連続マッピング（必要に応じて調整）
                # 0: 停止, 1: 前進, 2: 左旋回, 3: 右旋回
                mapping = {
                    0: np.array([0.0, 0.0], dtype=np.float32),      # stop
                    1: np.array([1.0, 0.0], dtype=np.float32),      # forward
                    2: np.array([1.0, -1.0], dtype=np.float32),     # forward-left
                    3: np.array([1.0, 1.0], dtype=np.float32),      # forward-right
                }
                action = mapping.get(idx, np.array([1.0, 0.0], dtype=np.float32))
                chosen_name = f"decoded_from_scalar:{first_name}={val:.3f}"

            # 安全のためクランプ
            action = np.clip(action, -1.0, 1.0)

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            return

        act_msg = Float32MultiArray()
        act_msg.data = action.tolist()
        self.pub_act.publish(act_msg)

        # Debug: rate and sample action
        self._infer_count += 1
        self._last_action = action
        if self.debug:
            now = self.get_clock().now()
            elapsed = (now - self._last_log_time).nanoseconds / 1e9
            if elapsed >= max(0.1, self.log_period_sec):
                rate = self._infer_count / max(1e-6, elapsed)
                act_preview = ",".join([f"{a:.3f}" for a in action[:4]])
                # 入力テンソル形状の確認（代表）
                img_shape = None
                vec_shape = None
                for name, arr in feed.items():
                    if isinstance(arr, np.ndarray) and arr.ndim == 4:
                        img_shape = arr.shape
                    elif isinstance(arr, np.ndarray) and arr.ndim == 2:
                        vec_shape = arr.shape
                self.get_logger().info(
                    f"infer_rate={rate:.1f} Hz, img={img_shape}, vec={vec_shape}, action=[{act_preview}]"
                )
                self._infer_count = 0
                self._last_log_time = now

def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    rclpy.spin(node)