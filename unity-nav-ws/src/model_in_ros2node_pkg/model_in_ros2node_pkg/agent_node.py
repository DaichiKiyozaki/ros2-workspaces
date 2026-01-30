import rclpy
from rclpy.node import Node
import onnxruntime as ort
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
from ament_index_python.packages import get_package_share_directory
import os
from collections import deque
import math
from rclpy.qos import qos_profile_sensor_data


class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')

        self.bridge = CvBridge()

        # ゴールと自己位置（map フレーム想定）
        self.goal_xy = None              # np.array([goal_x, goal_y], dtype=np.float32)
        self.robot_xyyaw = None          # np.array([rx, ry, yaw], dtype=np.float32)
        self._warned_goal_frame_mismatch = False
        self._warned_amcl_frame_mismatch = False
        self.goal_count = 0
        self.amcl_count = 0
        self._warned_no_goal = False
        self._warned_no_amcl = False
        self.last_goal_stamp = None
        self.last_amcl_stamp = None
        self._last_goal_log_time = self.get_clock().now()
        self._last_amcl_log_time = self.get_clock().now()

        # parameters
        self.declare_parameter('debug', True)
        self.declare_parameter('log_period_sec', 1.0)
        # モデル関連
        self.declare_parameter('model_file_name', 'balance.onnx')
        self.declare_parameter('img_width', 112)
        self.declare_parameter('img_height', 84)
        self.declare_parameter('stack_size', 5)
        # ベクトル観測の要素数（[angle_deg, distance_m] を基本に不足は 0 埋め、超過は切り捨て）
        self.declare_parameter('vec_obs_dim', 2)

        self.debug = bool(self.get_parameter('debug').value)
        self.log_period_sec = float(self.get_parameter('log_period_sec').value)
        self.img_width = int(self.get_parameter('img_width').value)
        self.img_height = int(self.get_parameter('img_height').value)
        self.stack_size = int(self.get_parameter('stack_size').value)
        self.vec_obs_dim = int(self.get_parameter('vec_obs_dim').value)

        if self.img_width <= 0 or self.img_height <= 0:
            raise ValueError(f"img_width/img_height must be > 0: ({self.img_width}, {self.img_height})")
        if self.stack_size <= 0:
            raise ValueError(f"stack_size must be > 0: {self.stack_size}")
        if self.vec_obs_dim <= 0:
            raise ValueError(f"vec_obs_dim must be > 0: {self.vec_obs_dim}")

        # 画像フレームバッファ（deque）
        self.frame_buffer = deque(maxlen=self.stack_size)
        self._infer_count = 0
        self._last_log_time = self.get_clock().now()

        # Subscribe：Unityカメラ画像（SensorData QoS で遅延を最小化）
        self.sub_cam = self.create_subscription(Image, '/unity/camera/image_raw', self.cb_cam, qos_profile_sensor_data)

        # Subscribe：RViz2 の 2D Nav Goal（標準は /goal_pose）
        self.sub_goal_pose = self.create_subscription(PoseStamped, '/goal_pose', self.cb_goal_pose, 10)

        # Subscribe：AMCL の自己位置
        # nav2_amcl は QoS が SensorData 互換なので受信ミスを防ぐため sensor_data プロファイルを使用
        self.sub_amcl_pose = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.cb_amcl_pose, qos_profile_sensor_data
        )

        # Publish：推論結果アクション
        self.pub_act = self.create_publisher(Float32MultiArray, '/agent/cmd', 10)

        # デバッグ用：スタック画像の可視化
        self.pub_debug_stack = self.create_publisher(Image, '/debug/stacked_image', 10)

        # モデル読み込み
        share_dir = get_package_share_directory('model_in_ros2node_pkg')
        model_file_name = str(self.get_parameter('model_file_name').value or '').strip()
        if not model_file_name:
            model_file_name = 'balance.onnx'
        model_path = os.path.join(share_dir, 'models', model_file_name)
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        # ONNX 入出力情報をログ出力
        self.inputs = self.session.get_inputs()
        self.input_names = [i.name for i in self.inputs]
        self.input_shapes = [i.shape for i in self.inputs]
        self.get_logger().info(f"ONNX inputs: {list(zip(self.input_names, self.input_shapes))}")

        self.declare_parameter('action_output_name', "")
        self.outputs = self.session.get_outputs()
        self.output_names = [o.name for o in self.outputs]
        self.output_shapes = [o.shape for o in self.outputs]
        self.get_logger().info(f"ONNX outputs: {list(zip(self.output_names, self.output_shapes))}")

        # アクション出力は起動時に1回だけ決定（推論時は固定参照）
        # continuous_actions: 連続値アクション（確率分布）
        # deterministic_continuous_actions: 分布の平均値（同じ入力に対して同じ出力）
        preferred = ['continuous_actions', 'deterministic_continuous_actions']
        param_action = str(self.get_parameter('action_output_name').value or "").strip()
        if param_action:
            if 'discrete' in param_action:
                raise RuntimeError(
                    "discrete action outputs are not supported. "
                    f"requested action_output_name='{param_action}'"
                )
            if param_action not in self.output_names:
                raise RuntimeError(
                    f"action_output_name='{param_action}' is not in model outputs. available_outputs={self.output_names}"
                )
            self.action_output_name = param_action
        else:
            self.action_output_name = next((n for n in preferred if n in self.output_names), None)
            if self.action_output_name is None:
                raise RuntimeError(
                    f"action output not found. preferred={preferred}, available_outputs={self.output_names}"
                )

        if 'discrete' in self.action_output_name:
            raise RuntimeError(
                "discrete action outputs are not supported. "
                f"selected action_output_name='{self.action_output_name}', available_outputs={self.output_names}"
            )
        self.get_logger().info(
            f"action_output_name={self.action_output_name}, available_outputs={self.output_names}"
        )

    # ----------------------------
    # Callback: Goal / AMCL
    # ----------------------------
    def cb_goal_pose(self, msg: PoseStamped):
        # frame_id は map を想定（異なる場合は警告だけ出す：TF変換は未実装）
        frame = (msg.header.frame_id or "").strip()
        if frame != 'map':
            if not self._warned_goal_frame_mismatch:
                self.get_logger().warn(f"Ignoring /goal_pose frame_id='{frame}' (expected: 'map').")
                self._warned_goal_frame_mismatch = True
            return

        goal_x = float(msg.pose.position.x)
        goal_y = float(msg.pose.position.y)
        self.goal_xy = np.array([goal_x, goal_y], dtype=np.float32)
        self.last_goal_stamp = msg.header.stamp
        self.goal_count += 1

        now = self.get_clock().now()
        if self.debug and (self.goal_count == 1 or (now - self._last_goal_log_time).nanoseconds / 1e9 >= self.log_period_sec):
            self.get_logger().info(
                f"goal updated #{self.goal_count}: frame_id='{frame}', goal=({goal_x:.3f},{goal_y:.3f}), stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}"
            )
            self._last_goal_log_time = now

    def cb_amcl_pose(self, msg: PoseWithCovarianceStamped):
        # frame_id は map を想定（異なる場合は警告だけ出す：TF変換は未実装）
        frame = (msg.header.frame_id or "").strip()
        if frame != 'map':
            if not self._warned_amcl_frame_mismatch:
                self.get_logger().warn(f"Ignoring /amcl_pose frame_id='{frame}' (expected: 'map').")
                self._warned_amcl_frame_mismatch = True
            return

        px = float(msg.pose.pose.position.x)
        py = float(msg.pose.pose.position.y)

        # クォータニオン（四次元回転）から yaw（Z軸回転角度） を算出
        q = msg.pose.pose.orientation
        yaw = self.quat_to_yaw(q.x, q.y, q.z, q.w)

        self.robot_xyyaw = np.array([px, py, yaw], dtype=np.float32)
        self.last_amcl_stamp = msg.header.stamp
        self.amcl_count += 1

        now = self.get_clock().now()
        if self.debug and (self.amcl_count == 1 or (now - self._last_amcl_log_time).nanoseconds / 1e9 >= self.log_period_sec):
            self.get_logger().info(
                f"amcl updated #{self.amcl_count}: frame_id='{frame}', pose=({px:.3f},{py:.3f},{yaw:.3f}), stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}"
            )
            self._last_amcl_log_time = now

    # ----------------------------
    # 数学ユーティリティ
    # ----------------------------
    @staticmethod
    def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
        """クォータニオンから yaw（Z軸回転）を算出"""
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(math.atan2(siny_cosp, cosy_cosp))

    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        """角度を [-pi, pi] に正規化"""
        a = (angle + math.pi) % (2.0 * math.pi) - math.pi
        return float(a)

    @staticmethod
    def wrap_to_180_deg(angle_deg: float) -> float:
        """角度を [-180, 180] に正規化（deg）"""
        a = (angle_deg + 180.0) % 360.0 - 180.0
        return float(a)

    def compute_goal_vector(self) -> np.ndarray:
        """
        Unity学習時と同じ (1,2) ベクトル観測 [方向(deg), 距離d] を基準に返す
        方向は yaw - goal_heading を [-pi, pi] に正規化し deg へ変換する
        ゴールまたは自己位置が未取得なら 0 埋めで返す
        vec_obs_dim に合わせて 0 埋め/切り捨てを行う
        """
        if self.goal_xy is None or self.robot_xyyaw is None:
            return np.zeros((1, self.vec_obs_dim), dtype=np.float32)

        goal_x, goal_y = float(self.goal_xy[0]), float(self.goal_xy[1])
        rx, ry, yaw = float(self.robot_xyyaw[0]), float(self.robot_xyyaw[1]), float(self.robot_xyyaw[2])

        dx = goal_x - rx
        dy = goal_y - ry
        d = math.sqrt(dx * dx + dy * dy)

        goal_heading = math.atan2(dy, dx)
        # 角度差を [-pi, pi] に正規化して度(degree)へ変換
        # Unity の SignedAngle(targetDirection, forward, up) と同じ符号にするため goal_heading - yaw
        signed_rad = self.wrap_to_pi(goal_heading - yaw)
        signed_deg = self.wrap_to_180_deg(math.degrees(signed_rad))

        base = np.array([[signed_deg, d]], dtype=np.float32)
        if self.vec_obs_dim == 2:
            return base
        if self.vec_obs_dim < 2:
            return base[:, : self.vec_obs_dim]
        return np.pad(base, ((0, 0), (0, self.vec_obs_dim - 2)), constant_values=0.0)

    # ----------------------------
    # カメラコールバック：推論処理
    # ----------------------------
    def cb_cam(self, msg: Image):
        # 画像変換（RGB）
        try:
            img_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')  # (H,W,3), uint8
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # 学習時解像度へリサイズ
        img_rgb = cv2.resize(img_rgb, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)

        # [0,1] 正規化
        img_f = img_rgb.astype(np.float32) / 255.0  # (H,W,3)

        # CHW へ変換
        img_chw = np.transpose(img_f, (2, 0, 1))  # (3,H,W)

        # スタック用バッファへ投入
        if len(self.frame_buffer) == 0:
            # 初回は同一フレームで埋める
            for _ in range(self.stack_size):
                self.frame_buffer.append(img_chw)
        else:
            self.frame_buffer.append(img_chw)

        # スタック作成： (3*Stack,H,W)
        stacked_chw = np.concatenate(list(self.frame_buffer), axis=0)
        img_nchw = np.expand_dims(stacked_chw, axis=0)  # (1, C*Stack, H, W)

        # NHWC も用意（モデルが NHWC の場合に備える）
        img_nhwc = np.transpose(img_nchw, (0, 2, 3, 1))  # (1, H, W, C*Stack)

        # ゴール/AMCL 未受信なら推論をスキップ（1回だけ通知）
        if self.goal_xy is None:
            if not self._warned_no_goal and self.debug:
                self.get_logger().info("waiting for /goal_pose")
                self._warned_no_goal = True
            return
        else:
            self._warned_no_goal = False
            # AMCL 未受信の場合は 0 埋めで推論するが通知だけ出す
            if self.robot_xyyaw is None and not self._warned_no_amcl and self.debug:
                self.get_logger().info("waiting for /amcl_pose")
                self._warned_no_amcl = True
            elif self.robot_xyyaw is not None:
                self._warned_no_amcl = False

        # デバッグ：スタック画像の可視化（横に並べて publish）
        if self.debug:
            try:
                frames_vis = []
                for f in self.frame_buffer:
                    f_hwc = np.transpose(f, (1, 2, 0))
                    f_uint8 = (f_hwc * 255.0).astype(np.uint8)
                    frames_vis.append(f_uint8)
                stacked_vis = np.concatenate(frames_vis, axis=1)
                debug_msg = self.bridge.cv2_to_imgmsg(stacked_vis, encoding='rgb8')
                self.pub_debug_stack.publish(debug_msg)
            except Exception as e:
                self.get_logger().warn(f"Failed to publish debug image: {e}")

        # ベクトル観測：/goal_pose と /amcl_pose から計算
        vec = self.compute_goal_vector()  # (1,2) float32

        # 入力 feed を構築
        # feed: {input_name: input_value, ...}
        feed = {}
        for name, shape in zip(self.input_names, self.input_shapes):
            if len(shape) == 4:
                # 画像入力：NCHW か NHWC を簡易判定
                # NCHW: (1, C, H, W), NHWC: (1, H, W, C)
                ch_dim1 = shape[1]
                if isinstance(ch_dim1, int) and (ch_dim1 == 3 or ch_dim1 == 3 * self.stack_size):
                    feed[name] = img_nchw
                else:
                    feed[name] = img_nhwc
            else:
                # ベクトル入力：次元を合わせてパディング/トリム
                target_dim = shape[-1] if isinstance(shape[-1], int) else vec.shape[1]
                v = vec
                if isinstance(target_dim, int) and v.shape[1] != target_dim:
                    if v.shape[1] < target_dim:
                        v = np.pad(v, ((0, 0), (0, target_dim - v.shape[1])), constant_values=0.0)
                    else:
                        v = v[:, :target_dim]
                feed[name] = v.astype(np.float32)

        # 推論実行
        try:
            # 決定済みの出力のみ取得してアクション化
            action = self.session.run([self.action_output_name], feed)[0]
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            action = np.clip(action, -1.0, 1.0)

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            return

        # 行動出力 (後進不可)
        if action.size >= 1:
            action[0] = max(0.0, float(action[0]))

        act_msg = Float32MultiArray()
        act_msg.data = action.tolist()
        self.pub_act.publish(act_msg)

        # Debug: Inference rate, input shapes, distance/angle logs
        self._infer_count += 1
        if self.debug:
            now = self.get_clock().now()
            elapsed = (now - self._last_log_time).nanoseconds / 1e9
            if elapsed >= max(0.1, self.log_period_sec):
                rate = self._infer_count / max(1e-6, elapsed)
                act_preview = ",".join([f"{a:.3f}" for a in action[:4]])

                img_shape = None
                vec_shape = None
                for _, arr in feed.items():
                    if isinstance(arr, np.ndarray) and arr.ndim == 4:
                        img_shape = arr.shape
                    elif isinstance(arr, np.ndarray) and arr.ndim == 2:
                        vec_shape = arr.shape
                
                # Extract distance and angle from vec observation
                angle_to_goal = float(vec[0, 0]) if vec.shape[1] >= 1 else 0.0
                distance_to_goal = float(vec[0, 1]) if vec.shape[1] >= 2 else 0.0
                
                self.get_logger().info(
                    f"infer_rate={rate:.1f} Hz, img={img_shape}, vec={vec_shape}, "
                    f"vec_obs=[{angle_to_goal:.2f}°, {distance_to_goal:.3f}m], action=[{act_preview}]"
                )

                self._infer_count = 0
                self._last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    rclpy.spin(node)
