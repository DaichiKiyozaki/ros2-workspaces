import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/daichi-kiyozaki/ros_pj/unity-inference-ws/install/model_in_ros2node_pkg'
