from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # --- Launch args ---
    use_sim_time = LaunchConfiguration("use_sim_time")
    map_yaml = LaunchConfiguration("map")
    rviz = LaunchConfiguration("rviz")
    rviz_config = LaunchConfiguration("rviz_config")

    global_frame_id = LaunchConfiguration("global_frame_id")
    odom_frame_id = LaunchConfiguration("odom_frame_id")
    base_frame_id = LaunchConfiguration("base_frame_id")
    scan_topic = LaunchConfiguration("scan_topic")

    # --- Nodes ---
    map_server_node = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "yaml_filename": map_yaml,
            }
        ],
    )

    amcl_node = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "global_frame_id": global_frame_id,  # map
                "odom_frame_id": odom_frame_id,      # odom
                "base_frame_id": base_frame_id,      # base_link
                "scan_topic": scan_topic,            # /scan
                "update_min_d": 0.1,                 # 移動距離の閾値を下げて更新頻度を上げる
                "update_min_a": 0.1,                 # 回転角の閾値を下げて更新頻度を上げる
                "transform_tolerance": 0.1,          # tf変換の許容時間
                "min_particles": 500,
                "max_particles": 2000,
            }
        ],
    )

    # lifecycle_manager が /map_server と /amcl を configure -> activate まで自動遷移
    lifecycle_manager_node = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "autostart": True,
                "node_names": ["map_server", "amcl"],
            }
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        condition=IfCondition(rviz),
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # --- Launch description ---
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "map",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("model_in_ros2node_pkg"), "map", "my_map.yaml"]
                ),
                description="Map YAML file for nav2_map_server",
            ),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("model_in_ros2node_pkg"), "rviz", "unity_ros_inf_only_sensors.rviz"]
                ), # rviz設定ファイルのパス
                description="RViz config file path. If empty, RViz opens with default config.",
            ),
            DeclareLaunchArgument("global_frame_id", default_value="map"),
            DeclareLaunchArgument("odom_frame_id", default_value="odom"),
            DeclareLaunchArgument("base_frame_id", default_value="base_link"),
            DeclareLaunchArgument("scan_topic", default_value="/scan"),
            map_server_node,
            amcl_node,
            lifecycle_manager_node,
            rviz_node,
        ]
    )
