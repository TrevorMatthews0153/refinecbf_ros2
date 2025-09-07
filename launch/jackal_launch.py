import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchContext
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, PythonExpression
import yaml

package_name = 'refinecbf_ros2'
robot = 'jackal'

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def generate_launch_description():
    topics_config_path = os.path.join(get_package_share_directory(package_name), 'config', 'topics_config.yaml')
    topics_config = load_yaml(topics_config_path)['/**']['ros__parameters']

    jackal_topics_config_path = os.path.join(get_package_share_directory(package_name), 'config', 'jackal', 'topics_config.yaml')
    jackal_topics_config = load_yaml(jackal_topics_config_path)['/**']['ros__parameters']

    return LaunchDescription([
        DeclareLaunchArgument(
            'safety_filter_active', default_value='True',
            description='Activate refinecbf based safety filter'),
        DeclareLaunchArgument(
            'update_vf_online', default_value='True',
            description='Update value function online using HJ Reachability'),
        DeclareLaunchArgument(
            'vf_initialization_method', default_value='file',
            description='Value function initialization method'),
        DeclareLaunchArgument(
            'backend', default_value='hardware',
            description='sim or hardware backend for jackal'),
        DeclareLaunchArgument(
            'vf_update_method', default_value='pubsub',
            description='Message parsing method for VF update'),
        DeclareLaunchArgument(
            'vf_update_accuracy', default_value='high',
            description='Accuracy of HJ Reachability computation'),
        # DeclareLaunchArgument(
        #     'env_config_file', default_value='env.yaml',
        #     description='Environment config file'),
        # DeclareLaunchArgument(
        #     'control_config_file', default_value='control.yaml',
        #     description='Control config file'),
        # DeclareLaunchArgument(
        #     'CBF_parameter_file', default_value='turtlebot_CBF_params.yaml',
        #     description='CBF parameter file'),
        DeclareLaunchArgument(
            'exp', default_value='1',
            description='Which experiment to run'),

        # DeclareLaunchArgument(
        #     'initial_vf_file', default_value='vf.npy',
        #     description='Initial VF file'),
        # DeclareLaunchArgument(
        #     'use_sim_time',
        #     default_value='true',
        #     description='Use simulation (Gazebo) clock if true'),
        Node(
            package='refinecbf_ros2',
            executable='jackal_nominal_controller.py',
            name='jackal_nominal_control',
            output='screen',
            parameters=[topics_config_path,
                        jackal_topics_config_path,
                        {'robot': robot,
                         'exp': LaunchConfiguration('exp'),
                         'use_sim_time': PythonExpression(["'", LaunchConfiguration('backend'), "' == 'sim'"]),
                         },
                         ],
        ),
        Node(
            package='refinecbf_ros2',
            executable='jackal_hw_interface.py',
            name='jackal_hw_interface',
            output='screen',
            parameters=[topics_config_path,
                        jackal_topics_config_path,
                        {'robot': robot,
                         'exp': LaunchConfiguration('exp'),
                         'backend': LaunchConfiguration('backend'),
                         'use_sim_time': PythonExpression(["'", LaunchConfiguration('backend'), "' == 'sim'"]),
                        },
                        ],
            remappings=[('robot/final_control', '/j100_0678/cmd_vel')]
        ),
        # Node(
        #     package='refinecbf_ros2',
        #     executable='tb_visualization.py',
        #     output='screen',
        #     parameters=[topics_config_path,
        #                 {'vf_update_method': LaunchConfiguration('vf_update_method'),
        #                  'env_config_file': LaunchConfiguration('env_config_file'),
        #                  'control_config_file': LaunchConfiguration('control_config_file'),
        #                  }
        #                  ],
        # ),
        # Include other launch files
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare(package_name),
                    'launch',
                    'refine_cbf_launch.py'
                ])
            ]),
            launch_arguments={
                topics_config_path: topics_config_path,
                'robot': robot,
                'exp': LaunchConfiguration('exp'),
                'safety_filter_active': LaunchConfiguration('safety_filter_active'),
                'update_vf_online': LaunchConfiguration('update_vf_online'),
                'vf_initialization_method': LaunchConfiguration('vf_initialization_method'),
                'vf_update_method': LaunchConfiguration('vf_update_method'),
                'vf_update_accuracy': LaunchConfiguration('vf_update_accuracy'),
                'use_sim_time': PythonExpression(["'", LaunchConfiguration('backend'), "' == 'sim'"]),
            }.items()
        ),
        # GroupAction(
        #     actions=[
        #         SetRemap('/cmd_vel', topics_config['topics']['robot_safe_control']),

        #         IncludeLaunchDescription(
        #             PythonLaunchDescriptionSource([
        #                 PathJoinSubstitution([
        #                     FindPackageShare('turtlebot3_fake_node'),
        #                     'launch',
        #                     'turtlebot3_fake_node.launch.py'
        #                 ])
        #             ]),
        #                 launch_arguments={
        #                     'use_sim_time': LaunchConfiguration('use_sim_time')
        #                 }.items()
        #         ),
        #     ]
        # )
    ])






# import os
# from pathlib import Path
# from ament_index_python.packages import get_package_share_directory
# from launch import LaunchDescription, LaunchContext
# from launch_ros.actions import Node, SetRemap
# from launch_ros.substitutions import FindPackageShare
# from launch.actions import DeclareLaunchArgument, GroupAction
# from launch.substitutions import LaunchConfiguration, TextSubstitution
# from launch.actions import IncludeLaunchDescription
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from launch.substitutions import PathJoinSubstitution, TextSubstitution
# import yaml

# package_name = 'refinecbf_ros2'


# def load_yaml(file_path):
#     with open(file_path, 'r') as file:
#         return yaml.safe_load(file)


# def generate_launch_description():
#     topics_config_path = os.path.join(get_package_share_directory(package_name), 'config', 'topics_config.yaml')
#     topics_config = load_yaml(topics_config_path)['/**']['ros__parameters']

#     jackal_topics_config_path = os.path.join(get_package_share_directory(package_name), 'config','jackal', 'topics_config.yaml')
#     jackal_topics_config = load_yaml(jackal_topics_config_path)['/**']['ros__parameters']

#     return LaunchDescription([
#         DeclareLaunchArgument(
#             'safety_filter_active', default_value='True',
#             description='Activate refinecbf based safety filter'),
#         DeclareLaunchArgument(
#             'update_vf_online', default_value='True',
#             description='Update value function online using HJ Reachability'),
#         DeclareLaunchArgument(
#             'vf_initialization_method', default_value='file',
#             description='Value function initialization method'),
#         DeclareLaunchArgument(
#             'backend', default_value='hardware',
#             description='sim or hardware backend for jackal'),
#         DeclareLaunchArgument(
#             'vf_update_method', default_value='file',
#             description='Message parsing method for VF update'),
#         DeclareLaunchArgument(
#             'vf_update_accuracy', default_value='high',
#             description='Accuracy of HJ Reachability computation'),
#         DeclareLaunchArgument(
#             'env_config_file', default_value='env.yaml',
#             description='Environment config file'),
#         DeclareLaunchArgument(
#             'control_config_file', default_value='control.yaml',
#             description='Control config file'),
#         DeclareLaunchArgument(
#             'CBF_parameter_file', default_value='jackal_CBF_params.yaml',
#             description='CBF parameter file'),
#         DeclareLaunchArgument(
#             'initial_vf_file', default_value='vf.npy',
#             description='Initial VF file'),
#         DeclareLaunchArgument(
#             'use_sim_time',
#             default_value='false',
#             description='Use simulation (Gazebo) clock if true'),
#         Node(
#             package='refinecbf_ros2',
#             executable='jackal_nominal_controller.py',
#             name='jackal_nominal_control',
#             output='screen',
#             parameters=[topics_config_path,
#                         jackal_topics_config_path,
#                         '/home/administrator/refine_ws/src/refinecbf_ros2/config/jackal.yaml',
#                         {'env_config_file': LaunchConfiguration('env_config_file'),
#                          'control_config_file': LaunchConfiguration('control_config_file'),
#                          }
#                          ],
#         ),
#         Node(
#             package='refinecbf_ros2',
#             executable='jackal_hw_interface.py',
#             name='jackal_hw_interface',
#             output='screen',
#             parameters=[topics_config_path,
#                         jackal_topics_config_path,
#                         {'env_config_file': LaunchConfiguration('env_config_file'),
#                          'control_config_file': LaunchConfiguration('control_config_file'),
#                          }
#                          ],
#             remappings=[('robot/final_control', '/j100_0678/cmd_vel')]
#         ),
#         # Include other launch files
#         IncludeLaunchDescription(
#             PythonLaunchDescriptionSource([
#                 PathJoinSubstitution([
#                     FindPackageShare(package_name),
#                     'launch',
#                     'refine_cbf_launch.py'
#                 ])
#             ]),
#             launch_arguments={
#                 'topics_config_path': topics_config_path,
#                 'env_config_file': LaunchConfiguration('env_config_file'),
#                 'control_config_file': LaunchConfiguration('control_config_file'),
#                 'safety_filter_active': LaunchConfiguration('safety_filter_active'),
#                 'update_vf_online': LaunchConfiguration('update_vf_online'),
#                 'vf_initialization_method': LaunchConfiguration('vf_initialization_method'),
#                 'vf_update_method': LaunchConfiguration('vf_update_method'),
#                 'vf_update_accuracy': LaunchConfiguration('vf_update_accuracy'),
#                 'CBF_parameter_file': LaunchConfiguration('CBF_parameter_file'),
#                 'initial_vf_file': LaunchConfiguration('initial_vf_file'),
#                 'use_sim_time' : LaunchConfiguration('use_sim_time')
#             }.items()
#         ),
        
#     ])


