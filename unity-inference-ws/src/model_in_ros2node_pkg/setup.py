from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'model_in_ros2node_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('models/*.onnx')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'map'), glob('map/*')),
        (os.path.join('share', package_name, 'rviz'),  glob('rviz/*.rviz')),
    ],
    install_requires=[
        'setuptools',
        'onnxruntime',
        'numpy',
        'opencv-python',
    ],
    zip_safe=True,
    maintainer='daichi-kiyozaki',
    maintainer_email='kiyodai02@yahoo.co.jp',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'agent_node = model_in_ros2node_pkg.agent_node:main',
        ],
    },
)
