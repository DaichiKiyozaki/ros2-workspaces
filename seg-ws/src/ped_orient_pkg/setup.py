from setuptools import setup, find_packages

package_name = 'ped_orient_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    package_data={
        package_name: [
            'yolov8n-seg.pt',
            'MEBOW/**/*',
        ],
    },
    include_package_data=True,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='daichi-kiyozaki',
    maintainer_email='user@example.com',
    description='YOLOv8-seg + MEBOW pedestrian orientation node.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'ped_orient_node = ped_orient_pkg.ped_orient_node:main',
        ],
    },
)
