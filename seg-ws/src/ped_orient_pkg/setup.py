import os
from glob import glob
from pathlib import Path

from setuptools import setup, find_packages

package_name = 'ped_orient_pkg'


def _collect_tree(src_root: Path, dst_root: Path):
    """Collect files preserving relative dirs for data_files."""
    buckets = {}
    for f in glob(str(src_root / '**'), recursive=True):
        p = Path(f)
        if not p.is_file():
            continue
        if p.suffix == '.pyc' or '__pycache__' in p.parts:
            continue
        rel_parent = p.relative_to(src_root).parent
        dst = dst_root / rel_parent
        buckets.setdefault(str(dst), []).append(f)
    return [(dst, files) for dst, files in buckets.items()]

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=(
        [
            ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
            ('share/' + package_name, ['package.xml']),
            ('share/' + package_name + '/models', ['ped_orient_pkg/yolov8n-seg.pt']),
        ]
        + _collect_tree(Path('ped_orient_pkg/MEBOW'), Path('share') / package_name / 'MEBOW')
    ),
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
