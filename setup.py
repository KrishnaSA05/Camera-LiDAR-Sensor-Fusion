from setuptools import setup, find_packages

setup(
    name="camera_lidar_fusion",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
