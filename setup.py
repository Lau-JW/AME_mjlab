"""Installation script for AME_mjlab."""
from setuptools import setup, find_packages

setup(
    name="ame_mjlab",
    packages=["src"],
    version="0.0.1",
    install_requires=["mjlab==1.2.0"],
)
