#!/usr/bin/env python3

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

SUBPACKAGES = ["model", "planning", "motion", "sensing", "ros"]

d = generate_distutils_setup(
    packages=["tiago_door_planning"] + ["tiago_door_planning." + name for name in SUBPACKAGES],
    package_dir={"": "src"},
)

setup(**d)
