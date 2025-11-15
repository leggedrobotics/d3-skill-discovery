# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'unsupervised_RL' python package."""

from setuptools import setup, find_packages

# Read version from package
def get_version():
    """Get version from the package __init__.py file."""
    import os
    here = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(here, "unsupervised_RL", "__init__.py")
    
    with open(version_file, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "psutil",
]

# Installation operation
setup(
    name="unsupervised_RL",
    packages=find_packages(),
    author="Legged Robotics Lab",
    maintainer="Rafael Cabrera",
    url="https://github.com/leggedrobotics/unsupervised_RL",
    version=get_version(),
    description="Unsupervised skill discovery environments for Isaac Lab",
    keywords=["robotics", "rl", "isaac", "simulation"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 2023.1.1",
        "Isaac Sim :: 4.0.0",
    ],
    zip_safe=False,
)
