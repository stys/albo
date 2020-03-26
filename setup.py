#!/usr/bin/env python3

import os
import re

from setuptools import setup, find_packages

# get version string from module
with open(os.path.join(os.path.dirname(__file__), 'albo/__init__.py'), 'r') as f:
    version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M).group(1)

setup(
    name='albo',
    version=version,
    description='Augmented Lagrangians for constrained Bayesian optimization in BoTorch',
    author='Alexey Stysin',
    license='MIT',
    python_requires='>=3.6',
    install_requires=[
        'botorch>=0.2.1',
        'ax>=0.1.9'
    ],
    packages=find_packages()
)
