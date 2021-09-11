# -*- coding: utf-8 -*-

"""
This module contains RL for trading exploration
"""
import os
from setuptools import setup, find_packages
import rltrading


README = os.path.join(os.path.dirname(__file__), 'README.md')
long_description = open(README).read() + '\n\n'


setup(name='rltrading',
      version=rltrading.__version__,
      description="RL for trading exploration",
      long_description=long_description,
      author='Guillaume Androz',
      packages=find_packages(),
      include_package_data=True)
