#!/usr/bin/env python

from setuptools import setup

__author__ = 'anonymous'
__version__ = '1.0'

setup(
    name='gentle',
    version=__version__,
    description='Library for RL projects',
    long_description=open('README.md').read(),
    author=__author__,
    author_email='Jared.Markowitz@jhuapl.edu',
    license='Apache 2.0',
    packages=['gentle'],
    keywords='deep reinforcement learning, constrained RL',
    classifiers=[],
    install_requires=['numpy', 'torch', 'torchvision', 'scipy', 'tensorboard', 'mpi4py', 'matplotlib', 'seaborn', 'python-box', 'gymnasium']
)
