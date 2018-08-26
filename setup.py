#!/usr/bin/env python
# coding: utf-8
import platform
from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

install_requires=['torch','numpy','numba','visdom']

setup(
    name='pointvision',
    version='0.0.0.2',
    author='Yinglin Zheng',
    author_email='admin@hypercube.top',
    description=u'Point Cloud Deep Vision Library',
    keywords='pointcloud deeplearning torch',
    url='https://github.com/elliottzheng/PointVision',
    packages=['pointvision'],
    package_data  = {
        "CopyTranslator": ["logo.ico"]},
    install_requires=install_requires,
    project_urls={  # Optional
            'Bug Reports': 'https://github.com/elliottzheng/PointVision/issues',
            'Say Thanks!': 'https://saythanks.io/to/elliottzheng',
            'Source': 'https://github.com/elliottzheng/PointVision',
    },
    classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Education',
    'Operating System :: OS Independent',
    'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
	'Programming Language :: Python :: 3',
	'Programming Language :: Python :: 3.1',
	'Programming Language :: Python :: 3.2',
	'Programming Language :: Python :: 3.3',
	'Programming Language :: Python :: 3.4',
	'Programming Language :: Python :: 3.5',
	'Programming Language :: Python :: 3.6',
],
)