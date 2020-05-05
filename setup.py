#Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Setup for pip package."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from setuptools import find_packages
from setuptools import setup

version_path = os.path.join(os.path.dirname(__file__), 'tensorflow_graphics')
sys.path.append(version_path)
from version import __version__  # pylint: disable=g-import-not-at-top

tensorflow_version = '1.13.1'

INSTALL_PACKAGES = [line.strip() for line in open('requirements.txt')]
INSTALL_PACKAGES = [line for line in INSTALL_PACKAGES \
                    if not line.startswith('#')]

if '--compute_platform' in sys.argv:
  compute_platform_idx = sys.argv.index('--compute_platform')
  compute_platform = sys.argv[compute_platform_idx + 1]
  sys.argv.remove('--compute_platform')
  sys.argv.pop(compute_platform_idx)
else:
  compute_platform = 'none'

if compute_platform not in ('none', 'cpu', 'gpu'):
  sys.exit('Supported compute platforms are none, cpu, or gpu')

if compute_platform == 'cpu':
  INSTALL_PACKAGES.append('tensorflow >= ' + tensorflow_version)
  package_name = 'tensorflow-graphics'
elif compute_platform == 'gpu':
  INSTALL_PACKAGES.append('tensorflow-gpu >= ' + tensorflow_version)
  package_name = 'tensorflow-graphics-gpu'
else:
  package_name = 'tensorflow-graphics'

SETUP_PACKAGES = [
    'pytest-runner',
]

TEST_PACKAGES = [
    'pytest',
    'pytest-mock',
    'python-coveralls',
]

EXTRA_PACKAGES = {
    'test': TEST_PACKAGES,
    'tf': ['tensorflow >= ' + tensorflow_version],
    'tf-gpu': ['tensorflow-gpu >= ' + tensorflow_version],
    'tf-nightly': ['tf-nightly-2.0-preview'],
    'tf-nightly-gpu': ['tf-nightly-gpu-2.0-preview'],
}

setup(
    name=package_name,
    version=__version__,
    description=('A library that contains well defined, reusable and cleanly '
                 'written graphics related ops and utility functions for '
                 'TensorFlow.'),
    long_description='',
    url='https://github.com/tensorflow/graphics',
    author='Google LLC',
    author_email='tf-graphics-eng@google.com',
    install_requires=INSTALL_PACKAGES,
    setup_requires=SETUP_PACKAGES,
    tests_require=TEST_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    license='Apache 2.0',
    keywords=[
        'tensorflow',
        'graphics',
        'machine learning',
    ],
)
