#Copyright 2018 Google LLC
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

INSTALL_PACKAGES = [
    'absl-py >= 0.6.1',
    'numpy >= 1.15.4',
    'scipy >= 1.1.0',
    'six >= 1.11.0',
]

if '--compute_platform' in sys.argv:
  compute_platform_idx = sys.argv.index('--compute_platform')
  compute_platform = sys.argv[compute_platform_idx + 1]
  sys.argv.remove('--compute_platform')
  sys.argv.pop(compute_platform_idx)
else:
  compute_platform = 'cpu'

if compute_platform not in ('cpu', 'gpu'):
  sys.exit('Supported compute platforms are cpu or gpu')

if compute_platform == 'cpu':
  INSTALL_PACKAGES.append('tensorflow >= 1.13.1')
  package_name = 'tensorflow-graphics'
else:
  INSTALL_PACKAGES.append('tensorflow-gpu >= 1.13.1')
  package_name = 'tensorflow-graphics-gpu'

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
    'tf': ['tf-nightly'],
    'tf-gpu': ['tf-nightly-gpu'],
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
