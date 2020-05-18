# Copyright 2020 Google LLC
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
"""Setup for pip package.

Usage:
  python setup.py sdist bdist_wheel
"""


import os
import sys
import setuptools


def get_version():
  """NOTE: assumes this file lives at the root of the project."""
  version_path = os.path.join(os.path.dirname(__file__), 'tensorflow_graphics')
  sys.path.append(version_path)
  from version import __version__  # pylint: disable=g-import-not-at-top
  return __version__

INSTALL_PACKAGES = [line.strip() for line in open('requirements.txt')]
INSTALL_PACKAGES = [line for line in INSTALL_PACKAGES \
                    if not line.startswith('#')]

setuptools.setup(
    name='tensorflow-graphics',
    version=get_version(),
    description=('A library that contains well defined, reusable and cleanly '
                 'written graphics related ops and utility functions for '
                 'TensorFlow.'),
    long_description='',
    url='https://github.com/tensorflow/graphics',
    author='Google LLC',
    author_email='tf-graphics-eng@google.com',
    install_requires=INSTALL_PACKAGES,
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    license='Apache 2.0',
    keywords=[
        'machine learning',
        'tensorflow',
        'graphics',
        'geometry',
        '3D',
    ],
)
