# Copyright 2020 The TensorFlow Authors
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
  python setup.py sdist bdist_wheel [--nightly]

Note:
  assumes setup.py file lives at the root of the project.
"""

import datetime
import os
import sys
import setuptools

# --- Name to use in PyPi
if "--nightly" in sys.argv:
  sys.argv.remove("--nightly")
  NAME = "tfg-nightly"
else:
  NAME = "tensorflow-graphics"

# --- compute the version (for both nightly and normal)
now = datetime.datetime.now()
VERSION = "{}.{}.{}".format(now.year, now.month, now.day)
curr_path = os.path.dirname(__file__)
ini_file_path = os.path.join(curr_path, "tensorflow_graphics/__init__.py")
ini_file_lines = list(open(ini_file_path))
with open(ini_file_path, "w") as f:
  for line in ini_file_lines:
    f.write(line.replace("__version__ = \"HEAD\"",
                         "__version__ = \"{}\"".format(VERSION)))

# --- Extract the dependencies
REQS = [line.strip() for line in open("requirements.txt")]
INSTALL_PACKAGES = [line for line in REQS if not line.startswith("#")]

# --- Build the whl file
setuptools.setup(
    name=NAME,
    version=VERSION,
    description=("A library that contains well defined, reusable and cleanly "
                 "written graphics related ops and utility functions for "
                 "TensorFlow."),
    long_description="",
    url="https://github.com/tensorflow/graphics",
    author="Google LLC",
    author_email="tf-graphics-eng@google.com",
    install_requires=INSTALL_PACKAGES,
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="Apache 2.0",
    keywords=[
        "machine learning",
        "tensorflow",
        "graphics",
        "geometry",
        "3D",
    ],
)
