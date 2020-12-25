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
"""Set-up script for installing extension modules."""
from Cython.Build import cythonize
import numpy
from setuptools import Extension
from setuptools import setup

# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# mise (efficient mesh extraction)
mise_module = Extension(
    "lib.libmise.mise",
    sources=["lib/libmise/mise.pyx"],
)

# Gather all extension modules
ext_modules = [
    mise_module,
]

setup(ext_modules=cythonize(ext_modules),)
