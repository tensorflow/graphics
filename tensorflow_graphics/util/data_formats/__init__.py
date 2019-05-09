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
"""Data format module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# pylint: disable=g-import-not-at-top
try:
  from tensorflow_graphics.util.data_formats import exr
except ImportError:
  print(
      "Warning: To use the exr data format, please install the OpenEXR"
      " package following the instructions detailed in the README at"
      " github.com/tensorflow/graphics.",
      file=sys.stderr)
# pylint: enable=g-import-not-at-top

# The util modules are not exported.
__all__ = []
