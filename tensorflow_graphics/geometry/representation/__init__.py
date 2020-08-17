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
"""Geometry module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_graphics.geometry.representation import grid
from tensorflow_graphics.geometry.representation import mesh
from tensorflow_graphics.geometry.representation import point
from tensorflow_graphics.geometry.representation import ray
from tensorflow_graphics.geometry.representation import triangle
from tensorflow_graphics.math import vector
from tensorflow_graphics.util import export_api as _export_api

# API contains submodules of tensorflow_graphics.geometry.representation.
__all__ = _export_api.get_modules()
