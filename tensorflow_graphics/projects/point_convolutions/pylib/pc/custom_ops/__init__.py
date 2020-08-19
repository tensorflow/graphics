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
"""Loads custom ops if installed, else loads tensorflow implementations"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  import tfg_custom_ops
  CUSTOM = 1
except ImportError:
  CUSTOM = 0

if CUSTOM:
  from .custom_ops_wrapper import basis_proj
  from .custom_ops_wrapper import build_grid_ds
  from .custom_ops_wrapper import compute_keys
  from .custom_ops_wrapper import compute_pdf
  from .custom_ops_wrapper import find_neighbors
  from .custom_ops_wrapper import sampling
else:
  from .custom_ops_tf import basis_proj_tf as basis_proj
  from .custom_ops_tf import build_grid_ds_tf as build_grid_ds
  from .custom_ops_tf import compute_keys_tf as compute_keys
  from .custom_ops_tf import compute_pdf_tf as compute_pdf
  from .custom_ops_tf import find_neighbors_tf as find_neighbors
  from .custom_ops_tf import sampling_tf as sampling
