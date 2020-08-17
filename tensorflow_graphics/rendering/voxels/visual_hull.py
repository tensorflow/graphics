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
"""This module implements the visual hull voxel rendering."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def render(voxels, axis=2, name=None):
  """Renders the visual hull of a voxel grid, as described in ["Escaping Plato's Cave: 3D Shape From Adversarial Rendering" (Henzler 2019)](https://github.com/henzler/platonicgan).

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    voxels: A tensor of shape `[A1, ..., An, Vx, Vy, Vz, Vd]`, where Vx, Vy, Vz
      are the dimensions of the voxel grid and Vd the dimension of the
      information stored in each voxel (e.g. 3 for RGB color).
    axis: An index to the projection axis (0 for X, 1 for Y or 2 for Z).
    name: A name for this op. Defaults to "visual_hull_render".

  Returns:
    A tensor of shape `[A1, ..., An, Vx, Vy, Vd]` representing images of size
    (Vx,Vy).

  Raises:
    ValueError: If the shape of the input tensors are not supported.
  """
  with tf.compat.v1.name_scope(name, "visual_hull_render", [voxels]):
    voxels = tf.convert_to_tensor(value=voxels)

    shape.check_static(
        tensor=voxels, tensor_name="voxels", has_rank_greater_than=3)
    if axis not in [0, 1, 2]:
      raise ValueError("'axis' needs to be 0, 1 or 2")

    image = tf.reduce_sum(input_tensor=voxels, axis=axis - 4)
    image = tf.ones_like(image) - tf.math.exp(-image)
    return image


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
