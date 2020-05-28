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
"""This module implements the absorption-only voxel rendering."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def render(voxels, absorption_factor=0.1, cell_size=1.0, name=None):
  """Renders a voxel grid using the absorption-only model, as described in ["Escaping Plato's Cave: 3D Shape From Adversarial Rendering" (Henzler 2019)](https://github.com/henzler/platonicgan).

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    voxels: A tensor of shape `[A1, ..., An, Vx, Vy, Vz, Vd]`, where Vx, Vy, Vz
      are the dimensions of the voxel grid and Vd the dimension of the
      information stored in each voxel (e.g. 3 for RGB color).
    absorption_factor: A scalar representing the density of the volume.
    cell_size: A scalar representing the size of a cell.
    name: A name for this op. Defaults to "absorption_render".

  Returns:
    A tensor of shape `[A1, ..., An, Vx, Vy, Vd]` representing images of size
    (Vx,Vy).

  Raises:
    ValueError: If the shape of the input tensors are not supported.
  """
  with tf.compat.v1.name_scope(name, "absorption_render", [voxels]):
    voxels = tf.convert_to_tensor(value=voxels)

    shape.check_static(
        tensor=voxels, tensor_name="voxels", has_rank_greater_than=3)

    transmission = tf.scalar_mul(absorption_factor / cell_size, voxels)
    transmission = tf.ones_like(transmission) - transmission
    transmission = tf.clip_by_value(
        transmission, clip_value_min=1e-6, clip_value_max=1.0)

    image = tf.math.log(transmission)
    image = tf.reduce_sum(input_tensor=image, axis=-2)
    image = tf.math.exp(image)
    image = tf.ones_like(image) - image
    return image


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
