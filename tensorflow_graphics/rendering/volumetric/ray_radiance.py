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
"""This module implements the radiance-based ray rendering."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def compute_radiance(rgba_values, distances, name="ray_radiance"):
  """Renders the rgba values for points along a ray, as described in ["NeRF Representing Scenes as Neural Radiance Fields for View Synthesis"](https://github.com/bmild/nerf).

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    rgba_values: A tensor of shape `[A1, ..., An, N, 4]`, where N are the
      samples on the ray.
    distances: A tensor of shape `[A1, ..., An, N]` containing the distances
      between the samples, where N are the samples on the ray.
    name: A name for this op. Defaults to "ray_radiance".

  Returns:
    A tensor of shape `[A1, ..., An, 3]` for the estimated rgb values,
    a tensor of shape `[A1, ..., An, 1]` for the estimated density values,
    and a tensor of shape `[A1, ..., An, N]` for the sample weights.
  """

  with tf.name_scope(name):
    rgba_values = tf.convert_to_tensor(value=rgba_values)
    distances = tf.convert_to_tensor(value=distances)
    distances = tf.expand_dims(distances, -1)

    shape.check_static(
        tensor=rgba_values, tensor_name="rgba_values", has_dim_equals=(-1, 4))
    shape.check_static(
        tensor=rgba_values, tensor_name="rgba_values", has_rank_greater_than=1)
    shape.check_static(
        tensor=distances, tensor_name="distances", has_rank_greater_than=1)
    shape.compare_batch_dimensions(
        tensors=(rgba_values, distances),
        tensor_names=("ray_values", "dists"),
        last_axes=-3,
        broadcast_compatible=True)
    shape.compare_dimensions(
        tensors=(rgba_values, distances),
        tensor_names=("ray_values", "dists"),
        axes=-2)

    rgb, density = tf.split(rgba_values, [3, 1], axis=-1)
    alpha = 1. - tf.exp(-density * distances)
    alpha = tf.squeeze(alpha, -1)
    ray_sample_weights = alpha * tf.math.cumprod(
        1. - alpha + 1e-10, -1, exclusive=True)
    ray_rgb = tf.reduce_sum(
        input_tensor=tf.expand_dims(ray_sample_weights, -1) * rgb, axis=-2)
    ray_alpha = tf.expand_dims(
        tf.reduce_sum(input_tensor=ray_sample_weights, axis=-1), axis=-1)
    return ray_rgb, ray_alpha, ray_sample_weights


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
