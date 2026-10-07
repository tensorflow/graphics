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
"""This module implements phong specular reflectance.

For a derivation of the normalization factor ensuring energy conservation, we
refer the interested reader to
Eric P. Lafortune, and Yves D. Willems.
"Using the modified phong reflectance model for physically based rendering".
1994.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

from tensorflow_graphics.math import vector
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape
from tensorflow_graphics.util import type_alias


def _brdf_normalization_factor(shininess: type_alias.TensorLike) -> tf.Tensor:
  """Returns the normalization factor needed to ensure energy conservation."""
  return (shininess + 2.0) / (2.0 * math.pi)


def brdf(direction_incoming_light: type_alias.TensorLike,
         direction_outgoing_light: type_alias.TensorLike,
         surface_normal: type_alias.TensorLike,
         shininess: type_alias.TensorLike,
         albedo: type_alias.TensorLike,
         brdf_normalization: bool = True,
         name: str = "phong_brdf") -> tf.Tensor:
  """Evaluates the specular brdf of the Phong model.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Note:
    The gradient of this function is not smooth when the dot product of the
    normal with any light is 0.0.

  Args:
    direction_incoming_light: A tensor of shape `[A1, ..., An, 3]`, where the
      last dimension represents a normalized incoming light vector.
    direction_outgoing_light: A tensor of shape `[A1, ..., An, 3]`, where the
      last dimension represents a normalized outgoing light vector.
    surface_normal: A tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents a normalized surface normal.
    shininess: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      represents a non-negative shininess coefficient.
    albedo: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents albedo with values in [0,1].
    brdf_normalization: A `bool` indicating whether normalization should be
      applied to enforce the energy conservation property of BRDFs. Note that
      `brdf_normalization` must be set to False in order to use the original
      Blinn specular model.
    name: A name for this op. Defaults to "phong_brdf".

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
      the amount of light reflected in the outgoing light direction.

  Raises:
    ValueError: if the shape of `direction_incoming_light`,
    `direction_outgoing_light`, `surface_normal`, `shininess` or `albedo` is not
    supported.
    InvalidArgumentError: if not all of shininess values are non-negative, or if
    at least one element of `albedo` is outside of [0,1].
  """
  with tf.name_scope(name):
    direction_incoming_light = tf.convert_to_tensor(
        value=direction_incoming_light)
    direction_outgoing_light = tf.convert_to_tensor(
        value=direction_outgoing_light)
    surface_normal = tf.convert_to_tensor(value=surface_normal)
    shininess = tf.convert_to_tensor(value=shininess)
    albedo = tf.convert_to_tensor(value=albedo)

    shape.check_static(
        tensor=direction_incoming_light,
        tensor_name="direction_incoming_light",
        has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=direction_outgoing_light,
        tensor_name="direction_outgoing_light",
        has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=surface_normal,
        tensor_name="surface_normal",
        has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=shininess, tensor_name="shininess", has_dim_equals=(-1, 1))
    shape.check_static(
        tensor=albedo, tensor_name="albedo", has_dim_equals=(-1, 3))
    shape.compare_batch_dimensions(
        tensors=(direction_incoming_light, direction_outgoing_light,
                 surface_normal, shininess, albedo),
        tensor_names=("direction_incoming_light", "direction_outgoing_light",
                      "surface_normal", "shininess", "albedo"),
        last_axes=-2,
        broadcast_compatible=True)
    direction_incoming_light = asserts.assert_normalized(
        direction_incoming_light)
    direction_outgoing_light = asserts.assert_normalized(
        direction_outgoing_light)
    surface_normal = asserts.assert_normalized(surface_normal)
    albedo = asserts.assert_all_in_range(albedo, 0.0, 1.0, open_bounds=False)
    shininess = asserts.assert_all_above(shininess, 0.0, open_bound=False)

    # Checks whether the incoming or outgoing light point behind the surface.
    dot_incoming_light_surface_normal = vector.dot(-direction_incoming_light,
                                                   surface_normal)
    dot_outgoing_light_surface_normal = vector.dot(direction_outgoing_light,
                                                   surface_normal)
    min_dot = tf.minimum(dot_incoming_light_surface_normal,
                         dot_outgoing_light_surface_normal)
    perfect_reflection_direction = vector.reflect(direction_incoming_light,
                                                  surface_normal)
    perfect_reflection_direction = tf.math.l2_normalize(
        perfect_reflection_direction, axis=-1)
    cos_alpha = vector.dot(
        perfect_reflection_direction, direction_outgoing_light, axis=-1)
    cos_alpha = tf.maximum(cos_alpha, tf.zeros_like(cos_alpha))
    phong_model = albedo * tf.pow(cos_alpha, shininess)
    if brdf_normalization:
      phong_model *= _brdf_normalization_factor(shininess)
    common_shape = shape.get_broadcasted_shape(min_dot.shape, phong_model.shape)
    d_val = lambda dim: 1 if dim is None else tf.compat.dimension_value(dim)
    common_shape = [d_val(dim) for dim in common_shape]
    condition = tf.broadcast_to(tf.greater_equal(min_dot, 0.0), common_shape)
    phong_model = tf.broadcast_to(phong_model, common_shape)
    return tf.where(condition, phong_model, tf.zeros_like(phong_model))


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
