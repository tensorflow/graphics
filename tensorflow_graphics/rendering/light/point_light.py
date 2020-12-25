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
"""This module implements the rendering equation for a point light."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from tensorflow_graphics.math import vector
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def estimate_radiance(point_light_radiance,
                      point_light_position,
                      surface_point_position,
                      surface_point_normal,
                      observation_point,
                      brdf,
                      name=None,
                      reflected_light_fall_off=False):
  """Estimates the spectral radiance of a point light reflected from the surface point towards the observation point.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.
    B1 to Bm are optional batch dimensions for the lights, which must be
    broadcast compatible.

  Note:
    In case the light or the observation point are located behind the surface
    the function will return 0.

  Note:
    The gradient of this function is not smooth when the dot product of the
    normal with the light-to-surface or surface-to-observation vectors is 0.

  Args:
    point_light_radiance: A tensor of shape '[B1, ..., Bm, K]', where the last
      axis represents the radiance of the point light at a specific wave length.
    point_light_position: A tensor of shape `[B1, ..., Bm, 3]`, where the last
      axis represents the position of the point light.
    surface_point_position: A tensor of shape `[A1, ..., An, 3]`, where the last
      axis represents the position of the surface point.
    surface_point_normal: A tensor of shape `[A1, ..., An, 3]`, where the last
      axis represents the normalized surface normal at the given surface point.
    observation_point: A tensor of shape `[A1, ..., An, 3]`, where the last axis
      represents the observation point.
    brdf: The BRDF of the surface as a function of:
      incoming_light_direction - The incoming light direction as the last axis
      of a tensor with shape `[A1, ..., An, 3]`.
      outgoing_light_direction - The outgoing light direction as the last axis
      of a tensor with shape `[A1, ..., An, 3]`.
      surface_point_normal - The surface normal as the last axis of a tensor
      with shape `[A1, ..., An, 3]`.
      Note - The BRDF should return a tensor of size '[A1, ..., An, K]' where
      the last axis represents the amount of reflected light in each wave
      length.
    name: A name for this op. Defaults to "estimate_radiance".
    reflected_light_fall_off: A boolean specifying whether or not to include the
      fall off of the light reflected from the surface towards the observation
      point in the calculation. Defaults to False.

  Returns:
    A tensor of shape `[A1, ..., An, B1, ..., Bm, K]`, where the last
      axis represents the amount of light received at the observation point
      after being reflected from the given surface point.

  Raises:
    ValueError: if the shape of `point_light_position`,
    `surface_point_position`, `surface_point_normal`, or `observation_point` is
    not supported.
    InvalidArgumentError: if 'surface_point_normal' is not normalized.
  """
  with tf.compat.v1.name_scope(name, "estimate_radiance", [
      point_light_radiance, point_light_position, surface_point_position,
      surface_point_normal, observation_point, brdf
  ]):
    point_light_radiance = tf.convert_to_tensor(value=point_light_radiance)
    point_light_position = tf.convert_to_tensor(value=point_light_position)
    surface_point_position = tf.convert_to_tensor(value=surface_point_position)
    surface_point_normal = tf.convert_to_tensor(value=surface_point_normal)
    observation_point = tf.convert_to_tensor(value=observation_point)

    shape.check_static(
        tensor=point_light_position,
        tensor_name="point_light_position",
        has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=surface_point_position,
        tensor_name="surface_point_position",
        has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=surface_point_normal,
        tensor_name="surface_point_normal",
        has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=observation_point,
        tensor_name="observation_point",
        has_dim_equals=(-1, 3))
    shape.compare_batch_dimensions(
        tensors=(surface_point_position, surface_point_normal,
                 observation_point),
        tensor_names=("surface_point_position", "surface_point_normal",
                      "observation_point"),
        last_axes=-2,
        broadcast_compatible=True)
    shape.compare_batch_dimensions(
        tensors=(point_light_radiance, point_light_position),
        tensor_names=("point_light_radiance", "point_light_position"),
        last_axes=-2,
        broadcast_compatible=True)
    surface_point_normal = asserts.assert_normalized(surface_point_normal)

    # Get the number of lights dimensions (B1,...,Bm).
    lights_num_dimensions = max(
        len(point_light_radiance.shape), len(point_light_position.shape)) - 1
    # Reshape the other parameters so they can be broadcasted to the output of
    # shape [A1,...,An, B1,...,Bm, K].
    surface_point_position = tf.reshape(
        surface_point_position,
        surface_point_position.shape[:-1] + (1,) * lights_num_dimensions + (3,))
    surface_point_normal = tf.reshape(
        surface_point_normal,
        surface_point_normal.shape[:-1] + (1,) * lights_num_dimensions + (3,))
    observation_point = tf.reshape(
        observation_point,
        observation_point.shape[:-1] + (1,) * lights_num_dimensions + (3,))

    light_to_surface_point = surface_point_position - point_light_position
    distance_light_surface_point = tf.norm(
        tensor=light_to_surface_point, axis=-1, keepdims=True)
    incoming_light_direction = tf.math.l2_normalize(
        light_to_surface_point, axis=-1)
    surface_to_observation_point = observation_point - surface_point_position
    outgoing_light_direction = tf.math.l2_normalize(
        surface_to_observation_point, axis=-1)
    brdf_value = brdf(incoming_light_direction, outgoing_light_direction,
                      surface_point_normal)
    incoming_light_dot_surface_normal = vector.dot(-incoming_light_direction,
                                                   surface_point_normal)
    outgoing_light_dot_surface_normal = vector.dot(outgoing_light_direction,
                                                   surface_point_normal)

    estimated_radiance = (point_light_radiance * \
                          brdf_value * incoming_light_dot_surface_normal) / \
        (4. * math.pi * tf.math.square(distance_light_surface_point))

    if reflected_light_fall_off:
      distance_surface_observation_point = tf.norm(
          tensor=surface_to_observation_point, axis=-1, keepdims=True)
      estimated_radiance = estimated_radiance / \
          tf.math.square(distance_surface_observation_point)

    # Create a condition for checking whether the light or observation point are
    # behind the surface.
    min_dot = tf.minimum(incoming_light_dot_surface_normal,
                         outgoing_light_dot_surface_normal)
    common_shape = shape.get_broadcasted_shape(min_dot.shape,
                                               estimated_radiance.shape)
    d_val = lambda dim: 1 if dim is None else tf.compat.v1.dimension_value(dim)
    common_shape = [d_val(dim) for dim in common_shape]
    condition = tf.broadcast_to(tf.greater_equal(min_dot, 0.0), common_shape)

    return tf.compat.v1.where(condition, estimated_radiance,
                              tf.zeros_like(estimated_radiance))


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
