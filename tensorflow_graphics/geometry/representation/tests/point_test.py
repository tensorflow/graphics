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
"""Tests for point."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.geometry.representation import point
from tensorflow_graphics.util import test_case


class PointTest(test_case.TestCase):

  @parameterized.parameters(
      ((None,), (None,), (None,)),
      ((None, None), (None, None), (None, None)),
      ((3,), (3,), (3,)),
      ((2, 3), (2, 3), (2, 3)),
      ((1, 2, 3), (2, 2, 3), (2, 3)),
  )
  def test_distance_to_ray_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(point.distance_to_ray, shapes)

  @parameterized.parameters(
      ("must have the same number of dimensions", (0,), (None,), (None,)),
      ("must have the same number of dimensions", (None,), (0,), (None,)),
      ("must have the same number of dimensions", (None,), (None,), (0,)),
      ("must have the same number of dimensions", (0,), (1,), (1,)),
      ("must have the same number of dimensions", (2,), (1,), (2,)),
      ("must have the same number of dimensions", (2, 3), (2, 2), (2, 0)),
      ("Not all batch dimensions are broadcast-compatible.", (0, 3), (3, 3),
       (3, 3)),
      ("Not all batch dimensions are broadcast-compatible.", (1, 3), (2, 3),
       (3, 3)),
  )
  def test_distance_to_ray_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(point.distance_to_ray, error_msg, shapes)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_distance_to_ray_jacobian_random(self):
    """Tests the Jacobian of the distance to ray function."""
    eps = sys.float_info.epsilon
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    point_init = np.random.random(size=tensor_shape + [3])
    origin_init = np.random.random(size=tensor_shape + [3])
    direction_init = np.random.random(size=tensor_shape + [3])
    direction_init /= np.maximum(
        np.linalg.norm(direction_init, axis=-1, keepdims=True), eps)

    self.assert_jacobian_is_correct_fn(
        lambda x: point.distance_to_ray(x, origin_init, direction_init),
        [point_init])
    self.assert_jacobian_is_correct_fn(
        lambda x: point.distance_to_ray(point_init, x, direction_init),
        [origin_init])
    self.assert_jacobian_is_correct_fn(
        lambda x: point.distance_to_ray(point_init, origin_init, x),
        [direction_init])

  @parameterized.parameters(
      (((1., 1., 1.), (-10., 1., 1.), (1., 0., 0.)), ((0.,),)),)
  def test_distance_to_ray_preset(self, test_inputs, test_outputs):
    """Tests 0 distance from point to ray."""
    self.assert_output_is_correct(point.distance_to_ray, test_inputs,
                                  test_outputs)

  def test_distance_to_ray_random(self):
    """Tests distance point to ray."""
    eps = sys.float_info.epsilon
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    random_origin = np.random.random(size=tensor_shape + [3])
    random_direction = np.random.random(size=tensor_shape + [3])
    random_direction /= np.maximum(
        np.linalg.norm(random_direction, axis=-1, keepdims=True), eps)
    random_distances = np.random.random(size=tensor_shape + [1]) * 1000
    x, y, _, _ = np.split(random_direction, (1, 2, 3), axis=-1)

    # Find perpendicular vector.
    perp = np.concatenate((-y, x, np.zeros_like(x)), axis=-1)
    perp /= np.maximum(np.linalg.norm(perp, axis=-1, keepdims=True), eps)
    # Choose a point on perpendicular unit vector at the good distance.
    points = random_origin + perp * random_distances
    distance = point.distance_to_ray(points, random_origin, random_direction)

    self.assertAllClose(random_distances, distance)

  @parameterized.parameters(
      ((None,), (None,), (None,)),
      ((None, None), (None, None), (None, None)),
      ((3,), (3,), (3,)),
      ((2, 3), (2, 3), (2, 3)),
      ((1, 2, 3), (2, 2, 3), (2, 3)),
  )
  def test_project_to_ray_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(point.project_to_ray, shapes)

  @parameterized.parameters(
      ("must have the same number of dimensions", (0,), (None,), (None,)),
      ("must have the same number of dimensions", (None,), (0,), (None,)),
      ("must have the same number of dimensions", (None,), (None,), (0,)),
      ("must have the same number of dimensions", (0,), (1,), (1,)),
      ("must have the same number of dimensions", (2,), (1,), (2,)),
      ("must have the same number of dimensions", (2, 3), (2, 2), (2, 0)),
      ("Not all batch dimensions are broadcast-compatible.", (0, 3), (3, 3),
       (3, 3)),
      ("Not all batch dimensions are broadcast-compatible.", (1, 3), (2, 3),
       (3, 3)),
  )
  def test_project_to_ray_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(point.project_to_ray, error_msg, shapes)

  @parameterized.parameters(
      (((1., 1., 1.), (-10., 1., 1.), (1., 0., 0.)), ((1., 1., 1.),)),)
  def test_project_to_ray_preset(self, test_inputs, test_outputs):
    """Tests that a point on a ray projects to itself."""
    self.assert_output_is_correct(point.project_to_ray, test_inputs,
                                  test_outputs)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_project_to_ray_jacobian_random(self):
    """Tests the Jacobian of the distance to ray function."""
    eps = sys.float_info.epsilon
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    point_init = np.random.random(size=tensor_shape + [3])
    origin_init = np.random.random(size=tensor_shape + [3])
    direction_init = np.random.random(size=tensor_shape + [3])
    direction_init /= np.maximum(
        np.linalg.norm(direction_init, axis=-1, keepdims=True), eps)
    point_tensor = tf.convert_to_tensor(value=point_init)
    origin_tensor = tf.convert_to_tensor(value=origin_init)
    direction_tensor = tf.convert_to_tensor(value=direction_init)

    self.assert_jacobian_is_correct_fn(
        lambda x: point.project_to_ray(x, origin_tensor, direction_tensor),
        [point_init])
    self.assert_jacobian_is_correct_fn(
        lambda x: point.project_to_ray(point_tensor, x, direction_tensor),
        [origin_init])
    self.assert_jacobian_is_correct_fn(
        lambda x: point.project_to_ray(point_tensor, origin_tensor, x),
        [direction_init])

  def test_project_to_ray_random(self):
    """Tests the function that projects a point to a ray."""
    eps = sys.float_info.epsilon
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    random_origin = np.random.random(size=tensor_shape + [3])
    random_direction = np.random.random(size=tensor_shape + [3])
    random_direction /= np.maximum(
        np.linalg.norm(random_direction, axis=-1, keepdims=True), eps)
    random_distances = np.random.random(size=tensor_shape + [1]) * 1000
    x, y, _, _ = np.split(random_direction, (1, 2, 3), axis=-1)

    # Find perpendicular vector.
    perp = np.concatenate((-y, x, np.zeros_like(x)), axis=-1)
    perp /= np.maximum(np.linalg.norm(perp, axis=-1, keepdims=True), eps)
    # Choose a point on perpendicular unit vector at the good distance.
    points = random_origin + perp * random_distances
    points = point.project_to_ray(points, random_origin, random_direction)

    self.assertAllClose(random_origin, points)


if __name__ == "__main__":
  test_case.main()
