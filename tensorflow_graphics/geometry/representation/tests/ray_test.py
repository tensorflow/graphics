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
r"""Tests for ray."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.geometry.representation import ray
from tensorflow_graphics.util import test_case

FLAGS = flags.FLAGS


class RayTest(test_case.TestCase):

  def _generate_random_example(self):
    num_cameras = 4
    num_keypoints = 3
    batch_size = 2
    self.points_values = np.random.random_sample((batch_size, num_keypoints, 3))
    points_expanded_values = np.expand_dims(self.points_values, axis=-2)
    startpoints_values = np.random.random_sample(
        (batch_size, num_keypoints, num_cameras, 3))

    difference = points_expanded_values - startpoints_values
    difference_norm = np.sqrt((difference * difference).sum(axis=-1))
    direction = difference / np.expand_dims(difference_norm, axis=-1)

    self.startpoints_values = points_expanded_values - 0.5 * direction
    self.endpoints_values = points_expanded_values + 0.5 * direction
    self.weights_values = np.ones((batch_size, num_keypoints, num_cameras))

    # Wrap these with identies because some assert_* ops look at the constant
    # tensor values and mark these as unfeedable.
    self.points = tf.identity(tf.convert_to_tensor(value=self.points_values))
    self.startpoints = tf.identity(
        tf.convert_to_tensor(value=self.startpoints_values))
    self.endpoints = tf.identity(
        tf.convert_to_tensor(value=self.endpoints_values))
    self.weights = tf.identity(tf.convert_to_tensor(value=self.weights_values))

  @parameterized.parameters(
      ("Not all batch dimensions are identical.", (4, 3), (5, 3), (4,)),
      ("must have exactly 3 dimensions in axis", (4, 2), (4, 2), (4,)),
      ("must have a rank greater than 1", (3,), (3,), (None,)),
      ("must have greater than 1 dimensions in axis -2", (1, 3), (1, 3), (1,)),
      ("Not all batch dimensions are identical.", (2, 4, 3), (2, 4, 3), (2, 5)),
  )
  def test_triangulate_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(ray.triangulate, error_msg, shapes)

  @parameterized.parameters(
      ((4, 3), (4, 3), (4,)),
      ((5, 4, 3), (5, 4, 3), (5, 4)),
      ((6, 5, 4, 3), (6, 5, 4, 3), (6, 5, 4)),
  )
  def test_triangulate_exception_is_not_raised(self, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_not_raised(ray.triangulate, shapes)

  def test_triangulate_jacobian_is_correct(self):
    """Tests that Jacobian is correct."""
    self._generate_random_example()

    self.assert_jacobian_is_correct_fn(
        lambda x: ray.triangulate(x, self.endpoints, self.weights),
        [self.startpoints_values])
    self.assert_jacobian_is_correct_fn(
        lambda x: ray.triangulate(self.startpoints, x, self.weights),
        [self.endpoints_values])
    self.assert_jacobian_is_correct_fn(
        lambda x: ray.triangulate(self.startpoints, self.endpoints, x),
        [self.weights_values])

  def test_triangulate_jacobian_is_finite(self):
    """Tests that Jacobian is finite."""
    self._generate_random_example()

    self.assert_jacobian_is_finite_fn(
        lambda x: ray.triangulate(x, self.endpoints, self.weights),
        [self.startpoints_values])
    self.assert_jacobian_is_finite_fn(
        lambda x: ray.triangulate(self.startpoints, x, self.weights),
        [self.endpoints_values])
    self.assert_jacobian_is_finite_fn(
        lambda x: ray.triangulate(self.startpoints, self.endpoints, x),
        [self.weights_values])

  def test_triangulate_random(self):
    """Tests that original points are recovered by triangualtion."""
    self._generate_random_example()
    test_inputs = (self.startpoints, self.endpoints, self.weights)
    test_outputs = (self.points_values,)

    self.assert_output_is_correct(
        ray.triangulate,
        test_inputs,
        test_outputs,
        rtol=1e-05,
        atol=1e-08,
        tile=False)

  def test_negative_weights_exception_raised(self):
    """Tests that exceptions are properly raised."""
    self._generate_random_example()
    self.weights = -1.0 * tf.ones_like(self.weights, dtype=tf.float64)

    with self.assertRaises(tf.errors.InvalidArgumentError):
      points = ray.triangulate(self.startpoints, self.endpoints, self.weights)
      self.evaluate(points)

  def test_less_that_two_nonzero_weights_exception_raised(self):
    """Tests that exceptions are properly raised."""
    self._generate_random_example()
    self.weights = tf.convert_to_tensor(
        value=np.array([[[1., 1., 0., 0.], [1., 1., 0., 0.], [1., 1., 0., 0.]],
                        [[1., 1., 0., 0.], [1., 1., 0., 0.], [1., 0., 0., 0.]]],
                       dtype=np.float64))

    with self.assertRaises(tf.errors.InvalidArgumentError):
      points = ray.triangulate(self.startpoints, self.endpoints, self.weights)
      self.evaluate(points)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis 0", (2,), (1,), (3,), (3,)),
      ("must have a rank of 1", (2, 3), (1,), (3,), (3,)),
      ("must have exactly 1 dimensions in axis 0", (3,), (2,), (3,), (3,)),
      ("must have a rank of 1", (3,), (2, 1), (3,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (1,), (2,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (1,), (3,), (2,)),
      ("Not all batch dimensions are identical.", (3,), (1,), (3,), (2, 3)),
  )
  def test_intersection_ray_sphere_shape_raised(self, error_msg, *shapes):
    """tests that exceptions are raised when shapes are not supported."""
    self.assert_exception_is_raised(ray.intersection_ray_sphere, error_msg,
                                    shapes)

  @parameterized.parameters(
      ((3,), (1,), (3,), (3,)),
      ((3), (1), (None, 3), (None, 3)),
  )
  def test_intersection_ray_sphere_shape_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised on supported shapes."""
    self.assert_exception_is_not_raised(ray.intersection_ray_sphere, shapes)

  def test_intersection_ray_sphere_exception_raised(self):
    """Tests that exceptions are properly raised."""
    sphere_center = np.random.uniform(size=(3,))
    point_on_ray = np.random.uniform(size=(3,))
    sample_ray = np.random.uniform(size=(3,))
    normalized_sample_ray = sample_ray / np.linalg.norm(sample_ray, axis=-1)
    positive_sphere_radius = np.random.uniform(
        sys.float_info.epsilon, 1.0, size=(1,))
    negative_sphere_radius = np.random.uniform(-1.0, 0.0, size=(1,))

    with self.subTest(name="positive_radius"):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(
            ray.intersection_ray_sphere(sphere_center, negative_sphere_radius,
                                        normalized_sample_ray, point_on_ray))

    with self.subTest(name="normalized_ray"):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(
            ray.intersection_ray_sphere(sphere_center, positive_sphere_radius,
                                        sample_ray, point_on_ray))

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_intersection_ray_sphere_jacobian_random(self):
    """Test the Jacobian of the intersection_ray_sphere function."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    sphere_center_init = np.random.uniform(0.0, 1.0, size=(3,))
    sphere_radius_init = np.random.uniform(10.0, 11.0, size=(1,))
    ray_init = np.random.uniform(size=tensor_shape + [3])
    ray_init /= np.linalg.norm(ray_init, axis=-1, keepdims=True)
    point_on_ray_init = np.random.uniform(0.0, 1.0, size=tensor_shape + [3])

    def intersection_ray_sphere_position(sphere_center, sphere_radius,
                                         input_ray, point_on_ray):
      y_p, _ = ray.intersection_ray_sphere(sphere_center, sphere_radius,
                                           input_ray, point_on_ray)
      return y_p

    def intersection_ray_sphere_normal(sphere_center, sphere_radius, input_ray,
                                       point_on_ray):
      _, y_n = ray.intersection_ray_sphere(sphere_center, sphere_radius,
                                           input_ray, point_on_ray)
      return y_n

    self.assert_jacobian_is_correct_fn(
        intersection_ray_sphere_position,
        [sphere_center_init, sphere_radius_init, ray_init, point_on_ray_init])
    self.assert_jacobian_is_correct_fn(
        intersection_ray_sphere_normal,
        [sphere_center_init, sphere_radius_init, ray_init, point_on_ray_init])

  @parameterized.parameters(
      (((0.0, 0.0, 3.0), (1.0,), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
       (((0.0, 0.0, 2.0), (0.0, 0.0, 4.0)), ((0.0, 0.0, -1.0),
                                             (0.0, 0.0, 1.0)))),
      (((0.0, 0.0, 3.0), (1.0,), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
       (((1.0, 0.0, 3.0), (1.0, 0.0, 3.0)), ((1.0, 0.0, 0.0),
                                             (1.0, 0.0, 0.0)))),
  )
  def test_intersection_ray_sphere_preset(self, test_inputs, test_outputs):
    self.assert_output_is_correct(
        ray.intersection_ray_sphere, test_inputs, test_outputs, tile=False)


if __name__ == "__main__":
  test_case.main()
