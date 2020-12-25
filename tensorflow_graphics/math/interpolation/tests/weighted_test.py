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
"""Tests for google3.third_party.py.tensorflow_graphics.interpolation.weighted."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.math.interpolation import weighted
from tensorflow_graphics.util import test_case


class WeightedTest(test_case.TestCase):

  def _get_tensors_from_shapes(self, num_points, dim_points, num_outputs,
                               num_pts_to_interpolate):
    points = np.random.uniform(size=(num_points, dim_points))
    weights = np.random.uniform(size=(num_outputs, num_pts_to_interpolate))
    indices = np.asarray([
        np.random.permutation(num_points)[:num_pts_to_interpolate].tolist()
        for _ in range(num_outputs)
    ])
    indices = np.expand_dims(indices, axis=-1)
    return points, weights, indices

  @parameterized.parameters(
      (3, 4, 2, 3),
      (5, 4, 5, 3),
      (5, 6, 5, 5),
      (2, 6, 5, 1),
  )
  def test_interpolate_exception_not_raised(self, dim_points, num_points,
                                            num_outputs,
                                            num_pts_to_interpolate):
    """Tests whether exceptions are not raised for compatible shapes."""
    points, weights, indices = self._get_tensors_from_shapes(
        num_points, dim_points, num_outputs, num_pts_to_interpolate)

    self.assert_exception_is_not_raised(
        weighted.interpolate,
        shapes=[],
        points=points,
        weights=weights,
        indices=indices,
        normalize=True)

  @parameterized.parameters(
      ("must have a rank greater than 1", ((3,), (None, 2), (None, 2, 0))),
      ("must have a rank greater than 1", ((None, 3), (None, 2), (1,))),
      ("must have exactly 1 dimensions in axis -1", ((None, 3), (None, 2),
                                                     (None, 2, 2))),
      ("must have the same number of dimensions", ((None, 3), (None, 2),
                                                   (None, 3, 1))),
      ("Not all batch dimensions are broadcast-compatible.",
       ((None, 3), (None, 5, 2), (None, 4, 2, 1))),
  )
  def test_interpolate_exception_raised(self, error_msg, shapes):
    """Tests whether exceptions are raised for incompatible shapes."""
    self.assert_exception_is_raised(
        weighted.interpolate, error_msg, shapes=shapes, normalize=False)

  @parameterized.parameters(
      (((-1.0, 1.0), (1.0, 1.0), (3.0, 1.0), (-1.0, -1.0), (1.0, -1.0),
        (3.0, -1.0)), ((0.25, 0.25, 0.25, 0.25), (0.5, 0.5, 0.0, 0.0)),
       (((0,), (1,), (3,), (4,)), ((1,), (2,), (4,),
                                   (5,))), False, ((0.0, 0.0), (2.0, 1.0))),)
  def test_interpolate_preset(self, points, weights, indices, _, out):
    """Tests whether interpolation results are correct."""
    weights = tf.convert_to_tensor(value=weights)

    result_unnormalized = weighted.interpolate(
        points=points, weights=weights, indices=indices, normalize=False)
    result_normalized = weighted.interpolate(
        points=points, weights=2.0 * weights, indices=indices, normalize=True)
    estimated_unnormalized = self.evaluate(result_unnormalized)
    estimated_normalized = self.evaluate(result_normalized)

    self.assertAllClose(estimated_unnormalized, out)
    self.assertAllClose(estimated_normalized, out)

  @parameterized.parameters(
      (3, 4, 2, 3),
      (5, 4, 5, 3),
      (5, 6, 5, 5),
      (2, 6, 5, 1),
  )
  def test_interpolate_negative_weights_raised(self, dim_points, num_points,
                                               num_outputs,
                                               num_pts_to_interpolate):
    """Tests whether exception is raised when weights are negative."""
    points, weights, indices = self._get_tensors_from_shapes(
        num_points, dim_points, num_outputs, num_pts_to_interpolate)
    weights *= -1.0

    with self.assertRaises(tf.errors.InvalidArgumentError):
      result = weighted.interpolate(
          points=points, weights=weights, indices=indices, normalize=True)
      self.evaluate(result)

  @parameterized.parameters(
      (((-1.0, 1.0), (1.0, 1.0), (3.0, 1.0), (-1.0, -1.0), (1.0, -1.0),
        (3.0, -1.0)), ((1.0, -1.0, 1.0, -1.0), (0.0, 0.0, 0.0, 0.0)),
       (((0,), (1,), (3,), (4,)), ((1,), (2,), (4,), (5,))), ((0.0, 0.0),
                                                              (0.0, 0.0))))
  def test_interp_unnormalizable_raised_(self, points, weights, indices, _):
    """Tests whether exception is raised when weights are unnormalizable."""
    with self.assertRaises(tf.errors.InvalidArgumentError):
      result = weighted.interpolate(
          points=points,
          weights=weights,
          indices=indices,
          normalize=True,
          allow_negative_weights=True)
      self.evaluate(result)

  @parameterized.parameters(
      (3, 4, 2, 3),
      (5, 4, 5, 3),
      (5, 6, 5, 5),
      (2, 6, 5, 1),
  )
  def test_interpolate_jacobian_random(self, dim_points, num_points,
                                       num_outputs, num_pts_to_interpolate):
    """Tests whether jacobian is correct."""
    points_np, weights_np, indices_np = self._get_tensors_from_shapes(
        num_points, dim_points, num_outputs, num_pts_to_interpolate)

    def interpolate_fn(points, weights):
      return weighted.interpolate(
          points=points, weights=weights, indices=indices_np, normalize=True)

    self.assert_jacobian_is_correct_fn(interpolate_fn, [points_np, weights_np])

  @parameterized.parameters(
      ((3, 2), (2, 2)),
      ((None, 3, 2), (None, 1, 2)),
      ((10, 5, 3, 2), (10, 5, 2, 2)),
  )
  def test_get_barycentric_coordinates_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(weighted.get_barycentric_coordinates,
                                        shapes)

  @parameterized.parameters(
      ("triangle_vertices must have exactly 2 dimensions in axis -1", (3, 1),
       (1, 2)),
      ("triangle_vertices must have exactly 3 dimensions in axis -2", (2, 2),
       (1, 2)),
      ("pixels must have exactly 2 dimensions in axis -1", (3, 2), (1, 3)),
      ("Not all batch dimensions are broadcast-compatible", (5, 3, 2),
       (2, 10, 2)),
  )
  def test_get_barycentric_coordinates_exception_raised(self, error_msg,
                                                        *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(weighted.get_barycentric_coordinates,
                                    error_msg, shape)

  def test_get_barycentric_coordinates_jacobian_random(self):
    """Tests the Jacobian of get_barycentric_coordinates."""
    tensor_size = np.random.randint(2)
    tensor_shape = np.random.randint(1, 2, size=(tensor_size)).tolist()
    triangle_vertices_init = 0.4 * np.random.random(
        tensor_shape + [3, 2]).astype(np.float64) - 0.2
    triangle_vertices_init += np.array(
        ((0.25, 0.25), (0.5, 0.75), (0.75, 0.25)))
    pixels_init = np.random.random(tensor_shape + [3, 2]).astype(np.float64)

    barycentric_fn = weighted.get_barycentric_coordinates

    self.assert_jacobian_is_correct_fn(
        lambda vertices, pixels: barycentric_fn(vertices, pixels)[0],
        [triangle_vertices_init, pixels_init])

  def test_get_barycentric_coordinates_normalized(self):
    """Tests whether the barycentric coordinates are normalized."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    num_pixels = np.random.randint(1, 10)
    pixels_shape = tensor_shape + [num_pixels]
    triangle_vertices = np.random.random(tensor_shape + [3, 2])
    pixels = np.random.random(pixels_shape + [2])

    barycentric_coordinates, _ = weighted.get_barycentric_coordinates(
        triangle_vertices, pixels)
    barycentric_coordinates_sum = tf.reduce_sum(
        input_tensor=barycentric_coordinates, axis=-1)

    self.assertAllClose(barycentric_coordinates_sum, np.full(pixels_shape, 1.0))


if __name__ == "__main__":
  test_case.main()
