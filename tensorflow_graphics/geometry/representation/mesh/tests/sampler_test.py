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
"""Tests for uniform mesh sampler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from six.moves import range
import tensorflow as tf

from tensorflow_graphics.geometry.representation.mesh import sampler
from tensorflow_graphics.geometry.representation.mesh.tests import mesh_test_utils
from tensorflow_graphics.util import test_case


class MeshSamplerTest(test_case.TestCase):

  def setUp(self):
    """Sets up default parameters."""
    super(MeshSamplerTest, self).setUp()
    self._test_sigma_compare_tolerance = 4.0

  def compare_poisson_equivalence(self, expected, actual):
    """Performs equivalence check on Poisson-distributed random variables."""
    delta = np.sqrt(expected) * self._test_sigma_compare_tolerance
    self.assertAllClose(expected, actual, atol=delta)

  # Tests for generate_random_face_indices
  @parameterized.parameters(
      (((), (2,)), (tf.int32, tf.float32)),
      (((), (None, 1)), (tf.int64, tf.float32)),
      (((), (None, 3, 4)), (tf.int64, tf.float64)),
  )
  def test_random_face_indices_exception_not_raised(self, shapes, dtypes):
    """Tests that shape exceptions are not raised for random_face_indices."""
    self.assert_exception_is_not_raised(sampler.generate_random_face_indices,
                                        shapes, dtypes)

  @parameterized.parameters(
      ("face_weights must have a rank greater than 0.", (), ()),
      ("num_samples must have a rank of 0.", (None,), (1, 2)),
      ("num_samples must have a rank of 0.", (4, 2), (1,)),
  )
  def test_random_face_indices_shape_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised for random_face_indices."""
    self.assert_exception_is_raised(sampler.generate_random_face_indices,
                                    error_msg, shapes)

  def test_negative_weights_random_face_indices_exception(self):
    """Test for exception for random_face_indices with negative weights."""
    face_wts = np.array([0.1, -0.1], dtype=np.float32)
    num_samples = 10
    error_msg = "Condition x >= y did not hold."
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, error_msg):
      sampler.generate_random_face_indices(num_samples, face_weights=face_wts)

  @parameterized.parameters(
      ((0., 0.), 10, (5, 5)),
      ((0., 0.0, 0.001), 100, (0, 0, 100)),
      ((0.1, 0.2, 0.3), 1000, (167, 333, 500)),
  )
  def test_random_face_indices(self, face_weights, num_samples, expected):
    """Test for generate_random_face_indices."""
    face_weights = np.array(face_weights, dtype=np.float32)
    expected = np.array(expected, dtype=np.intp)
    sample_faces = sampler.generate_random_face_indices(num_samples,
                                                        face_weights)

    self.assertEqual(sample_faces.shape[0], num_samples)
    self.compare_poisson_equivalence(expected, tf.math.bincount(sample_faces))

  # Tests for generate_random_barycentric_coordinates
  @parameterized.parameters(
      ((1,), (tf.int32)),
      ((None,), (tf.int64)),
  )
  def test_random_barycentric_coordinates_exception_not_raised(
      self, shapes, dtypes):
    """Tests that shape exceptions are not raised for random_barycentric_coordinates."""
    self.assert_exception_is_not_raised(
        sampler.generate_random_barycentric_coordinates, shapes, dtypes)

  @parameterized.parameters(
      ("sample_shape must have a rank of 1.", ()),
      ("sample_shape must have a rank of 1.", (4, None)),
  )
  def test_random_barycentric_coordinates_shape_exception_raised(
      self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised for random_barycentric_coordinates."""
    self.assert_exception_is_raised(
        sampler.generate_random_barycentric_coordinates, error_msg, shapes)

  @parameterized.parameters(
      ((5,),),
      ((10, 1, 3),),
  )
  def test_random_barycentric_coordinates(self, sample_shape):
    """Test for generate_random_barycentric_coordinates."""
    sample_shape = np.array(sample_shape, dtype=np.intp)
    random_coordinates = sampler.generate_random_barycentric_coordinates(
        sample_shape=sample_shape)
    coordinate_sum = tf.reduce_sum(input_tensor=random_coordinates, axis=-1)
    expected_coordinate_sum = np.ones(shape=sample_shape)
    self.assertAllClose(expected_coordinate_sum, coordinate_sum)

  # Tests for weighted_random_sample_triangle_mesh
  @parameterized.parameters(
      (((4, 3), (5, 3), (), (5,)),
       (tf.float32, tf.int32, tf.int32, tf.float32)),
      (((None, 3), (None, 3), (), (None,)),
       (tf.float32, tf.int32, tf.int32, tf.float32)),
      (((3, None, 3), (3, None, 3), (), (3, None)),
       (tf.float32, tf.int64, tf.int64, tf.float64)),
      (((3, 6, 5), (3, 5, 3), (), (3, 5)),
       (tf.float64, tf.int32, tf.int32, tf.float32)),
  )
  def test_weighted_sampler_exception_not_raised(self, shapes, dtypes):
    """Tests that the shape exceptions are not raised for weighted sampler."""
    self.assert_exception_is_not_raised(
        sampler.weighted_random_sample_triangle_mesh, shapes, dtypes)

  @parameterized.parameters(
      ("vertex_attributes must have a rank greater than 1.", (3,), (None, 3),
       (), (None, 3)),
      ("faces must have a rank greater than 1.", (5, 2), (None,), (), (None,)),
      ("face_weights must have a rank greater than 0.", (1, None, 3), (None, 3),
       (), ()),
      ("Not all batch dimensions are identical", (4, 4, 2), (3, 5, 3), (),
       (3, 5)),
      ("Not all batch dimensions are identical", (4, 2), (5, 3), (), (4,)),
  )
  def test_weighted_sampler_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised for weighted sampler."""
    self.assert_exception_is_raised(
        sampler.weighted_random_sample_triangle_mesh, error_msg, shapes)

  def test_weighted_sampler_negative_weights(self):
    """Test for exception with negative weights."""
    vertices, faces = mesh_test_utils.create_square_triangle_mesh()
    face_wts = np.array([-0.3, 0.1, 0.5, 0.6], dtype=np.float32)
    num_samples = 10
    error_msg = "Condition x >= y did not hold."
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, error_msg):
      sampler.weighted_random_sample_triangle_mesh(
          vertices, faces, num_samples, face_weights=face_wts)

  def test_weighted_random_sample(self):
    """Test for provided face weights."""
    faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int32)
    vertex_attributes = np.array([[0.], [0.], [1.], [1.]], dtype=np.float32)

    # Equal face weights, mean of sampled attributes = 0.5.
    expected_mean = np.array([0.5], dtype=np.float32)
    sample_pts, _ = sampler.weighted_random_sample_triangle_mesh(
        vertex_attributes, faces, num_samples=1000000, face_weights=(0.5, 0.5))
    self.assertAllClose(
        expected_mean,
        tf.reduce_mean(input_tensor=sample_pts, axis=-2),
        atol=1e-3)
    # Face weights biased towards second face, mean > 0.5
    sample_pts, _ = sampler.weighted_random_sample_triangle_mesh(
        vertex_attributes, faces, num_samples=1000000, face_weights=(0.2, 0.8))
    self.assertGreater(
        tf.reduce_mean(input_tensor=sample_pts, axis=-2), expected_mean)

  def test_weighted_sampler_jacobian_random(self):
    """Test the Jacobian of weighted triangle random sampler."""
    tensor_vertex_size = np.random.randint(1, 3)
    tensor_out_shape = np.random.randint(1, 5, size=tensor_vertex_size)
    tensor_out_shape = tensor_out_shape.tolist()
    vertex_axis = np.array(((0., 0., 1), (1., 0., 0.), (0., 1., 0.),
                            (0., 0., -1.), (-1., 0., 0.), (0., -1., 0.)),
                           dtype=np.float32)
    vertex_axis = vertex_axis.reshape([1] * tensor_vertex_size + [6, 3])
    faces = np.array(((0, 1, 2), (0, 2, 4), (0, 4, 5), (0, 5, 1), (3, 2, 1),
                      (3, 4, 2), (3, 5, 4), (3, 1, 5)),
                     dtype=np.int32)
    faces = faces.reshape([1] * tensor_vertex_size + [8, 3])
    index_init = np.tile(faces, tensor_out_shape + [1, 1])
    vertex_scale = np.random.uniform(0.5, 5., tensor_out_shape + [1] * 2)
    vertex_init = vertex_axis * vertex_scale
    index_tensor = tf.convert_to_tensor(value=index_init)
    face_weights = np.random.uniform(size=index_init.shape[:index_init.ndim -
                                                           1])
    weights_tensor = tf.convert_to_tensor(value=face_weights)

    num_samples = np.random.randint(10, 100)

    def sampler_fn(vertices):
      sample_pts, _ = sampler.weighted_random_sample_triangle_mesh(
          vertices,
          index_tensor,
          num_samples,
          weights_tensor,
          seed=[0, 1],
          stateless=True)
      return sample_pts

    self.assert_jacobian_is_correct_fn(
        sampler_fn, [vertex_init], atol=1e-4, delta=1e-4)

  # Tests for area_weighted_random_sample_triangle_mesh
  @parameterized.parameters(
      (((4, 3), (5, 3), ()), (tf.float32, tf.int32, tf.int32)),
      (((None, 3), (None, 3), ()), (tf.float32, tf.int32, tf.int32)),
      (((3, None, 3), (3, None, 3), ()), (tf.float32, tf.int64, tf.int64)),
      # Test for vertex attributes + positions
      (((3, 6, 5), (3, 5, 3), (), (3, 6, 3)),
       (tf.float64, tf.int32, tf.int32, tf.float32)),
  )
  def test_area_sampler_exception_not_raised(self, shapes, dtypes):
    """Tests that the shape exceptions are not raised for area weighted sampler."""
    self.assert_exception_is_not_raised(
        sampler.area_weighted_random_sample_triangle_mesh, shapes, dtypes)

  @parameterized.parameters(
      ("vertices must have a rank greater than 1.", (3,), (None, 3), ()),
      ("vertices must have greater than 2 dimensions in axis -1.", (5, 2),
       (None, 3), ()),
      ("vertex_positions must have exactly 3 dimensions in axis -1.", (5, 3),
       (None, 3), (), (3, 2)),
  )
  def test_area_sampler_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised for area weighted sampler."""
    self.assert_exception_is_raised(
        sampler.area_weighted_random_sample_triangle_mesh, error_msg, shapes)

  def test_area_sampler_distribution(self):
    """Test for area weighted sampler distribution."""
    vertices, faces = mesh_test_utils.create_single_triangle_mesh()
    vertices = np.repeat(np.expand_dims(vertices, axis=0), 3, axis=0)
    faces = np.repeat(np.expand_dims(faces, axis=0), 3, axis=0)
    num_samples = 5000
    sample_pts, _ = sampler.area_weighted_random_sample_triangle_mesh(
        vertices, faces, num_samples)

    for i in range(3):
      samples = sample_pts[i, ...]
      self.assertEqual(samples.shape[-2], num_samples)
      # Test distribution in 4 quadrants of [0,1]x[0,1]
      v = samples[:, :2] < [0.5, 0.5]
      not_v = tf.logical_not(v)
      quad00 = tf.math.count_nonzero(tf.reduce_all(input_tensor=v, axis=-1))
      quad11 = tf.math.count_nonzero(tf.reduce_all(input_tensor=not_v, axis=-1))
      quad01 = tf.math.count_nonzero(
          tf.reduce_all(
              input_tensor=tf.stack((v[:, 0], not_v[:, 1]), axis=1), axis=-1))
      quad10 = tf.math.count_nonzero(
          tf.reduce_all(
              input_tensor=tf.stack((not_v[:, 0], v[:, 1]), axis=1), axis=-1))
      counts = tf.stack((quad00, quad01, quad10, quad11), axis=0)
      expected = np.array(
          [num_samples / 2, num_samples / 4, num_samples / 4, 0],
          dtype=np.float32)
      self.compare_poisson_equivalence(expected, counts)

  def test_face_distribution(self):
    """Test for distribution of face indices with area weighted sampler."""
    vertices, faces = mesh_test_utils.create_square_triangle_mesh()
    num_samples = 1000
    _, sample_faces = sampler.area_weighted_random_sample_triangle_mesh(
        vertices, faces, num_samples)

    # All points should be approx poisson distributed among the 4 faces.
    self.assertEqual(sample_faces.shape[0], num_samples)
    num_faces = faces.shape[0]
    expected = np.array([num_samples / num_faces] * num_faces, dtype=np.intp)
    self.compare_poisson_equivalence(expected, tf.math.bincount(sample_faces))

  def test_area_sampler_jacobian_random(self):
    """Test the Jacobian of area weighted triangle random sampler."""
    tensor_vertex_size = np.random.randint(1, 3)
    tensor_out_shape = np.random.randint(1, 5, size=tensor_vertex_size)
    tensor_out_shape = tensor_out_shape.tolist()
    vertex_axis = np.array(((0., 0., 1), (1., 0., 0.), (0., 1., 0.),
                            (0., 0., -1.), (-1., 0., 0.), (0., -1., 0.)),
                           dtype=np.float32)
    vertex_axis = vertex_axis.reshape([1] * tensor_vertex_size + [6, 3])
    faces = np.array(((0, 1, 2), (0, 2, 4), (0, 4, 5), (0, 5, 1), (3, 2, 1),
                      (3, 4, 2), (3, 5, 4), (3, 1, 5)),
                     dtype=np.int32)
    faces = faces.reshape([1] * tensor_vertex_size + [8, 3])
    index_init = np.tile(faces, tensor_out_shape + [1, 1])
    vertex_scale = np.random.uniform(0.5, 5., tensor_out_shape + [1] * 2)
    vertex_init = vertex_axis * vertex_scale
    index_tensor = tf.convert_to_tensor(value=index_init)

    num_samples = np.random.randint(10, 100)

    def sampler_fn(vertices):
      sample_pts, _ = sampler.area_weighted_random_sample_triangle_mesh(
          vertices, index_tensor, num_samples, seed=[0, 1], stateless=True)
      return sample_pts

    self.assert_jacobian_is_correct_fn(
        sampler_fn, [vertex_init], atol=1e-4, delta=1e-4)


if __name__ == "__main__":
  test_case.main()
