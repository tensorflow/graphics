#Copyright 2018 Google LLC
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
"""Tests for rasterizer functionalities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.notebooks.resources import tfg_simplified_logo
from tensorflow_graphics.rendering import rasterizer
from tensorflow_graphics.util import test_case


class RasterizerTest(test_case.TestCase):

  def setUp(self):
    """Sets the seed for tensorflow and numpy."""
    super(RasterizerTest, self).setUp()
    self.image_width = 25
    self.image_height = 25
    self.min_depth = 0.0
    self.max_depth = 100.0
    vertices = tfg_simplified_logo.mesh["vertices"].astype(np.float32)
    triangles = tfg_simplified_logo.mesh["faces"].astype(np.int32)
    self.vertices_init = vertices[:10, :]
    self.triangles_init = triangles[:6, :]

  @parameterized.parameters(
      ((3, 2), (2, 2)),
      ((None, 3, 2), (None, 1, 2)),
      ((10, 5, 3, 2), (10, 5, 2, 2)),
  )
  def test_get_barycentric_coordinates_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(rasterizer.get_barycentric_coordinates,
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
    self.assert_exception_is_raised(rasterizer.get_barycentric_coordinates,
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
    triangle_vertices = tf.convert_to_tensor(
        value=triangle_vertices_init, dtype=tf.float64)
    pixels = tf.convert_to_tensor(value=pixels_init, dtype=tf.float64)

    barycentric_coord, _ = rasterizer.get_barycentric_coordinates(
        triangle_vertices, pixels)

    with self.subTest(name="pixels_jacobian"):
      self.assert_jacobian_is_correct(pixels, pixels_init, barycentric_coord)

    with self.subTest(name="vertices_jacobian"):
      self.assert_jacobian_is_correct(triangle_vertices, triangle_vertices_init,
                                      barycentric_coord)

  def test_get_barycentric_coordinates_normalized(self):
    """Tests whether the barycentric coordinates are normalized."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    num_pixels = np.random.randint(1, 10)
    pixels_shape = tensor_shape + [num_pixels]
    triangle_vertices = np.random.random(tensor_shape + [3, 2])
    pixels = np.random.random(pixels_shape + [2])

    barycentric_coordinates, _ = rasterizer.get_barycentric_coordinates(
        triangle_vertices, pixels)
    barycentric_coordinates_sum = tf.reduce_sum(
        input_tensor=barycentric_coordinates, axis=-1)

    self.assertAllClose(barycentric_coordinates_sum, np.full(pixels_shape, 1.0))

  @parameterized.parameters(
      ((3, 2), (), ()),
      ((None, 3, 2), (), ()),
      ((10, 1, 3, 2), (), ()),
  )
  def test_get_bounding_box_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(rasterizer.get_bounding_box, shapes)

  @parameterized.parameters(
      ("triangle_vertices must have exactly 2 dimensions in axis -1", (3, 1),
       (), ()),
      ("triangle_vertices must have exactly 3 dimensions in axis -2", (1, 2),
       (), ()),
      ("image_width must have a rank of 0", (3, 2), (1,), ()),
      ("image_height must have a rank of 0", (3, 2), (), (1,)),
  )
  def test_get_bounding_box_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(rasterizer.get_bounding_box, error_msg,
                                    shape)

  @parameterized.parameters(
      (((), (128, 128, 5), (10, 3), (7, 3)),
       (tf.int32, tf.float32, tf.float32, tf.int32)),)
  def test_rasterize_triangle_exception_not_raised(self, shapes, dtypes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(rasterizer.rasterize_triangle, shapes,
                                        dtypes)

  @parameterized.parameters(
      ("index must have a rank of 0", ((1,), (128, 128, 5), (10, 3), (7, 3)),
       (tf.int32, tf.float32, tf.float32, tf.int32)),
      ("result_tensor must have exactly 5 dimensions in axis -1",
       ((), (128, 128, 1), (10, 3),
        (7, 3)), (tf.int32, tf.float32, tf.float32, tf.int32)),
      ("vertices must have exactly 3 dimensions in axis -1", ((), (128, 128, 5),
                                                              (10, 1), (7, 3)),
       (tf.int32, tf.float32, tf.float32, tf.int32)),
      ("triangles must have exactly 3 dimensions in axis -1",
       ((), (128, 128, 5), (10, 3),
        (7, 1)), (tf.int32, tf.float32, tf.float32, tf.int32)),
  )
  def test_rasterize_triangle_exception_raised(self, error_msg, shapes, dtypes):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(rasterizer.rasterize_triangle, error_msg,
                                    shapes, dtypes)

  def test_rasterize_triangle_jacobian_random(self):
    """Tests the Jacobian of rasterize_triangle."""
    index = 0
    depth_image = self.max_depth * np.ones(
        (self.image_height, self.image_width, 1))
    initial_values = -np.ones((self.image_height, self.image_width, 4))
    result_tensor_init = np.concatenate((depth_image, initial_values), axis=-1)
    vertices = tf.convert_to_tensor(value=self.vertices_init, dtype=tf.float32)
    triangles = tf.convert_to_tensor(value=self.triangles_init, dtype=tf.int32)
    result_tensor = tf.convert_to_tensor(
        value=result_tensor_init, dtype=tf.float32)
    x = self.image_width // 2
    y = self.image_height // 2

    new_result_tensor = rasterizer.rasterize_triangle(
        index, result_tensor, vertices, triangles, min_depth=self.min_depth)

    self.assert_jacobian_is_correct(vertices, self.vertices_init,
                                    new_result_tensor[y, x, :])

  @parameterized.parameters(
      (((5, 3), (10, 3), (), (), ()),
       (tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      (((1, 3), (1, 3), (), ()), (tf.float32, tf.int32, tf.int32, tf.int32)),
  )
  def test_rasterize_mesh_exception_not_raised(self, shapes, dtypes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(rasterizer.rasterize_mesh, shapes,
                                        dtypes)

  @parameterized.parameters(
      ("vertices must have a rank of 2", ((1, 5, 3), (10, 3), (), (), ()),
       (tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("vertices must have exactly 3 dimensions in axis -1", ((5, 2), (10, 3),
                                                              (), (), ()),
       (tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("triangles must have a rank of 2", ((5, 3), (1, 10, 3), (), (), ()),
       (tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("triangles must have exactly 3 dimensions in axis -1", ((5, 3), (10, 2),
                                                               (), (), ()),
       (tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("image_width must have a rank of 0", ((5, 3), (10, 3), (1,), (), ()),
       (tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("image_height must have a rank of 0", ((5, 3), (10, 3), (), (1,), ()),
       (tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("min_depth must have a rank of 0", ((5, 3), (10, 3), (), (), (1,)),
       (tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("max_depth must have a rank of 0", ((5, 3), (10, 3), (), (), (), (1,)),
       (tf.float32, tf.int32, tf.int32, tf.int32, tf.float32, tf.float32)),
  )
  def test_rasterize_mesh_exception_raised(self, error_msg, shapes, dtypes):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(rasterizer.rasterize_mesh, error_msg,
                                    shapes, dtypes)

  def test_rasterize_mesh_jacobian_random(self):
    """Tests the Jacobian of rasterizer_mesh."""
    vertices = tf.convert_to_tensor(value=self.vertices_init, dtype=tf.float32)
    triangles = tf.convert_to_tensor(value=self.triangles_init, dtype=tf.int32)
    x = self.image_width // 2
    y = self.image_height // 2

    result_tensor = rasterizer.rasterize_mesh(vertices, triangles,
                                              self.image_width,
                                              self.image_height, self.min_depth,
                                              self.max_depth)

    with self.subTest(name="depth_jacobian"):
      self.assert_jacobian_is_correct(vertices, self.vertices_init,
                                      result_tensor[y, x, 0])

    with self.subTest(name="barycentric_jacobian"):
      self.assert_jacobian_is_correct(vertices, self.vertices_init,
                                      result_tensor[y, x, 2:])

  @parameterized.parameters(
      (((5, 3), (10, 3), (), (), ()),
       (tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      (((1, 3), (1, 3), (), ()), (tf.float32, tf.int32, tf.int32, tf.int32)),
  )
  def test_rasterize_exception_not_raised(self, shapes, dtypes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(rasterizer.rasterize, shapes, dtypes)

  @parameterized.parameters(
      ("vertices must have a rank greater than 1", ((3,), (10, 3), (), (), ()),
       (tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("vertices must have exactly 3 dimensions in axis -1",
       ((10, 5, 2), (10, 10, 3), (), (),
        ()), (tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("triangles must have a rank greater than 1", ((5, 3), (3,), (), (), ()),
       (tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("triangles must have exactly 3 dimensions in axis -1", ((5, 3), (10, 2),
                                                               (), (), ()),
       (tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("Not all batch dimensions are identical", ((5, 10, 3), (2, 8, 3), (), (),
                                                  ()),
       (tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("image_width must have a rank of 0", ((5, 3), (10, 3), (1,), (), ()),
       (tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("image_height must have a rank of 0", ((5, 3), (10, 3), (), (1,), ()),
       (tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("min_depth must have a rank of 0", ((5, 3), (10, 3), (), (), (1,)),
       (tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("max_depth must have a rank of 0", ((5, 3), (10, 3), (), (), (), (1,)),
       (tf.float32, tf.int32, tf.int32, tf.int32, tf.float32, tf.float32)),
  )
  def test_rasterize_exception_raised(self, error_msg, shapes, dtypes):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(rasterizer.rasterize, error_msg, shapes,
                                    dtypes)

  def test_rasterize_jacobian_random(self):
    """Tests the Jacobian of rasterizer_rasterize."""
    batch_size = 2
    vertices_init = np.tile(self.vertices_init, (batch_size, 1, 1))
    triangles_init = np.tile(self.triangles_init, (batch_size, 1, 1))
    vertices = tf.convert_to_tensor(value=vertices_init, dtype=tf.float32)
    triangles = tf.convert_to_tensor(value=triangles_init, dtype=tf.int32)
    x = self.image_width // 2
    y = self.image_height // 2

    depth_maps, _, barycentric_coordinates = rasterizer.rasterize(
        vertices, triangles, self.image_width, self.image_height,
        self.min_depth, self.max_depth)

    with self.subTest(name="depth_jacobian"):
      self.assert_jacobian_is_correct(vertices, vertices_init,
                                      depth_maps[1, y, x] - depth_maps[0, y, x])

    with self.subTest(name="barycentric_jacobian"):
      self.assert_jacobian_is_correct(
          vertices, vertices_init,
          barycentric_coordinates[1, y, x] - barycentric_coordinates[0, y, x])


if __name__ == "__main__":
  test_case.main()
