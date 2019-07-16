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

from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.rendering.differentiable_renderer import rasterizer
from tensorflow_graphics.util import test_case


class RasterizerTest(test_case.TestCase):

  @parameterized.parameters(
      ((3, 2), (3, 2), (2, 2)),
      ((None, 3, 2), (None, 3, 2), (None, 1, 2)),
      ((10, 5, 3, 2), (10, 5, 3, 2), (10, 5, 2, 2)),
  )
  def test_rasterizer_barycentric_coordinates_exception_not_raised(
      self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(
        rasterizer.rasterizer_barycentric_coordinates, shapes)

  @parameterized.parameters(
      ("triangle_vertices_2d must have exactly 2 dimensions in axis -1", (3, 1),
       (3, 1), (1, 2)),
      ("triangle_vertices_2d must have exactly 3 dimensions in axis -2", (2, 2),
       (2, 2), (1, 2)),
      ("pixels must have exactly 2 dimensions in axis -1", (3, 2), (3, 2), (1, 3)),
      ("Not all batch dimensions are broadcast-compatible", (5, 3, 2), (5, 3, 2),
       (2, 10, 2)),
  )
  def test_rasterizer_barycentric_coordinates_exception_raised(
      self, error_msg, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(
        rasterizer.rasterizer_barycentric_coordinates, error_msg, shape)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_rasterizer_barycentric_coordinates_jacobian_random(self):
    """Tests the Jacobian of between_two_vectors_3d."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    triangle_vertices_init = np.random.random(tensor_shape + [3, 2])
    triangle_vertices_attributes_init = np.random.random(tensor_shape + [3, 2])
    pixels_init = np.random.random(tensor_shape + [10, 2])
    triangle_vertices = tf.convert_to_tensor(value=triangle_vertices_init)
    triangle_vertices_attributes = tf.convert_to_tensor(value=triangle_vertices_attributes_init)
    pixels = tf.convert_to_tensor(value=pixels_init)

    barycentric_coord, _a, _b = rasterizer.rasterizer_barycentric_coordinates(
        triangle_vertices, triangle_vertices_attributes, pixels)

    self.assert_jacobian_is_correct(
        triangle_vertices, triangle_vertices_init, barycentric_coord, atol=1e-4)
    self.assert_jacobian_is_correct(
        pixels, pixels_init, barycentric_coord, atol=1e-4)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_rasterizer_barycentric_coordinates_random(self):
    """Tests the Jacobian of between_two_vectors_3d."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    num_pixels = np.random.randint(1, 10)
    pixels_shape = tensor_shape + [num_pixels]

    triangle_vertices = np.random.random(tensor_shape + [3, 2])
    triangle_vertices_attributes = np.random.random(tensor_shape + [3, 2])
    pixels = np.random.random(pixels_shape + [2])
    barycentric_coordinates, _a, _b = rasterizer.rasterizer_barycentric_coordinates(
        triangle_vertices, triangle_vertices_attributes, pixels)
    barycentric_coordinates_sum = tf.reduce_sum(
        input_tensor=barycentric_coordinates, axis=-1)

    # Checks that resulting barycentric_coord are normalized.
    self.assertAllClose(
        barycentric_coordinates_sum, np.full(pixels_shape, 1.0), rtol=1e-4)

  @parameterized.parameters(
      ((3, 2), (), ()),
      ((None, 3, 2), (), ()),
      ((10, 1, 3, 2), (), ()),
  )
  def test_rasterizer_bounding_box_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(rasterizer.rasterizer_bounding_box,
                                        shapes)

  @parameterized.parameters(
      ("triangle_vertices_2d must have exactly 2 dimensions in axis -1", (3, 1),
       (), ()),
      ("triangle_vertices_2d must have exactly 3 dimensions in axis -2", (1, 2),
       (), ()),
      ("image_width must have a rank of 0", (3, 2), (1,), ()),
      ("image_height must have a rank of 0", (3, 2), (), (1,)),
  )
  def test_rasterizer_bounding_box_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(rasterizer.rasterizer_bounding_box,
                                    error_msg, shape)

#   @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
#   def test_rasterizer_bounding_box_jacobian_random(self):
#     """Tests the Jacobian of rasterizer_bounding_box."""
#     tensor_size = np.random.randint(3)
#     tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
#     image_width_init = np.random.randint(1, 256)
#     image_height_init = np.random.randint(1, 256)
#     triangle_vertices_init = np.random.random(tensor_shape + [3, 2])
#     triangle_vertices_init[..., 0] += 0.5*image_width_init
#     triangle_vertices_init[..., 1] += 0.5*image_height_init
#
#     triangle_vertices = tf.convert_to_tensor(
#         value=triangle_vertices_init, dtype=tf.float32)
#     image_width = tf.convert_to_tensor(
#         value=image_width_init, dtype=tf.float32)
#     image_height = tf.convert_to_tensor(
#         value=image_height_init, dtype=tf.float32)
#
#     bottom_right_corner, top_left_corner = rasterizer.rasterizer_bounding_box(
#         triangle_vertices, image_width, image_height)
#
#     self.assert_jacobian_is_correct(triangle_vertices, triangle_vertices_init,
#                                     bottom_right_corner, atol=1e-4)
#     self.assert_jacobian_is_correct(triangle_vertices, triangle_vertices_init,
#                                     top_left_corner, atol=1e-4)
#     self.assert_jacobian_is_correct(
#         image_width, image_width_init, bottom_right_corner, atol=1e-4)
#     self.assert_jacobian_is_correct(
#         image_width, image_width_init, top_left_corner, atol=1e-4)
#     self.assert_jacobian_is_correct(
#         image_height, image_height_init, bottom_right_corner, atol=1e-4)
#     self.assert_jacobian_is_correct(
#         image_height, image_height_init, top_left_corner, atol=1e-4)

  @parameterized.parameters(
      (((), (128, 128, 8), (10, 3), (10, 3), (7, 3)),
       (tf.int32, tf.float32, tf.float32, tf.float32, tf.int32)),)
  def test_rasterizer_rasterize_triangle_exception_not_raised(
      self, shapes, dtypes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(
        rasterizer.rasterizer_rasterize_triangle, shapes, dtypes)

  @parameterized.parameters(
      ("index must have a rank of 0", ((1,), (128, 128, 8), (10, 3), (10, 3), (7, 3)),
       (tf.int32, tf.float32, tf.float32, tf.float32, tf.int32)),
      ("result_tensor must have exactly 8 dimensions in axis -1",
       ((), (128, 128, 1), (10, 3), (10, 3), 
        (7, 3)), (tf.int32, tf.float32, tf.float32, tf.float32, tf.int32)),
      ("vertices must have exactly 3 dimensions in axis -1", ((), (128, 128, 8),
                                                              (10, 1), (10, 3), (7, 3)),
       (tf.int32, tf.float32, tf.float32, tf.float32, tf.int32)),
      ("triangles must have exactly 3 dimensions in axis -1",
       ((), (128, 128, 8), (10, 3), (10, 3)
        (7, 1)), (tf.int32, tf.float32, tf.float32, tf.float32, tf.int32)),
  )
  def test_rasterizer_rasterize_triangle_exception_raised(
      self, error_msg, shapes, dtypes):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(rasterizer.rasterizer_rasterize_triangle,
                                    error_msg, shapes, dtypes)

#   @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
#   def test_rasterizer_rasterize_triangle_jacobian_random(self):
#     """Tests the Jacobian of rasterize_triangle."""
#
#     num_vertices = np.random.randint(100)
#     num_triangles = np.random.randint(100)
#     index_init = np.random.randint(num_triangles)
#
#     vertices_init = np.random.random((num_vertices, 3))
#     triangles_init = np.random.randint(
#        0, num_vertices, size=(num_triangles, 3))
#     image_width_init = np.random.randint(1, 256)
#     image_height_init = np.random.randint(1, 256)
#     result_tensor_init = np.concatenate([
#         50.0 * np.ones((image_height_init, image_width_init, 1)),
#         -1.0 * np.ones((image_height_init, image_width_init, 1)),
#         -1.0 * np.ones((image_height_init, image_width_init, 3))
#     ], axis=-1)
#
#     index = tf.convert_to_tensor(value=index_init, dtype=tf.int32)
#     vertices = tf.convert_to_tensor(value=vertices_init, dtype=tf.float32)
#     triangles = tf.convert_to_tensor(value=triangles_init, dtype=tf.int32)
#     result_tensor = tf.convert_to_tensor(
#         value=result_tensor_init, dtype=tf.float32)
#
#     new_result_tensor = rasterizer.rasterizer_rasterize_triangle(
#         index, result_tensor, vertices, triangles)
#     x = np.random.randint(image_width_init)
#     y = np.random.randint(image_height_init)
#
#     self.assert_jacobian_is_correct(vertices, vertices_init,
#                                     new_result_tensor[y, x, :], atol=1e-4)

# @parameterized.parameters(
#     (((5, 3), (10, 3), (), (), ()),
#      (tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
#     (((1, 3), (1, 3), (), ()), (tf.float32, tf.int32, tf.int32, tf.int32)),
# )
# def test_rasterize_mesh_exception_not_raised(self, shapes, dtypes):
#   """Tests that the shape exceptions are not raised."""
#   self.assert_exception_is_not_raised(rasterizer._rasterize_mesh, shapes,
#                                       dtypes)

  @parameterized.parameters(
      ("vertices must have a rank of 2",
       ((1, 5, 3), (1, 5, 3), (10, 3), (), (), ()),
       (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("vertices must have exactly 3 dimensions in axis -1",
       ((5, 2), (5, 2), (10, 3), (), (), ()),
       (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("triangles must have a rank of 2",
       ((5, 3), (5, 3), (1, 10, 3), (), (), ()),
       (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("triangles must have exactly 3 dimensions in axis -1",
       ((5, 3), (5, 3), (10, 2), (), (), ()),
       (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("image_width must have a rank of 0",
       ((5, 3), (5, 3), (10, 3), (1,), (), ()),
       (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("image_height must have a rank of 0",
       ((5, 3), (5, 3), (10, 3), (), (1,), ()),
       (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("max_depth must have a rank of 0",
       ((5, 3), (5, 3), (10, 3), (), (), (1,)),
       (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
  )
  def test_rasterize_mesh_exception_raised(self, error_msg, shapes, dtypes):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(
        rasterizer._rasterize_mesh, error_msg, shapes, dtypes)

#   @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
#   def test_rasterize_mesh_jacobian_random(self):
#     """Tests the Jacobian of _rasterize_mesh."""
#
#     num_vertices = np.random.randint(100)
#     num_triangles = np.random.randint(20)
#     vertices_init = np.random.random((num_vertices, 3))
#     triangles_init = np.random.randint(0, num_vertices,
#                                        size=(num_triangles, 3))
#     image_width_init = np.random.randint(1, 256)
#     image_height_init = np.random.randint(1, 256)
#     max_depth_init = np.random.random()
#
#     vertices = tf.convert_to_tensor(value=vertices_init, dtype=tf.float32)
#     triangles = tf.convert_to_tensor(value=triangles_init, dtype=tf.int32)
#     image_width = tf.convert_to_tensor(value=image_width_init, dtype=tf.int32)
#     image_height = tf.convert_to_tensor(
#         value=image_height_init, dtype=tf.int32)
#     max_depth = tf.convert_to_tensor(value=max_depth_init, dtype=tf.float32)
#
#     result_tensor = rasterizer._rasterize_mesh(
#         vertices, triangles, image_width, image_height, max_depth)
#     x = np.random.randint(image_width_init)
#     y = np.random.randint(image_height_init)
#
#     self.assert_jacobian_is_correct(vertices, vertices_init,
#                                     result_tensor[y, x, 0], atol=1e-4)
#     self.assert_jacobian_is_correct(vertices, vertices_init,
#                                     result_tensor[y, x, 2:], atol=1e-4)
# @parameterized.parameters(
#     (((5, 3), (10, 3), (), (), ()),
#      (tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
#     (((1, 3), (1, 3), (), ()),
#      (tf.float32, tf.int32, tf.int32, tf.int32)),
# )
# def test_rasterize_exception_not_raised(self, shapes, dtypes):
#   """Tests that the shape exceptions are not raised."""
#   self.assert_exception_is_not_raised(rasterizer.rasterizer_rasterize,
#                                       shapes, dtypes)

  @parameterized.parameters(
      ("vertices must have a rank greater than 1",
       ((3,), (3,), (10, 3), (), (), ()),
       (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("vertices must have exactly 3 dimensions in axis -1",
       ((10, 5, 2), (10, 5, 2), (10, 10, 3), (), (), ()),
       (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("triangles must have a rank greater than 1",
       ((5, 3), (5, 3), (3,), (), (), ()),
       (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("triangles must have exactly 3 dimensions in axis -1",
       ((5, 3), (5, 3), (10, 2), (), (), ()),
       (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("Not all batch dimensions are identical",
       ((5, 10, 3), (5, 10, 3), (2, 8, 3), (), (), ()),
       (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("image_width must have a rank of 0",
       ((5, 3), (5, 3), (10, 3), (1,), (), ()),
       (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("image_height must have a rank of 0",
       ((5, 3), (5, 3), (10, 3), (), (1,), ()),
       (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
      ("max_depth must have a rank of 0",
       ((5, 3), (5, 3), (10, 3), (), (), (1,)),
       (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32)),
  )
  def test_rasterizer_rasterize_exception_raised(self,
                                                 error_msg,
                                                 shapes,
                                                 dtypes):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(
        rasterizer.rasterizer_rasterize, error_msg, shapes, dtypes)

#   @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
#   def test_rasterizer_rasterize_jacobian_random(self):
#     """Tests the Jacobian of rasterizer_rasterize."""
#     batch_size = 2
#     num_vertices = np.random.randint(100)
#     num_triangles = np.random.randint(5)
#     vertices_init = np.random.random((batch_size, num_vertices, 3))
#     triangles_init = np.random.randint(
#         0, num_vertices, size=(batch_size, num_triangles, 3))
#     image_width_init = np.random.randint(1, 256)
#     image_height_init = np.random.randint(1, 256)
#     max_depth_init = np.random.random()
#
#     vertices = tf.convert_to_tensor(value=vertices_init, dtype=tf.float32)
#     triangles = tf.convert_to_tensor(value=triangles_init, dtype=tf.int32)
#     image_width = tf.convert_to_tensor(value=image_width_init, dtype=tf.int32)
#     image_height = tf.convert_to_tensor(value=image_height_init,
#                                         dtype=tf.int32)
#     max_depth = tf.convert_to_tensor(value=max_depth_init, dtype=tf.float32)
#
#     depth_maps, _, barycentric_coordinates = rasterizer.rasterizer_rasterize(
#         vertices, triangles, image_width, image_height, max_depth)
#     b = np.random.randint(batch_size)
#     x = np.random.randint(image_width_init)
#     y = np.random.randint(image_height_init)
#
#     self.assert_jacobian_is_correct(vertices, vertices_init,
#                                     depth_maps[b, y, x], atol=1e-4)
#     self.assert_jacobian_is_correct(vertices, vertices_init,
#                                     barycentric_coordinates[b, y, x],
#                                     atol=1e-4)


if __name__ == "__main__":
  test_case.main()
