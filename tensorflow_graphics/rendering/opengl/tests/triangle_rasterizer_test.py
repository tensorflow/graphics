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
# Lint as: python3
"""Tests for google3.third_party.py.tensorflow_graphics.rendering.opengl.tests.triangle_rasterizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_graphics.rendering.opengl import triangle_rasterizer
from tensorflow_graphics.util import test_case


def _proxy_rasterizer(background_vertices, background_attributes,
                      background_triangles, camera_origin, look_at, camera_up,
                      field_of_view, near_plane, far_plane, bottom_left):
  return triangle_rasterizer.TriangleRasterizer(
      background_vertices, background_attributes, background_triangles,
      camera_origin, look_at, camera_up, field_of_view, (4, 3), near_plane,
      far_plane, bottom_left)


class TriangleRasterizerTest(test_case.TestCase):

  def setUp(self):
    super(TriangleRasterizerTest, self).setUp()
    camera_origin = (0.0, 0.0, 0.0)
    camera_up = (0.0, 1.0, 0.0)
    look_at = (0.0, 0.0, 1.0)
    field_of_view = (60 * np.math.pi / 180,)
    height = 3.0
    width = 5.0
    near_plane = 0.01
    far_plane = 400.0
    image_size = (height, width)
    self.image_size_int = (int(height), int(width))
    bottom_left = (0.0, 0.0)
    self.triangle_size = 1000
    background_geometry = np.array(
        ((-self.triangle_size, self.triangle_size, far_plane - 10.0),
         (self.triangle_size, self.triangle_size, far_plane - 10.0),
         (0.0, -self.triangle_size, far_plane - 10.0)),
        dtype=np.float32)
    background_attribute = np.array(
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), dtype=np.float32)
    background_triangle = np.array((0, 1, 2), np.int32)
    self.rasterizer = triangle_rasterizer.TriangleRasterizer(
        background_geometry, background_attribute, background_triangle,
        camera_origin, look_at, camera_up, field_of_view, image_size,
        (near_plane,), (far_plane,), bottom_left)

  @parameterized.parameters(
      (((1, 3), (1, 7), (3,), (3,), (3,), (3,), (1,), (1,), (1,), (2,)),
       (tf.float32, tf.float32, tf.int32, tf.float32, tf.float32, tf.float32,
        tf.float32, tf.float32, tf.float32, tf.float32)),
      (((5, 3), (5, 7), (3,), (3,), (3,), (3,), (1,), (1,), (1,),
        (2,)), (tf.float32, tf.float32, tf.int32, tf.float32, tf.float32,
                tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)),
  )
  def test_rasterizer_init_exception_not_raised(self, shapes, dtypes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(
        _proxy_rasterizer, shapes=shapes, dtypes=dtypes)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (3, 2), (3, 7), (3,), (3,),
       (3,), (3,), (1,), (1,), (1,), (2,)),
      ("must have a rank of 2", (1, 3, 3), (3, 7), (3,), (3,), (3,), (3,), (1,),
       (1,), (1,), (2,)), ("must have a rank of 2", (3, 3), (1, 3, 7), (3,),
                           (3,), (3,), (3,), (1,), (1,), (1,), (2,)),
      ("must have exactly 3 dimensions in axis -1", (3, 3), (3, 7), (2,), (3,),
       (3,), (3,), (1,), (1,), (1,), (2,)),
      ("Not all batch dimensions are identical", (1, 3), (3, 7), (3,), (3,),
       (3,), (3,), (1,), (1,), (1,), (2,)))
  def test_rasterizer_init_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(_proxy_rasterizer, error_msg, shapes)

  @parameterized.parameters(
      (((1, 3, 3), (1, 3, 3), (3,)), (tf.float32, tf.float32, tf.int32)),
      (((8, 5, 7, 3), (8, 5, 7, 3), (3,)), (tf.float32, tf.float32, tf.int32)),
  )
  def test_rasterizer_rasterize_exception_not_raised(self, shapes, dtypes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(
        self.rasterizer.rasterize, shapes=shapes, dtypes=dtypes)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (3, 2), (3, 7), (3,)),
      ("must have exactly 3 dimensions in axis -1", (4, 3), (4, 7), (2,)),
      ("Not all batch dimensions are identical", (3, 2, 3), (3, 1, 7), (3,)))
  def test_rasterizer_rasterize_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(self.rasterizer.rasterize, error_msg,
                                    shapes)

  @parameterized.parameters(((2, 1, 3),), ((1,),))
  def test_rasterizer_rasterize_preset(self, batch_shape):
    """Tests that the rasterizer yields expected results.

    Args:
      batch_shape: shape of the batch of geometry and attributes to render.  In
        more details, this test generates several scenes, each composed of a
        single triangle parallel to the image plane with 3 channel vertex
        attributes that are a function of the depth of the triangle. A
        batched-call to the rasterization OP is performed and the output is
        checked against ground-truth.
    """
    start_depth = 20
    depth_increment = 20
    num_batch_elements = 1
    triangles = np.array((0, 1, 2), np.int32)
    for batch_dimension in batch_shape:
      num_batch_elements = num_batch_elements * batch_dimension
    geometry = np.zeros(shape=(num_batch_elements, 3, 3), dtype=np.float32)
    attributes = np.zeros(shape=(num_batch_elements, 3, 3), dtype=np.float32)
    groundtruth = np.zeros(
        shape=(num_batch_elements,) + self.image_size_int + (3,),
        dtype=np.float32)
    for i in range(num_batch_elements):
      current_depth = i * depth_increment + start_depth
      geometry[i,
               ...] = ((-self.triangle_size, self.triangle_size, current_depth),
                       (self.triangle_size, self.triangle_size, current_depth),
                       (0.0, -self.triangle_size, current_depth))
      attributes[i, ...] = ((current_depth, current_depth + 1.0,
                             current_depth + 2.0),
                            (current_depth, current_depth + 1.0,
                             current_depth + 2.0), (current_depth,
                                                    current_depth + 1.0,
                                                    current_depth + 2.0))
      groundtruth[i, ...] = (current_depth, current_depth + 1.0,
                             current_depth + 2.0)
    geometry = np.reshape(geometry, batch_shape + (1, 3, 3))
    attributes = np.reshape(attributes, batch_shape + (1, 3, 3))
    groundtruth = np.reshape(groundtruth,
                             batch_shape + self.image_size_int + (3,))

    prediction = self.rasterizer.rasterize(geometry, attributes, triangles)

    self.assertAllClose(prediction, groundtruth)


if __name__ == "__main__":
  test_case.main()
