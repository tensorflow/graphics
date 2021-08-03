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
"""Tests for tensorflow_graphics.rendering.tests.interpolate."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from six.moves import range
import tensorflow as tf

from tensorflow_graphics.rendering import interpolate
from tensorflow_graphics.rendering import rasterization_backend
from tensorflow_graphics.rendering.tests import rasterization_test_utils
from tensorflow_graphics.util import test_case


class RasterizeTest(test_case.TestCase):

  def setUp(self):
    super(RasterizeTest, self).setUp()

    self.cube_vertex_positions = tf.constant(
        [[[-1, -1, 1], [-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [1, -1, 1],
          [1, -1, -1], [1, 1, -1], [1, 1, 1]]],
        dtype=tf.float32)
    self.cube_triangles = tf.constant(
        [[0, 1, 2], [2, 3, 0], [3, 2, 6], [6, 7, 3], [7, 6, 5], [5, 4, 7],
         [4, 5, 1], [1, 0, 4], [5, 6, 2], [2, 1, 5], [7, 4, 0], [0, 3, 7]],
        dtype=tf.int32)

    self.image_width = 640
    self.image_height = 480
    perspective = rasterization_test_utils.make_perspective_matrix(
        self.image_width, self.image_height)
    projection = tf.matmul(
        perspective,
        rasterization_test_utils.make_look_at_matrix(
            camera_origin=(2.0, 3.0, 6.0)))
    # Add batch dimension.
    self.projection = tf.expand_dims(projection, axis=0)

  @parameterized.parameters(True, False)
  def test_renders_colored_cube(self, enable_cull_face):
    """Renders a simple colored cube."""
    num_layers = 1
    rasterized = rasterization_backend.rasterize(
        self.cube_vertex_positions,
        self.cube_triangles,
        self.projection, (self.image_width, self.image_height),
        num_layers=num_layers,
        enable_cull_face=enable_cull_face,
        backend=rasterization_backend.RasterizationBackends.CPU).layer(0)

    vertex_rgb = (self.cube_vertex_positions * 0.5 + 0.5)
    vertex_rgba = tf.concat([vertex_rgb, tf.ones([1, 8, 1])], axis=-1)
    rendered = interpolate.interpolate_vertex_attribute(vertex_rgba,
                                                        rasterized).value

    face_culling_num = 1 if enable_cull_face else 0
    baseline_image = rasterization_test_utils.load_baseline_image(
        'Unlit_Cube_0_%d.png' % face_culling_num, rendered.shape)

    images_near, error_message = rasterization_test_utils.compare_images(
        self, baseline_image, rendered)
    self.assertTrue(images_near, msg=error_message)

  @parameterized.parameters([2, 3])
  def test_renders_colored_cube_multilayer(self, num_layers):
    """Renders a simple colored cube with multiple layers."""
    rasterized = rasterization_backend.rasterize(
        self.cube_vertex_positions,
        self.cube_triangles,
        self.projection, (self.image_width, self.image_height),
        num_layers=num_layers,
        enable_cull_face=False,
        backend=rasterization_backend.RasterizationBackends.CPU)

    vertex_rgb = (self.cube_vertex_positions * 0.5 + 0.5)
    vertex_rgba = tf.concat([vertex_rgb, tf.ones([1, 8, 1])], axis=-1)
    rendered = interpolate.interpolate_vertex_attribute(vertex_rgba,
                                                        rasterized).value

    image_shape = [1] + rendered.shape[2:]
    for layer_index in range(num_layers):
      baseline_image = rasterization_test_utils.load_baseline_image(
          'Unlit_Cube_{}_{}.png'.format(0, layer_index), image_shape)
      images_near, error_message = rasterization_test_utils.compare_images(
          self, baseline_image, rendered[:, layer_index, ...])
      self.assertTrue(images_near, msg=error_message)


if __name__ == '__main__':
  tf.test.main()
