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
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_graphics.geometry.representation import grid
from tensorflow_graphics.rendering.opengl import math as glm
from tensorflow_graphics.rendering.opengl import rasterization_backend
from tensorflow_graphics.rendering.tests import rasterization_test_utils
from tensorflow_graphics.util import test_case

_IMAGE_HEIGHT = 5
_IMAGE_WIDTH = 7
_TRIANGLE_SIZE = 2.0
_ENABLE_CULL_FACE = True
_NUM_LAYERS = 1


def _generate_vertices_and_view_matrices():
  model_to_eye_matrix_0 = rasterization_test_utils.make_look_at_matrix(
      look_at_point=(0.0, 0.0, 1.0))
  model_to_eye_matrix_1 = rasterization_test_utils.make_look_at_matrix(
      look_at_point=(0.0, 0.0, -1.0))
  world_to_camera = tf.stack((model_to_eye_matrix_0, model_to_eye_matrix_1))
  perspective_matrix = rasterization_test_utils.make_perspective_matrix(
      _IMAGE_WIDTH, _IMAGE_HEIGHT)
  # Shape [2, 4, 4]
  view_projection_matrix = tf.linalg.matmul(perspective_matrix, world_to_camera)
  depth = 1.0
  # Shape [2, 3, 3]
  vertices = (((-10.0 * _TRIANGLE_SIZE, 10.0 * _TRIANGLE_SIZE,
                depth), (10.0 * _TRIANGLE_SIZE, 10.0 * _TRIANGLE_SIZE, depth),
               (0.0, -10.0 * _TRIANGLE_SIZE, depth)),
              ((-_TRIANGLE_SIZE, 0.0, depth), (0.0, _TRIANGLE_SIZE, depth),
               (0.0, 0.0, depth)))
  return vertices, view_projection_matrix


def _proxy_rasterize(vertices, triangles, view_projection_matrices):
  return rasterization_backend.rasterize(vertices, triangles,
                                         view_projection_matrices,
                                         (_IMAGE_WIDTH, _IMAGE_HEIGHT),
                                         _ENABLE_CULL_FACE,
                                         _NUM_LAYERS).layer(0)


class RasterizationBackendTest(test_case.TestCase):

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (2, 32, 2), (17, 3),
       (2, 4, 4)),
      ("must have exactly 3 dimensions in axis -1", (2, 32, 3), (17, 2),
       (2, 4, 4)),
      ("must have a rank of 2", (2, 32, 3), (3, 17, 2), (2, 4, 4)),
      ("must have exactly 4 dimensions in axis -1", (2, 32, 3), (17, 3),
       (2, 4, 3)),
      ("must have exactly 4 dimensions in axis -2", (2, 32, 3), (17, 3),
       (2, 3, 4)),
      ("Not all batch dimensions are broadcast-compatible", (3, 32, 3), (17, 3),
       (5, 4, 4)),
      ("vertices must have a rank of 3, but it has rank 2", (32, 3), (17, 3),
       (4, 4)),
  )
  def test_rasterize_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(_proxy_rasterize, error_msg, shapes)

  @parameterized.parameters(
      (((1, 32, 3), (17, 3), (1, 4, 4)), (tf.float32, tf.int32, tf.float32)),
      (((None, 32, 3), (17, 3), (None, 4, 4)),
       (tf.float32, tf.int32, tf.float32)),
      (((2, 32, 3), (17, 3), (2, 4, 4)), (tf.float32, tf.int32, tf.float32)),
  )
  def test_rasterize_exception_not_raised(self, shapes, dtypes):
    self.assert_exception_is_not_raised(
        _proxy_rasterize, shapes=shapes, dtypes=dtypes)

  def test_rasterize_batch_vertices_only(self):
    triangles = np.array(((0, 1, 2),), np.int32)
    vertices, view_projection_matrix = _generate_vertices_and_view_matrices()
    # Use just first view projection matrix.
    view_projection_matrix = [
        view_projection_matrix[0], view_projection_matrix[0]
    ]
    predicted_fb = _proxy_rasterize(vertices, triangles, view_projection_matrix)
    mask = predicted_fb.foreground_mask
    self.assertAllEqual(mask[0, ...], tf.ones_like(mask[0, ...]))

    gt_layer_1 = np.zeros((_IMAGE_HEIGHT, _IMAGE_WIDTH, 1), np.float32)
    gt_layer_1[_IMAGE_HEIGHT // 2:, _IMAGE_WIDTH // 2:, 0] = 1.0
    self.assertAllEqual(mask[1, ...], gt_layer_1)

  def test_rasterize_batch_view_only(self):
    triangles = np.array(((0, 1, 2),), np.int32)
    vertices, view_projection_matrix = _generate_vertices_and_view_matrices()
    vertices = np.array([vertices[0], vertices[0]], dtype=np.float32)
    predicted_fb = _proxy_rasterize(vertices, triangles, view_projection_matrix)
    self.assertAllEqual(predicted_fb.foreground_mask[0, ...],
                        tf.ones_like(predicted_fb.foreground_mask[0, ...]))
    self.assertAllEqual(predicted_fb.foreground_mask[1, ...],
                        tf.zeros_like(predicted_fb.foreground_mask[1, ...]))

  def test_rasterize_preset(self):
    model_to_eye_matrix = rasterization_test_utils.make_look_at_matrix(
        look_at_point=(0.0, 0.0, 1.0))
    perspective_matrix = rasterization_test_utils.make_perspective_matrix(
        _IMAGE_WIDTH, _IMAGE_HEIGHT)
    view_projection_matrix = tf.linalg.matmul(perspective_matrix,
                                              model_to_eye_matrix)
    view_projection_matrix = tf.expand_dims(view_projection_matrix, axis=0)

    depth = 1.0
    vertices = np.array([[(-2.0 * _TRIANGLE_SIZE, 0.0, depth),
                          (0.0, _TRIANGLE_SIZE, depth), (0.0, 0.0, depth),
                          (0.0, -_TRIANGLE_SIZE, depth)]],
                        dtype=np.float32)
    triangles = np.array(((1, 2, 0), (0, 2, 3)), np.int32)

    predicted_fb = _proxy_rasterize(vertices, triangles, view_projection_matrix)

    with self.subTest(name="triangle_index"):
      groundtruth_triangle_index = np.zeros((1, _IMAGE_HEIGHT, _IMAGE_WIDTH, 1),
                                            dtype=np.int32)
      groundtruth_triangle_index[..., :_IMAGE_WIDTH // 2, 0] = 0
      groundtruth_triangle_index[..., :_IMAGE_HEIGHT // 2, _IMAGE_WIDTH // 2:,
                                 0] = 1
      self.assertAllEqual(groundtruth_triangle_index, predicted_fb.triangle_id)

    with self.subTest(name="mask"):
      groundtruth_mask = np.ones((1, _IMAGE_HEIGHT, _IMAGE_WIDTH, 1),
                                 dtype=np.int32)
      groundtruth_mask[..., :_IMAGE_WIDTH // 2, 0] = 0
      self.assertAllEqual(groundtruth_mask, predicted_fb.foreground_mask)

    attributes = np.array(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))).astype(np.float32)
    perspective_correct_interpolation = lambda geometry, pixels: glm.perspective_correct_interpolation(  # pylint: disable=g-long-lambda,line-too-long
        geometry, attributes, pixels, model_to_eye_matrix, perspective_matrix,
        np.array((_IMAGE_WIDTH, _IMAGE_HEIGHT)).astype(np.float32),
        np.array((0.0, 0.0)).astype(np.float32))
    with self.subTest(name="barycentric_coordinates_triangle_0"):
      geometry_0 = tf.gather(vertices, triangles[0, :], axis=1)
      pixels_0 = tf.transpose(
          a=grid.generate((3.5, 2.5), (6.5, 4.5), (4, 3)), perm=(1, 0, 2))
      barycentrics_gt_0 = perspective_correct_interpolation(
          geometry_0, pixels_0)
      self.assertAllClose(
          barycentrics_gt_0,
          predicted_fb.barycentrics.value[0, 2:, 3:, :],
          atol=1e-3)

    with self.subTest(name="barycentric_coordinates_triangle_1"):
      geometry_1 = tf.gather(vertices, triangles[1, :], axis=1)
      pixels_1 = tf.transpose(
          a=grid.generate((3.5, 0.5), (6.5, 1.5), (4, 2)), perm=(1, 0, 2))
      barycentrics_gt_1 = perspective_correct_interpolation(
          geometry_1, pixels_1)
      self.assertAllClose(
          barycentrics_gt_1,
          predicted_fb.barycentrics.value[0, 0:2, 3:, :],
          atol=1e-3)
