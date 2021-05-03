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
"""Tests for differentiable barycentrics."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_graphics.rendering import barycentrics
from tensorflow_graphics.rendering import rasterization_backend
from tensorflow_graphics.rendering import utils
from tensorflow_graphics.util import test_case


class BarycentricsTest(test_case.TestCase):

  @parameterized.parameters(
      (rasterization_backend.RasterizationBackends.CPU,))
  def test_differentialbe_barycentrics_close_to_rasterizer(
      self, rasterization_backend_type):
    image_width = 32
    image_height = 32

    initial_vertices = tf.constant([[[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0]]],
                                   dtype=tf.float32)
    triangles = [[0, 1, 2]]
    view_projection_matrix = tf.expand_dims(tf.eye(4), axis=0)

    # Compute rasterization barycentrics
    rasterized = rasterization_backend.rasterize(
        initial_vertices,
        triangles,
        view_projection_matrix, (image_width, image_height),
        num_layers=1,
        backend=rasterization_backend_type)

    ras_barycentrics = rasterized.barycentrics.value

    clip_space_vertices = utils.transform_homogeneous(view_projection_matrix,
                                                      initial_vertices)
    rasterized_w_diff_barycentrics = barycentrics.differentiable_barycentrics(
        rasterized, clip_space_vertices, triangles)
    diff_barycentrics = rasterized_w_diff_barycentrics.barycentrics.value

    self.assertAllClose(ras_barycentrics, diff_barycentrics, rtol=1e-4)
