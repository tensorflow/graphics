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
"""Test cases for the edge regularizer."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_graphics.projects.mesh_rcnn.loss import edge_regularizer
from tensorflow_graphics.projects.mesh_rcnn.structures.mesh import Meshes
from tensorflow_graphics.util import test_case


class EdgeRegularizerTest(test_case.TestCase):
  """Test cases for the edge regularizer."""

  def test_edge_regularizer_preset(self):
    """Tests results of edge regularizer on predefined mesh."""
    vertices = tf.constant(
        [[0, 0, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]],
        dtype=tf.float32)
    faces = tf.constant([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]],
                        dtype=tf.int32)

    meshes = Meshes([vertices], [faces])
    adjacency = meshes.vertex_neighbors()
    sizes = meshes.get_sizes()[0]
    loss = edge_regularizer.evaluate(meshes.get_padded()[0], adjacency, sizes)
    expected_loss = ((4 * 4) + (8 * 2)) / 17.
    self.assertAllClose(expected_loss, loss)

  @parameterized.parameters(
      (4, 1),
      (2, 2),
      (1, 4)
  )
  def test_edge_regularizer_multi_batch(self, b1, b2):
    """Tests ifedge regularizer works on multiple batch dimensions with one
    empty mesh."""
    verts1 = tf.constant([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=tf.float32)
    faces1 = tf.constant([[0, 1, 2]], dtype=tf.int32)

    verts2 = tf.constant([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                         dtype=tf.float32)
    faces2 = tf.constant([[0, 1, 2], [0, 2, 3]], dtype=tf.int32)

    verts3 = tf.constant(
        [[0, 0, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]],
        dtype=tf.float32)
    faces3 = tf.constant([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]],
                         dtype=tf.int32)

    verts4, faces4 = tf.zeros((0, 3)), tf.zeros((0, 3), dtype=tf.int32)

    vertices = [verts1, verts2, verts3, verts4]
    faces = [faces1, faces2, faces3, faces4]

    meshes = Meshes(vertices, faces, batch_sizes=[b1, b2])

    adjacency = meshes.vertex_neighbors()
    sizes = meshes.get_sizes()[0]
    losses = edge_regularizer.evaluate(meshes.get_padded()[0], adjacency, sizes)

    self.assertEqual([b1, b2], losses.shape)
