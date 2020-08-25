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
"""Test Cases for the mesh losses of Mesh R-CNN."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_graphics.projects.mesh_rcnn.loss import mesh_loss
from tensorflow_graphics.projects.mesh_rcnn.structures.mesh import Meshes
from tensorflow_graphics.util import test_case


# ToDo: Move normal loss and edge regularizer tests to own files
class NormalDistanceTest(test_case.TestCase):
  """Test cases for the (absolute) normal distance."""

  def test_normal_distance_preset(self):
    """Test implementation of normal distance with predefined point sets."""

    input_points_a = tf.reshape(tf.range(2. * 3.), (2, 3))
    input_points_b = tf.reshape(tf.range(2. * 3.), (2, 3))

    input_normals_a = tf.constant([[1., 1., 1.], [1., 0., 1.]])
    input_normals_b = tf.constant([[-1, 0., 1.], [-1., -1., 0.]])

    input_a = tf.concat([input_points_a, input_normals_a], -1)
    input_b = tf.concat([input_points_b, input_normals_b], -1)

    loss = mesh_loss.normal_distance(input_a, input_b)

    abs_normals_a = tf.abs(input_normals_a)
    abs_normals_b = tf.abs(input_normals_b)

    dot_product = tf.einsum('ai,ai->a', abs_normals_a, abs_normals_b)
    expected_loss = -2 * tf.reduce_mean(dot_product)

    self.assertAlmostEqual(expected_loss, loss)

  @parameterized.parameters(
      ([], 20, 25),
      ([2], 10, 15),
      ([1, 5], 5, 5),
      ([3, 2, 3], 10, 10),
      ([5, 3, 2], 10, 5)
  )
  def test_normal_distance_random(self, batch_shape, n_points_a, n_points_b):
    """Tests if normal distance works on multiple batch dimensions with
    random values"""

    input_a = tf.random.normal(batch_shape + [n_points_a] + [6])
    input_b = tf.random.normal(batch_shape + [n_points_b] + [6])

    loss = mesh_loss.normal_distance(input_a, input_b)

    self.assertEqual(batch_shape, loss.shape)

  def test_extract_normals_of_nearest_neighbors(self):
    """Tests function `_extract_normals_of_nearest_neighbors`."""
    input_points_a = tf.reshape(tf.range(2. * 3.), (2, 3))
    input_points_b = tf.reshape(tf.range(2. * 3.), (2, 3))

    input_normals_a = tf.constant([[1., 1., 1.], [2., 2., 2.]])
    input_normals_b = tf.constant([[3, 3., 3.], [4., 4., 4.]])

    inputs_a = tf.concat([input_points_a, input_normals_a], axis=-1)
    inputs_b = tf.concat([input_points_b, input_normals_b], axis=-1)

    normals_a2b, normals_b2a = mesh_loss._extract_normals_of_nearest_neighbors(
        inputs_a, inputs_b)

    self.assertAllEqual(input_normals_b, normals_a2b)
    self.assertAllEqual(input_normals_a, normals_b2a)

  @parameterized.parameters(
      ([], 20),
      ([2], 30),
      ([1, 5], 5),
      ([3, 2, 3], 10),
      ([5, 3, 2], 10)
  )
  def test_extract_normals_of_nearest_neighbors_multi_batch(self,
                                                            batch_shape,
                                                            n_points):
    """Tests function `_extract_normals_of_nearest_neighbors` on input
    tensors with multiple batch dimensions, but same number of points."""
    shape = batch_shape + [n_points, 3]
    input_points_a = tf.reshape(tf.range(tf.reduce_prod(shape)), shape)
    input_points_b = tf.reshape(tf.range(tf.reduce_prod(shape)), shape)

    input_normals_a = tf.reshape(tf.range(tf.reduce_prod(shape)), shape)
    input_normals_b = tf.reshape(tf.range(tf.reduce_prod(shape),
                                          2 * tf.reduce_prod(shape)), shape)

    inputs_a = tf.concat([input_points_a, input_normals_a], axis=-1)
    inputs_b = tf.concat([input_points_b, input_normals_b], axis=-1)

    normals_a2b, normals_b2a = mesh_loss._extract_normals_of_nearest_neighbors(
        inputs_a, inputs_b)

    self.assertAllEqual(input_normals_b, normals_a2b)
    self.assertAllEqual(input_normals_a, normals_b2a)


class EdgeRegularizerTest(test_case.TestCase):
  """Test cases for the edge regularizer."""

  def test_edge_regularizer_preset(self):
    """Tests results of edge regularizer on predefined mesh."""
    vertices = tf.constant(
        [[0, 0, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]],
        dtype=tf.float32)
    faces = tf.constant([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]],
                        dtype=tf.int32)

    meshes = Meshes([vertices, vertices], [faces, faces])
    adjacency = meshes.vertex_neighbors()
    sizes = meshes.get_sizes()[0]
    loss = mesh_loss.edge_regularizer(meshes.get_padded()[0], adjacency, sizes)
    expected_loss = ((4 * 4) + (8 * 2)) / 17.
    self.assertAllClose([expected_loss, expected_loss], loss)

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
    losses = mesh_loss.edge_regularizer(meshes.get_padded()[0], adjacency, sizes)

    self.assertEqual([b1, b2], losses.shape)


class MeshRcnnTest(test_case.TestCase):
  """Test cases for the edge regularizer."""


if __name__ == '__main__':
  test_case.main()
