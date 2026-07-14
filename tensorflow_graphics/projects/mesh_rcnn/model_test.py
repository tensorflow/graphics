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
"""Test cases for the Mesh R-CNN implementation."""

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_graphics.projects.mesh_rcnn.structures.mesh import Meshes
from tensorflow_graphics.projects.mesh_rcnn.model import MeshRCNN
from tensorflow_graphics.projects.mesh_rcnn.configs.config import MeshRCNNConfig
from tensorflow_graphics.util import test_case


class MeshRCNNTest(test_case.TestCase):
  """Unit tests for the Mesh R-CNN implementation."""

  def test_forward_pass_model_integration(self):
    """Tests a forward pass of Mesh R-CNN with random input data to verify if
    connection of branches works correctly."""
    config = MeshRCNNConfig()
    model = MeshRCNN(config)

    input_features = tf.random.normal((2, 14, 14, 128), dtype=tf.float32)
    input_intrinsics = tf.random.normal((2, 3, 3), dtype=tf.float32)
    inputs = [input_features, input_intrinsics]

    out_voxels, out_meshes = model(inputs, training=False)

    self.assertEqual([2, 28, 28, 28], out_voxels.shape)
    self.assertEqual(2, out_meshes.get_padded()[0].shape[0])

  def test_forward_pass_with_gradients(self):
    """Tests a train step of Mesh R-CNN with random input data."""
    config = MeshRCNNConfig()
    model = MeshRCNN(config)

    input_features = tf.random.normal((2, 14, 14, 128), dtype=tf.float32)
    input_intrinsics = tf.random.normal((2, 3, 3), dtype=tf.float32)
    inputs = [input_features, input_intrinsics]

    voxels = tf.ones((2, 28, 28, 28))

    verts1 = tf.constant(
        [[0, 0, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]],
        dtype=tf.float32)
    faces1 = tf.constant([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]],
                         dtype=tf.int32)

    verts2, faces2 = tf.zeros((0, 3)), tf.zeros((0, 3), dtype=tf.int32)

    meshes = Meshes([verts1, verts2], [faces1, faces2])
    ground_truths = [voxels, meshes]

    model.compile(loss_weights=config.loss_weights)

    losses = model.train_step([inputs, ground_truths])
    self.assertTrue('voxel_loss' in losses)
    self.assertTrue('mesh_loss' in losses)
    self.assertEqual([], losses['voxel_loss'].shape)
    self.assertEqual([], losses['mesh_loss'].shape)
    self.assertNotEqual(losses['voxel_loss'], losses['mesh_loss'])

  @parameterized.parameters(
      ([2, 8, 8, 1], [4, 3, 3], [2, 28, 28, 28], [2, 4, 3],
       'Not all batch dimensions are identical'),
      ([3, 6, 6], [3, 3], [2, 28, 28, 28], [2, 4, 3],
       'intrinsics must have a rank of 3'),
      ([2, 6, 6, 6], [2, 3, 4], [2, 28, 28, 28], [2, 4, 3],
        'intrinsics must have exactly 3 dimensions in axis -1'),
       ([2, 6, 6, 6], [2, 4, 3], [2, 28, 28, 28], [2, 4, 3],
         'intrinsics must have exactly 3 dimensions in axis -2'),
      ([2, 6, 6, 6], [2, 3, 3], [2, 28, 28, 8], [2, 4, 3],
       'ground_truth_voxels must have exactly 28 dimensions in axis -1'),
      ([2, 6, 6, 6], [2, 3, 3], [28, 28, 28], [2, 4, 3],
       'ground_truth_voxels must have a rank of 4'),
      ([2, 6, 6, 6], [2, 3, 3], [2, 28, 28, 28], None,
       'Ground truth mesh must be provided as an instance'),
      ([2, 6, 6, 6], [2, 3, 3], [2, 28, 28, 28], [4, 2, 3],
       'Not all batch dimensions are identical'),
      (None, [2, 3, 3], [2, 28, 28, 28], [2, 4, 3],
       '`inputs` must be a list or tuple of two tensors.')
  )
  def test_raising_inputs(self,
                          features_shape,
                          intrinsics_shape,
                          gt_voxel_shape,
                          gt_meshes_shape,
                          msg):
    """Tests the model forward pass with invalid input data."""

    if not features_shape is None:
      features = tf.random.normal(features_shape)
    else:
      features = None
    intrinsics = tf.random.normal(intrinsics_shape)
    gt_voxels = tf.random.normal(gt_voxel_shape)

    if not gt_meshes_shape is None:
      verts = []
      faces = []
      for _ in range(gt_meshes_shape[0]):
        verts.append(tf.random.normal(gt_meshes_shape[1:]))
        faces.append(tf.ones(gt_meshes_shape[1:], dtype=tf.int32))

      meshes = Meshes(verts, faces)
    else:
      meshes = tf.constant([0])

    if not features is None:
      data = [[features, intrinsics], [gt_voxels, meshes]]
    else:
      data = [intrinsics, [gt_voxels, meshes]]

    config = MeshRCNNConfig()
    model = MeshRCNN(config)

    with self.assertRaisesWithPredicateMatch(ValueError, msg):
      model.test_step(data)


if __name__ == "__main__":
  test_case.main()