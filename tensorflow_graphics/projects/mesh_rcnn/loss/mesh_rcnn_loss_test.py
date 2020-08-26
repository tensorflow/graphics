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
# pylint: disable=protected-access
"""Test Cases for the mesh losses of Mesh R-CNN."""

import math

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
from tensorflow_graphics.nn.loss import chamfer_distance
from tensorflow_graphics.projects.mesh_rcnn.loss import edge_regularizer
from tensorflow_graphics.projects.mesh_rcnn.loss import mesh_rcnn_loss
from tensorflow_graphics.projects.mesh_rcnn.loss import normal_distance
from tensorflow_graphics.projects.mesh_rcnn.structures.mesh import Meshes
from tensorflow_graphics.util import test_case


class MeshRcnnTest(test_case.TestCase):
  """Test cases for the edge regularizer."""

  @parameterized.parameters(
      ({'chamfer': 1., 'normal': 0., 'edge': 0.}, 2000, 2000),
      ({'chamfer': 0., 'normal': 1., 'edge': 0.}, 2000, 2000),
      ({'chamfer': 0., 'normal': 0., 'edge': 1.}, 2000, 2000),
      ({'chamfer': 0., 'normal': 0., 'edge': 0.}, 100, 100),
      ({'chamfer': 1., 'normal': 1., 'edge': 1.}, 1000, 2000),
      ({'chamfer': 1., 'normal': 1., 'edge': 1.}, 2000, 1000)
  )
  def test_mesh_r_cnn_loss_integration(self, weights, gt_points, pred_points):
    """Integration Test for Mesh R-CNN Loss with two meshes and different
    weights and sample sizes."""

    tf.random.set_seed(42)

    vertices1 = [tf.constant([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                              [-1., 0., 0.], [0., -1., 0.]], tf.float32)]
    faces1 = [tf.constant([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]],
                          tf.int32)]

    # second mesh is first mesh rotated by 45 degrees around x axis:
    rot_mat = rotation_matrix_3d.from_axis_angle([1., 0., 0.],
                                                 [45. * math.pi / 180.])
    vertices2 = [rotation_matrix_3d.rotate(vertices1[0], rot_mat)]
    faces2 = [tf.constant([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]],
                          tf.int32)]

    gt_mesh = Meshes(vertices2, faces2)
    pred_mesh = Meshes(vertices1, faces1)

    loss_fn = mesh_rcnn_loss.initialize(weights,
                                        gt_sample_size=gt_points,
                                        pred_sample_size=pred_points)

    gt_points_with_normals = mesh_rcnn_loss._sample_points_and_normals(
        vertices2[0],
        faces2[0],
        gt_points)

    pred_points_with_normals = mesh_rcnn_loss._sample_points_and_normals(
        vertices1[0],
        faces1[0],
        pred_points)

    expected_chamfer = chamfer_distance.evaluate(
        gt_points_with_normals[:, :3],
        pred_points_with_normals[:, :3])

    expeceted_normal = normal_distance.evaluate(gt_points_with_normals,
                                                pred_points_with_normals)

    pred_neighbors = pred_mesh.vertex_neighbors()
    pred_sizes = pred_mesh.get_sizes()[0]
    expected_edge_regularizer = edge_regularizer.evaluate(
        pred_mesh.get_padded()[0],
        pred_neighbors,
        pred_sizes)

    expected_loss = (weights['chamfer'] * expected_chamfer +
                     weights['normal'] * expeceted_normal +
                     weights['edge'] * expected_edge_regularizer) / 3.

    self.assertAllClose(expected_loss, loss_fn(gt_mesh, pred_mesh), rtol=10e-3)


if __name__ == '__main__':
  test_case.main()
