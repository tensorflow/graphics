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
"""Implementation of the mesh losses of Mesh R-CNN."""

import tensorflow as tf

from tensorflow_graphics.geometry.representation.mesh import sampler
from tensorflow_graphics.nn.loss import chamfer_distance


def weighted_mean_mesh_rcnn_loss(weights=None,
                                 gt_sample_size=5000,
                                 pred_sample_size=5000):
  """
  Compute the mesh prediction loss defined in the Mesh R-CNN paper.

  Args:
    weights: dictionary containing the weights for the different losses, e.g.
      weights = {'chamfer': 1.0, 'normals': 0.0, 'edge': 0.2}
    gt_sample_size: int, denoting the number of points to sample from ground
      truth meshes.
    pred_sample_size: int, denoting the number of points to sample from
      predicted meshes.

  Returns:
    A function with signature (y_true, y_pred) that can be passes to Keras'
    model.compile function.
  """

  w_chamfer = weights['chamfer']
  #w_normal = weights['normal']
  #w_edge = weights['edge']

  def mesh_rcnn_loss(y_true, y_pred):
    """
    Closure that can be passed to Keras' model.compile function.

    Args:
      y_true: Meshes object containing the ground truth meshes
      y_pred: Meshes object with predictioncs,
        storing the same number of meshes as y_true

    Returns:
      float32 scalar tensor containing the weighted Mesh R-CNN loss.

    """
    gt_vertices, gt_faces = y_true.get_padded()
    gt_face_weights = tf.ones((gt_faces.shape[:-1]))
    pred_vertices, pred_faces = y_pred.get_padded()
    pred_face_weights = tf.ones((pred_faces.shape[:-1]))
    gt_points, _ = sampler.weighted_random_sample_triangle_mesh(
        gt_vertices,
        gt_faces,
        gt_sample_size,
        gt_face_weights
    )
    pred_points, _ = sampler.weighted_random_sample_triangle_mesh(
        pred_vertices,
        pred_faces,
        pred_sample_size,
        pred_face_weights
    )

    l_chamfer = chamfer_distance.evaluate(gt_points, pred_points)

    return w_chamfer * l_chamfer

  return mesh_rcnn_loss
