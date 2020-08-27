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

from tensorflow_graphics.geometry.representation.mesh import normals
from tensorflow_graphics.geometry.representation.mesh import sampler
from tensorflow_graphics.nn.loss import chamfer_distance
from tensorflow_graphics.projects.mesh_rcnn.loss import edge_regularizer
from tensorflow_graphics.projects.mesh_rcnn.loss import normal_distance


def initialize(weights=None,
               gt_sample_size=5000,
               pred_sample_size=5000):
  """Initialize the mesh prediction loss as defined in the Mesh R-CNN paper.

  This functions builds a closure that evaluates the loss function
  on ground truth and predicted `Meshes`, that can be passed to a Keras Model in
  in its compile function.

  Example:
    ```python
    weights = {'chamfer': 1., 'normal': 0.5, 'edge': 2.}
    model = tf.keras.Model(inputs=[], outputs=[])
    model.compile(loss=mesh_rcnn_loss.initialize(weigths, 10000, 10000),
                  optimizer='adam',
                  metrics='mae')

    # This calls the enclosed loss.evaluate(y_true, y_pred) during training.
    ```

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
  w_normal = weights['normal']
  w_edge = weights['edge']

  weigths = tf.constant([w_chamfer, w_normal, w_edge])

  def evaluate(y_true, y_pred):
    """Closure that can be passed to Keras' model.compile function.

    Args:
      y_true: Meshes object containing the ground truth meshes
      y_pred: Meshes object with predictioncs,
        storing the same number of meshes as y_true

    Returns:
      float32 scalar tensor containing the weighted Mesh R-CNN loss.
    """
    gt_vertices, gt_faces = y_true.get_flattened()
    pred_vertices, pred_faces = y_pred.get_flattened()

    dim = gt_vertices.shape[-1]

    gt_points_with_normals = _sample_points_and_normals(gt_vertices,
                                                        gt_faces,
                                                        gt_sample_size)

    pred_points_with_normals = _sample_points_and_normals(pred_vertices,
                                                          pred_faces,
                                                          pred_sample_size)

    l_chamfer = chamfer_distance.evaluate(gt_points_with_normals[..., :dim],
                                          pred_points_with_normals[..., :dim])
    l_normal = normal_distance.evaluate(gt_points_with_normals,
                                        pred_points_with_normals)
    l_edge = edge_regularizer.evaluate(y_pred.get_padded()[0],
                                       y_pred.vertex_neighbors(),
                                       y_pred.get_sizes()[0])

    losses = tf.stack([l_chamfer, l_normal, l_edge], -1)

    return tf.reduce_mean(weigths * losses, -1)

  return evaluate


def _sample_points_and_normals(vertices, faces, sample_size):
  """
  Helper function to jointly sample points from a mesh together with associated
  face normal vectors.
  Args:
    vertices: A float32 tensor of shape `[N, D]` representing the
      mesh vertices. D denotes the dimensionality of the input space.
    faces: A float32 tensor of shape `[M, V]` representing the
      mesh vertices. V denotes the number of vertices per face.
    sample_size: An int scalar denoting the number of points to be sampled from
      the mesh surface.

  Returns:
    A float32 tensor of shape `[A1, ..., An, N, 2D]` containing
    the sampled points and normals of the provided mesh. On the last axis, the
    first D entries correspond to point locations and the last D
    entries correspond to normal vectors of the corresponding mesh faces
    in a D dimensional space.

  """
  points, face_idx = sampler.area_weighted_random_sample_triangle_mesh(
      vertices,
      faces,
      sample_size
  )
  face_positions = normals.gather_faces(vertices, faces)
  # Setting clockwise to false, since cubify outputs them in CCW order.
  face_normals = normals.face_normals(face_positions, clockwise=False)
  sampled_point_normals = tf.gather(face_normals, face_idx, axis=None)
  points_with_normals = tf.concat([points, sampled_point_normals], -1)

  return points_with_normals
