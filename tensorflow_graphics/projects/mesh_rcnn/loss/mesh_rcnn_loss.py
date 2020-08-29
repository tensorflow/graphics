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
               pred_sample_size=5000,
               sampling_seed=None,
               stateless_sampling=False):
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
    sampling_seed: Optional seed for the sampling op.
    stateless_sampling: Optional flag to use stateless random sampler.
      If stateless_sampling=True, then seed must be provided as shape `[2]` int
      tensor. Stateless random sampling is useful for testing to generate same
      sequence across calls.


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
      y_pred: Meshes object with predictions,
        storing the same number of meshes as y_true

    Returns:
      float32 scalar tensor containing the weighted Mesh R-CNN loss.
    """
    gt_points_with_normals = _sample_points_and_normals(
        y_true,
        gt_sample_size,
        seed=sampling_seed,
        stateless=stateless_sampling
    )

    pred_points_with_normals = _sample_points_and_normals(
        y_pred,
        pred_sample_size,
        seed=sampling_seed,
        stateless=stateless_sampling
    )

    dim = tf.cast(gt_points_with_normals.shape[-1] / 2., tf.int32)
    l_chamfer = chamfer_distance.evaluate(gt_points_with_normals[:, :dim],
                                          pred_points_with_normals[:, :dim])
    l_normal = normal_distance.evaluate(gt_points_with_normals,
                                        pred_points_with_normals)
    l_edge = edge_regularizer.evaluate(y_pred.get_padded()[0],
                                       y_pred.vertex_neighbors(),
                                       y_pred.get_sizes()[0])

    losses = tf.stack([l_chamfer, l_normal, l_edge], -1)

    return tf.reduce_mean(weigths * losses, -1)

  return evaluate


def _sample_points_and_normals(meshes, sample_size, seed=None, stateless=False):
  """
  Helper function to jointly sample points from a mesh together with associated
  face normal vectors.
  Args:
    meshes: A `Meshes` object containing a batch of meshes.
    sample_size: An int scalar denoting the number of points to be sampled from
      the mesh surface.
    seed: Optional random seed for the sampling op.
    stateless: Optional flag to use stateless random sampler. If stateless=True,
      then seed must be provided as shape `[2]` int tensor. Stateless random
      sampling is useful for testing to generate same sequence across calls.

  Returns:
    A float32 tensor of shape `[A1, ..., An, N, 2D]` containing
    the sampled points and normals of the provided mesh. On the last axis, the
    first D entries correspond to point locations and the last D
    entries correspond to normal vectors of the corresponding mesh faces
    in a D dimensional space.

  """
  padded_vertices, padded_faces = meshes.get_padded()
  batch_size = padded_vertices.shape[:-2].as_list()
  flat_batch_vertices = tf.reshape(padded_vertices,
                                   [-1] + padded_vertices.shape[-2:].as_list())
  flat_batch_faces = tf.reshape(padded_faces,
                                [-1] + padded_faces.shape[-2:].as_list())

  vertex_sizes, face_sizes = meshes.get_sizes()
  vertex_sizes = tf.reshape(vertex_sizes, [-1])
  face_sizes = tf.reshape(face_sizes, [-1])

  def sample_per_mesh(args):
    """Samples points and face indices per sampled point per mesh."""
    vertices, faces, b_id = args
    vertices = vertices[:vertex_sizes[b_id]]
    faces = faces[:face_sizes[b_id]]

    # return zero tensor for empty meshes, because otherwise it will fail
    # in `normals.gather_faces`
    if tf.reduce_all(vertices == 0):
      return tf.zeros((sample_size, vertices.shape[-1] * 2), tf.float32)

    points, face_idx = sampler.area_weighted_random_sample_triangle_mesh(
        vertices, faces, sample_size, seed=seed, stateless=stateless
    )

    face_positions = normals.gather_faces(vertices, faces)
    # Setting clockwise to false, since cubify outputs them in CCW order.
    face_normals = normals.face_normals(face_positions, clockwise=False)
    sampled_point_normals = tf.gather(face_normals, face_idx, axis=None)
    return tf.concat([points, sampled_point_normals], -1)

  # This workaround is needed, since the sampler does not support padded inputs.
  points_with_normals = tf.map_fn(
      sample_per_mesh,
      (
          flat_batch_vertices,
          flat_batch_faces,
          tf.range(padded_vertices.shape[0], dtype=tf.int32)
      ),
      dtype=tf.float32
  )

  return tf.reshape(points_with_normals,
                    batch_size + points_with_normals.shape[1:].as_list())
