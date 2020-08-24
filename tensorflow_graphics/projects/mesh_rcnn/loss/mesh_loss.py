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
from tensorflow_graphics.util import shape


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
  w_normal = weights['normal']
  w_edge = weights['edge']

  weigths = tf.constant([w_chamfer, w_normal, w_edge])

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
    gt_vertices, gt_faces = y_true.get_flattened()
    pred_vertices, pred_faces = y_pred.get_flattened()

    dim = gt_vertices.shape[-1]

    # ToDo: check if cubify generated faces follow clockwise orientation.
    gt_points_with_normals = _sample_points_and_normals(gt_vertices,
                                                        gt_faces,
                                                        gt_sample_size)

    pred_points_with_normals = _sample_points_and_normals(pred_vertices,
                                                          pred_faces,
                                                          pred_sample_size)

    l_chamfer = chamfer_distance.evaluate(gt_points_with_normals[..., :dim],
                                          pred_points_with_normals[..., :dim])
    l_normal = normal_distance(gt_points_with_normals, pred_points_with_normals)
    # l_edge = edge_regularizer()

    return 0

  return mesh_rcnn_loss


def normal_distance(point_set_a, point_set_b):
  """Computes the normal distance between two point sets.

  Note:
    This is a symmetric version of the absolute normal distance, calculated as
    the sum of the average minimum distance of point normals from point set A to
    point normals of point set B and vice versa.

    The normal distance is defined as follows:

    $$
      L_{norm}(A,B) = -|A|^{-1} \sum_{a \in A}\min_{b \in B} |u_a \cdot u_b| -
      |B|^{-1} \sum_{b \in B} \min_{a \in A} |u_b \cdot u_a|
    $$

  Args:
    point_set_a: A float32 tensor of shape `[A1, ..., An, N, 2D]` containing
      the points and normals of the first point set. On the last axis, the
      first D entries should correspond to point locations and the last D
      entries should correspond to normal vectors in a D dimensional space.
    point_set_b: A float32 tensor of shape `[A1, ..., An, M, 2D]` containing
      the points and normals of the second point set. On the last axis, the
      first D entries should correspond to point locations and the last D
      entries should correspond to normal vectors in a D dimensional space.

    Returns:
      A float32 tensor of shape `[A1, ..., An]` storing the normal distances
      between the two point sets.

    Raises:
      ValueError: if the shape of `point_set_a`, `point_set_b` is not supported.
  """
  point_set_a = tf.convert_to_tensor(point_set_a)
  point_set_b = tf.convert_to_tensor(point_set_b)

  shape.compare_batch_dimensions(
      tensors=(point_set_a, point_set_b),
      tensor_names=("point_set_a", "point_set_b"),
      last_axes=-3,
      broadcast_compatible=True)
  # Verify that the last axis of the tensors has the same dimension.
  dimension = point_set_a.shape.as_list()[-1]
  shape.check_static(
      tensor=point_set_b,
      tensor_name="point_set_b",
      has_dim_equals=(-1, dimension))

  if not dimension % 2 == 0:
    raise ValueError('Last dimension of input must be evenly divisible by 2!')

  points_a, normals_a = tf.split(point_set_a, 2, axis=-1)
  points_b, normals_b = tf.split(point_set_b, 2, axis=-1)

  closest_point_normals_a_to_b, closest_point_normals_b_to_a = (
      _extract_normals_of_nearest_neighbors(
          point_set_a,
          point_set_b))

  normal_distances_a_to_b = tf.einsum('...i,...i->...',
                                      tf.abs(normals_a),
                                      tf.abs(closest_point_normals_a_to_b))
  normal_distances_b_to_a = tf.einsum('...i,...i->...',
                                      tf.abs(normals_b),
                                      tf.abs(closest_point_normals_b_to_a))
  return (- tf.reduce_mean(normal_distances_a_to_b, axis=-1) -
          tf.reduce_mean(normal_distances_b_to_a, axis=-1))


def edge_regularizer(edges):
  """ Computes an edge loss, which can be used as a shape regularizer for
  learning of high-quality mesh predictions.

  The edge loss is defined as follows:

  $$
    L_{edge}(V, E) = \frac{1}{E} \sum_{(v, v') \in E}||v - v'||^2
  $$

  where `E` represents the set of edges of a mesh.

  Args:
    edges: A float32 tensor of shape `[A1, ..., An, E, 2]` where E denotes the
    number of edges and A1, ..., An are optional batch dimensions.

  Returns:
    A float32 tensor of shape ´[A1, ..., An]´ storing the edge losses.
  """
  edges = tf.convert_to_tensor(edges)


def _sample_points_and_normals(vertices, faces, sample_size):
  """
  Helper function to jointly sample points from a mesh together with associated
  face normal vectors.
  Args:
    vertices: A float32 tensor of shape `[N, D]` representing the
      mesh vertices. D denotes the dimensionality of the input space.
    faces: A float32 tensor of shape `[M, V]` representing the
      mesh vertices. V denotes the number of vertices per face.
    sample_size:

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
  face_normals = normals.face_normals(face_positions)
  sampled_point_normals = tf.gather(face_normals, face_idx)
  points_with_normals = tf.concat([points, sampled_point_normals], -1)

  return points_with_normals


def _extract_normals_of_nearest_neighbors(point_set_a, point_set_b):
  """
  Extracts point normals of close points between two point sets. The nearest
  neighbors are computedbased on the L2 norm from one point set to another and
  vice versa.

  Args:
    point_set_a: A float32 tensor of shape `[A1, ..., An, N, 2D]` containing
      the points and normals of the first point set. On the last axis, the
      first D entries should correspond to point locations and the last D
      entries should correspond to normal vectors in a D dimensional space.
    point_set_b: A float32 tensor of shape `[A1, ..., An, M, 2D]` containing
      the points and normals of the second point set. On the last axis, the
      first D entries should correspond to point locations and the last D
      entries should correspond to normal vectors in a D dimensional space.

  Returns:
    * A float32 tensor of shape `[A1, ..., An, D]` containing normal vectors of
      the nearest neighbor from `point_set_b` for each point in `point_set_a`.
    * A float32 tensor of shape `[A1, ..., An, D]` containing normal vectors of
      the nearest neighbor from `point_set_a` for each point in `point_set_b`.
  """
  points_a, normals_a = tf.split(point_set_a, 2, axis=-1)
  points_b, normals_b = tf.split(point_set_b, 2, axis=-1)

  # Create N x M matrix where the entry i,j corresponds to ai - bj (vector of
  # dimension D).
  difference = (
      tf.expand_dims(points_a, axis=-2) -
      tf.expand_dims(points_b, axis=-3))
  # Calculate the square distances between each two points: |ai - bj|^2.
  square_distances = tf.einsum("...i,...i->...", difference, difference)

  nearest_neighbors_a_to_b = tf.argmin(square_distances, axis=-1)
  nearest_neighbors_b_to_a = tf.argmin(square_distances, axis=-2)

  point_dims_a = points_a.shape[-2:].as_list()
  point_dims_b = points_b.shape[-2:].as_list()

  normals_a_flat_batch = tf.reshape(normals_a, [-1] + point_dims_a)
  normals_b_flat_batch = tf.reshape(normals_b, [-1] + point_dims_b)

  idx_a2b_2d = tf.reshape(nearest_neighbors_a_to_b,
                          [-1, point_dims_a[0]])
  idx_b2a_2d = tf.reshape(nearest_neighbors_b_to_a,
                          [-1, point_dims_b[0]])

  nn_a2b_flat_batch = tf.vectorized_map(_gather_normals_fn,
                                        (normals_b_flat_batch, idx_a2b_2d))
  nn_b2a_flat_batch = tf.vectorized_map(_gather_normals_fn,
                                        (normals_a_flat_batch, idx_b2a_2d))
  normals_a2b = tf.reshape(nn_a2b_flat_batch, points_a.shape)
  normals_b2a = tf.reshape(nn_b2a_flat_batch, points_b.shape)

  return normals_a2b, normals_b2a


def _gather_normals_fn(args):
  """
  Function handle passed to tf.vectorized_map to extract normal vectors from a
  tensor with one batch dimension.
  Args:
    args: Tuple of 2 tensors:
      1. tensor: float32 of shape `[N, P, D]` storing the point normals of a D-
         dimensional space in the last axis.
      2. tensor: int32 of shape `[N, M]` storing the indices of the normals that
         should be extracted from the first tensor.

  Returns:
    A float32 tensor of shape `[N, M, D]` containing the sampled normal vectors.
  """
  params, idx = args
  return tf.gather(params, idx)
