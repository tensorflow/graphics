"""This module implements similar functions to utils in pure tensorfow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.util import export_api


def extract_unique_edges_from_triangular_mesh(
        faces, num_vertices, directed_edges=False, name=None):
  """Extracts all the unique edges using the faces of a mesh.

  Args:
    faces: A tensor of shape [T, 3], where T is the number of triangular
      faces in the mesh. Each entry in this array describes the index of a
      vertex in the mesh.
    num_vertices: A scalar, total number of vertices in the mesh. All values in faces
      must be less than this. Must be the same dtype as faces.
    directed_edges: A boolean flag, whether to treat an edge as directed or
      undirected.  If (i, j) is an edge in the mesh and directed_edges is True,
      then both (i, j) and (j, i) are returned in the list of edges.
      If (i, j) is an edge in the mesh and directed_edges is False,
      then one of (i, j) or (j, i) is returned.
    name: A name for this op. Defaults to
      `extract_unique_edges_from_triangular_mesh`.


  Returns:
    A tensor of shape [E, 2], where E is the number of edges in
    the mesh.


    For eg: given faces = [[0, 1, 2], [0, 1, 3]], then
      for directed_edges = False, one valid output is
        [[0, 1], [0, 2], [0, 3], [1, 2], [3, 1]]
      for directed_edges = True, one valid output is
        [[0, 1], [0, 2], [0, 3], [1, 0], [1, 2], [1, 3],
         [2, 0], [2, 1], [3, 0], [3, 1]]


  Raises:
    ValueError: If `faces` of `num_vertices` shape or dtype is not supported.
  """
  with tf.compat.v1.name_scope(
      name, "graph_convolution_feature_steered_convolution", [faces, num_vertices]):
    faces = tf.convert_to_tensor(faces)
    num_vertices = tf.convert_to_tensor(num_vertices, dtype=faces.dtype)
    if faces.shape.ndims != 2:
      raise ValueError(
            "'faces' must have a rank equal to 2, but it has rank {} and shape {}."
            .format(faces.shape.ndims, faces.shape))
    if faces.shape[1] != 3:
      raise ValueError(
          "'faces' must have exactly 3 dimensions in the last axis, but it has {}"
          " dimensions and is of shape {}."
          .format(faces.shape[1], faces.shape))
    if not faces.dtype.is_integer:
      raise ValueError(
          "'faces' must have integer type but has dtype {}".format(faces.dtype))
    if num_vertices.shape.ndims != 0:
      raise ValueError(
          "'num_vertices' must be a scalar but it has shape {}"
          .format(num_vertices.dtype))
    rolled_faces = tf.roll(faces, shift=-1, axis=1)
    # we could make indices by stacking faces and rolled faces
    # but unique requires our tensor to be 1D, so we'll ravel the index
    # that means there's no need to stack in the first place
    i = tf.reshape(faces, (-1, ))
    j = tf.reshape(rolled_faces, (-1, ))
    ravelled = i * num_vertices + j
    unique, _ = tf.unique(ravelled)
    indices = tf.unravel_index(unique, (num_vertices, num_vertices))
    indices = tf.transpose(indices, (1, 0))
    if directed_edges:
      indices = tf.concat((indices, tf.reverse(indices, axis=[1])), axis=0)
    return indices


def get_degree_based_edge_weights(edges, dtype=tf.float32, name=None):
  r"""Computes vertex degree based weights for edges of a mesh.

  The degree (valence) of a vertex is number of edges incident on the vertex.
  The weight for an edge $w_{ij}$ connecting vertex $v_i$ and vertex $v_j$
  is defined as,
  $$
  w_{ij} = 1.0 / degree(v_i)
  \sum_{j} w_{ij} = 1
  $$

  Args:
    edges: An int tensor of shape [E, 2],
      where E is the number of directed edges in the mesh.
    dtype: A float dtype. The output weights are of data type dtype.
    name: A name for this op. Defaults to `degree_based_edge_weights`.

  Returns:
    weights: A dtype tensor of shape [E,] denoting edge weights.

  Raises:
    ValueError: If `edges` is not of a supported type / shape, or dtype is
        not a floating dtype.
  """
  if not dtype.is_floating:
    raise ValueError("'dtype' must be a tensorflow float type.")
  with tf.compat.v1.name_scope(name, "degree_based_edge_weights", [edges]):
    edges = tf.convert_to_tensor(edges)
    if not edges.dtype.is_integer:
      raise ValueError(
        "'edges' must have an integer dtype but it is {}".format(edges.dtype))
    if edges.shape.ndims != 2:
      raise ValueError(
          "'edges' must have a rank equal to 2, but it has rank {} and shape {}."
          .format(edges.shape, edges.shape.ndims))
    if edges.shape[1] != 2:
      raise ValueError(
        "'edges' must have exactly 2 dimensions in the last axis, but it has {}"
        " dimensions and is of shape {}."
        .format(edges.shape[1], edges.shape))
    _, index, counts = tf.unique_with_counts(edges[:, 0])
    return 1. / tf.cast(tf.gather(counts, index), dtype)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
