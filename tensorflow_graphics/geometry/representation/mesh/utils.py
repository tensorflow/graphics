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
"""This module implements utility functions for meshes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def extract_unique_edges_from_triangular_mesh(faces, directed_edges=False):
  """Extracts all the unique edges using the faces of a mesh.

  Args:
    faces: A numpy.ndarray of shape [T, 3], where T is the number of triangular
      faces in the mesh. Each entry in this array describes the index of a
      vertex in the mesh.
    directed_edges: A boolean flag, whether to treat an edge as directed or
      undirected.  If (i, j) is an edge in the mesh and directed_edges is True,
      then both (i, j) and (j, i) are returned in the list of edges. If (i, j)
      is an edge in the mesh and directed_edges is False, then one of (i, j) or
      (j, i) is returned.

  Returns:
    A numpy.ndarray of shape [E, 2], where E is the number of edges in
    the mesh.


    For eg: given faces = [[0, 1, 2], [0, 1, 3]], then
      for directed_edges = False, one valid output is
        [[0, 1], [0, 2], [0, 3], [1, 2], [3, 1]]
      for directed_edges = True, one valid output is
        [[0, 1], [0, 2], [0, 3], [1, 0], [1, 2], [1, 3],
         [2, 0], [2, 1], [3, 0], [3, 1]]


  Raises:
    ValueError: If `faces` is not a numpy.ndarray or if its shape is not
      supported.
  """
  if not isinstance(faces, np.ndarray):
    raise ValueError("'faces' must be a numpy.ndarray.")
  faces_shape = faces.shape
  faces_rank = len(faces_shape)
  if faces_rank != 2:
    raise ValueError(
        "'faces' must have a rank equal to 2, but it has rank {} and shape {}."
        .format(faces_rank, faces_shape))
  if faces_shape[1] != 3:
    raise ValueError(
        "'faces' must have exactly 3 dimensions in the last axis, but it has {}"
        " dimensions and is of shape {}.".format(faces_shape[1], faces_shape))
  edges = np.concatenate([faces[:, 0:2], faces[:, 1:3], faces[:, [2, 0]]],
                         axis=0)
  if directed_edges:
    edges = np.concatenate([edges, np.flip(edges, axis=-1)])
  unique_edges = np.unique(edges, axis=0)
  return unique_edges


def get_degree_based_edge_weights(edges, dtype=np.float32):
  r"""Computes vertex degree based weights for edges of a mesh.

  The degree (valence) of a vertex is number of edges incident on the vertex.
  The weight for an edge $w_{ij}$ connecting vertex $v_i$ and vertex $v_j$
  is defined as,
  $$
  w_{ij} = 1.0 / degree(v_i)
  \sum_{j} w_{ij} = 1
  $$

  Args:
    edges: A numpy.ndarray of shape [E, 2], where E is the number of directed
      edges in the mesh.
    dtype: A numpy float data type. The output weights are of data type dtype.

  Returns:
    weights: A dtype numpy.ndarray of shape [E,] denoting edge weights.

  Raises:
    ValueError: If `edges` is not a numpy.ndarray or if its shape is not
      supported, or dtype is not a float type.

  """
  if not isinstance(dtype(1), np.floating):
    raise ValueError("'dtype' must be a numpy float type.")
  if not isinstance(edges, np.ndarray):
    raise ValueError("'edges' must be a numpy.ndarray.")
  edges_shape = edges.shape
  edges_rank = len(edges_shape)
  if edges_rank != 2:
    raise ValueError(
        "'edges' must have a rank equal to 2, but it has rank {} and shape {}."
        .format(edges_rank, edges_shape))
  if edges_shape[1] != 2:
    raise ValueError(
        "'edges' must have exactly 2 dimensions in the last axis, but it has {}"
        " dimensions and is of shape {}.".format(edges_shape[1], edges_shape))
  degree = np.bincount(edges[:, 0])
  rep_degree = degree[edges[:, 0]]
  weights = 1.0 / rep_degree.astype(dtype)
  return weights


# API contains all public functions and classes.
__all__ = []
