"""
Created by Robin Baumann <https://github.com/RobinBaumann> at June 26, 2020.
"""

import numpy as np
import tensorflow.compat.v1 as tf


def _ravel_index(index, dims):
  """
  Computes a linear index in an array of shape dims.
  Reverse functionality of tf.unravel_index.
  Args:
    index: A Tensor with each row corresponding to indices into an array
    of dimension dims.
    dims: The shape of the array to be indexed.

  Returns:

  """
  strides = tf.cumprod(dims, exclusive=True, reverse=True)
  return tf.reduce_sum(index * tf.expand_dims(strides, 1), axis=0)


def cubify(voxel_grid, threshold):
  """
  Converts voxel occupancy probability grids into a triangle mesh.

  Each occupied voxel is replaced
  with a cuboid triangle mesh with 8 vertices, 18 edges, and
  12 faces. Shared vertices and edges between adjacent occupied voxels are
  merged, and shared interior faces are eliminated. This results in a
  watertight mesh whose topology depends on the voxel occupancy probabilities.


  Args:
    voxel_grid: flaot32 tensor of shape (N, D, H, W) containing the voxel
    occupancy probabilities.
    threshold: float32 denoting the threshold above which a voxel is considered occupied.

  Returns:
    A watertight mesh whose topology depends on the voxel occupancy probabilities.
  """
  # Todo shape tests.
  voxel_grid = tf.convert_to_tensor(voxel_grid)
  threshold = tf.convert_to_tensor(threshold)

  N, D, H, W = voxel_grid.shape
  unit_cube_verts = tf.constant(
    ((np.arange(8)[:, None] & (1 << np.arange(3))) > 0),
    dtype=tf.uint8)

  unit_cube_faces = tf.constant(
    [
      [0, 1, 2],
      [1, 3, 2],  # left face: 0, 1
      [2, 3, 6],
      [3, 7, 6],  # bottom face: 2, 3
      [0, 2, 6],
      [0, 6, 4],  # front face: 4, 5
      [0, 5, 1],
      [0, 4, 5],  # up face: 6, 7
      [6, 7, 5],
      [6, 5, 4],  # right face: 8, 9
      [1, 7, 3],
      [1, 5, 7],  # back face: 10, 11
    ],
    dtype=tf.uint8
  )

  # binarize voxel occupancy probabilities according to threshold
  voxel_thresholded = tf.cast(tf.math.greater_equal(voxel_grid, threshold),
                              dtype=tf.uint8)
  voxel_thresholded = tf.expand_dims(voxel_thresholded, axis=1)

  kernel_x = tf.constant([.5, .5], shape=(1, 1, 1, 1, 2), dtype=tf.float32)
  kernel_y = tf.constant([.5, .5], shape=(1, 1, 1, 2, 1), dtype=tf.float32)
  kernel_z = tf.constant([.5, .5], shape=(1, 1, 2, 1, 1), dtype=tf.float32)

  voxel_thresholded_x = tf.math.greater(
    tf.nn.conv3d(voxel_thresholded, kernel_x), 0.5)
  voxel_thresholded_y = tf.math.greater(
    tf.nn.conv3d(voxel_thresholded, kernel_y), 0.5)
  voxel_thresholded_z = tf.math.greater(
    tf.nn.conv3d(voxel_thresholded, kernel_z), 0.5)

  # 12 x N x 1 x D x H x W
  faces_idx = tf.ones((unit_cube_faces.shape(0), N, 1, D, H, W))

  # add left face
  faces_idx[0, :, :, :, :, 1:] = 1 - voxel_thresholded_x
  faces_idx[1, :, :, :, :, 1:] = 1 - voxel_thresholded_x
  # add bottom face
  faces_idx[2, :, :, :, :-1, :] = 1 - voxel_thresholded_y
  faces_idx[3, :, :, :, :-1, :] = 1 - voxel_thresholded_y
  # add front face
  faces_idx[4, :, :, 1:, :, :] = 1 - voxel_thresholded_z
  faces_idx[5, :, :, 1:, :, :] = 1 - voxel_thresholded_z
  # add up face
  faces_idx[6, :, :, :, 1:, :] = 1 - voxel_thresholded_y
  faces_idx[7, :, :, :, 1:, :] = 1 - voxel_thresholded_y
  # add right face
  faces_idx[8, :, :, :, :, :-1] = 1 - voxel_thresholded_x
  faces_idx[9, :, :, :, :, :-1] = 1 - voxel_thresholded_x
  # add back face
  faces_idx[10, :, :, :-1, :, :] = 1 - voxel_thresholded_z
  faces_idx[11, :, :, :-1, :, :] = 1 - voxel_thresholded_z

  faces_idx *= voxel_thresholded

  # N x H x W x D x 12
  faces_idx = tf.squeeze(tf.transpose(faces_idx, perm=(1, 2, 4, 5, 3, 0)),
                         axis=1)
  # (NHWD) x 12
  faces_idx = tf.reshape(faces_idx, shape=(-1, unit_cube_faces.shape[0]))

  linear_index = tf.where(tf.greater(tf.math.abs(faces_idx), 0))
  nyxz = tf.unravel_index(linear_index[:, 0], (N, H, W, D))

  if len(nyxz) == 0:
    verts_list = [tf.Tensor([], dtype=tf.float32)] * N
    faces_list = [tf.Tensor([], dtype=tf.int64)] * N
    return verts_list, faces_list

  faces = tf.gather(unit_cube_faces, linear_index[:, 1])

  grid_faces = []
  for d in range(unit_cube_faces.shape[1]):
    xyz = tf.gather(unit_cube_verts, faces[:, d])
    permute_idx = tf.constant([1, 0, 2])
    yxz = tf.gather(xyz, permute_idx, axis=1)
    yxz += nyxz[:, 1:]
    grid_faces.append(_ravel_index(yxz, (H + 1, W + 1, D + 1)))

  grid_faces = tf.stack(grid_faces, axis=1)

  x, y, z = tf.meshgrid(tf.range(W + 1), tf.range(H + 1), tf.range(D + 1))

  x = x * 2.0 / (W - 1) - 1.0
  y = y * 2.0 / (H - 1) - 1.0
  z = z * 2.0 / (D - 1) - 1.0

  grid_vertices = tf.stack((x, y, z), dim=3).reshape((-1, 3))

  n_vertices = grid_vertices.shape[0]
  grid_faces += nyxz[:, 0].reshape(-1, 1) * n_vertices
  idle_vertices = tf.ones(n_vertices * N, dtype=tf.uint8)
  idle_vertices = tf.tensor_scatter_add(idle_vertices,
                                        tf.reshape(grid_faces, -1), 0)
  grid_faces -= nyxz[:, 0].reshape((-1, 1)) * n_vertices
  split_size = tf.bincount(nyxz[:, 0], minlength=N)
  faces_list = list(tf.split(grid_faces, split_size.numpy().tolist(), 0))

  idle_vertices = tf.reshape(idle_vertices, (N, n_vertices))
  idle_n = tf.math.cumsum(idle_vertices, axis=1)

  verts_list = [
    tf.gather(grid_vertices,
              tf.where(
                tf.greater(tf.math.abs(idle_vertices[n]),
                           0))) for n in range(N)
  ]

  faces_list = [face - idle_n[n][face] for n, face in enumerate(faces_list)]

  return verts_list, faces_list
