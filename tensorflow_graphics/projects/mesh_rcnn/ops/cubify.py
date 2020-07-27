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
"""Implementation of the cubify operation for Mesh R-CNN."""

import tensorflow as tf


def _ravel_index(index, dims):
  """
  Computes a linear index in an array of shape dims.
  Reverse functionality of tf.unravel_index.
  Args:
    index: A Tensor with each row corresponding to indices into an array
    of dimension dims.
    dims: The shape of the array to be indexed.

  Note:
    Only supports dims=(H,W,D)

  Returns:
    Integer Tensor containing the index into an array of shape dims.
  """
  if len(dims) != 3:
    raise ValueError("Expects a 3-element list")
  if index.shape[1] != 3:
    raise ValueError("Expects an index tensor of shape Nx3")
  _, W, D = dims  # pylint: disable=C0103
  linear_index = index[:, 0] * W * D + index[:, 1] * D + index[:, 2]
  return linear_index


def cubify(voxel_grid, threshold=0.5):
  """
  Converts voxel occupancy probability grids into a triangle mesh.

  Each occupied voxel is replaced
  with a cuboid triangle mesh with 8 vertices, 18 edges, and
  12 faces. Shared vertices and edges between adjacent occupied voxels are
  merged, and shared interior faces are eliminated. This results in a
  watertight mesh whose topology depends on the voxel occupancy probabilities.


  Args:
    voxel_grid: flaot32 tensor of shape `[N, D, H, W]` containing the voxel
      occupancy probabilities.
    threshold: float32 denoting the threshold above which a voxel is
      considered occupied. Defaults to 0.5.

  Returns:
    Watertight mesh whose topology depends on the voxel occupancy probabilities.
  """
  voxel_grid = tf.convert_to_tensor(voxel_grid)
  threshold = tf.convert_to_tensor(threshold)

  if voxel_grid.shape.rank != 4:
    raise ValueError("Voxel Occupancy probability grid needs to be a Tensor "
                     "of Rank 4 with dimension: N, D, H, W.")

  N, D, H, W = voxel_grid.shape  # pylint: disable=C0103
  unit_cube_verts = tf.constant(
      [
          [0, 0, 0],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 1],
          [1, 0, 0],
          [1, 0, 1],
          [1, 1, 0],
          [1, 1, 1],
      ], dtype=tf.int32)

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
      dtype=tf.int32
  )

  voxel_mask = tf.math.greater_equal(voxel_grid, threshold)
  voxel_thresholded = tf.cast(voxel_mask, tf.float32)

  if tf.reduce_all(tf.math.logical_not(voxel_mask)):
    return [tf.constant([], dtype=tf.float32)], [
        tf.constant([], dtype=tf.float32)]

  # NDHWC, since NDCHW is only supported on GPU
  voxel_thresholded = tf.expand_dims(voxel_thresholded, axis=-1)

  kernel_x = tf.constant([.5, .5], shape=(1, 1, 2, 1, 1), dtype=tf.float32)
  kernel_y = tf.constant([.5, .5], shape=(1, 2, 1, 1, 1), dtype=tf.float32)
  kernel_z = tf.constant([.5, .5], shape=(2, 1, 1, 1, 1), dtype=tf.float32)

  # build filters for masks indicating whether the neighboring voxel is occupied.
  voxel_thresholded_x = tf.cast(
      tf.math.greater(
          tf.nn.conv3d(voxel_thresholded, kernel_x, strides=[1] * 5,
                       padding="VALID"),
          0.5),
      tf.float32)

  voxel_thresholded_y = tf.cast(
      tf.math.greater(
          tf.nn.conv3d(voxel_thresholded, kernel_y, strides=[1] * 5,
                       padding="VALID"),
          0.5),
      tf.float32)

  voxel_thresholded_z = tf.cast(
      tf.math.greater(
          tf.nn.conv3d(voxel_thresholded, kernel_z, strides=[1] * 5,
                       padding="VALID"),
          0.5),
      tf.float32)

  voxel_thresholded_x = tf.reshape(voxel_thresholded_x,
                                   shape=[N, 1, D, H, W - 1])
  voxel_thresholded_y = tf.reshape(voxel_thresholded_y,
                                   shape=[N, 1, D, H - 1, W])
  voxel_thresholded_z = tf.reshape(voxel_thresholded_z,
                                   shape=[N, 1, D - 1, H, W])

  # create masks in x directions
  x_left_faces = tf.concat([tf.ones((N, 1, D, H, 1), dtype=tf.float32),
                            1 - voxel_thresholded_x], -1)
  x_right_faces = tf.concat([1 - voxel_thresholded_x,
                             tf.ones((N, 1, D, H, 1), dtype=tf.float32)], -1)

  # create masks in y directions
  y_bottom_faces = tf.concat([1 - voxel_thresholded_y,
                              tf.ones((N, 1, D, 1, W), dtype=tf.float32)], -2)
  y_top_faces = tf.concat([tf.ones((N, 1, D, 1, W), dtype=tf.float32),
                           1 - voxel_thresholded_y], -2)

  # create masks in z directions
  z_front_faces = tf.concat([tf.ones((N, 1, 1, H, W), dtype=tf.float32),
                             1 - voxel_thresholded_z], -3)
  z_back_faces = tf.concat([1 - voxel_thresholded_z,
                            tf.ones((N, 1, 1, H, W), dtype=tf.float32)], -3)

  faces_indices = [
      x_left_faces,
      x_left_faces,  # faces 0, 1
      y_bottom_faces,
      y_bottom_faces,  # faces 2, 3
      z_front_faces,
      z_front_faces,  # faces 4, 5
      y_top_faces,
      y_top_faces,  # faces 6, 7
      x_right_faces,
      x_right_faces,  # faces 8, 9
      z_back_faces,
      z_back_faces,  # faces 10, 11
  ]

  # 12 x N x 1 x D x H x W
  faces_idx = tf.stack(faces_indices, axis=0)

  faces_idx *= tf.reshape(voxel_thresholded, shape=(N, 1, D, H, W))

  # N x H x W x D x 12
  faces_idx = tf.squeeze(tf.transpose(faces_idx, perm=(1, 2, 4, 5, 3, 0)),
                         axis=1)
  # (NHWD) x 12
  faces_idx = tf.reshape(faces_idx, shape=(-1, unit_cube_faces.shape[0]))
  linear_index = tf.cast(tf.where(tf.not_equal(faces_idx, 0)), tf.int32)
  nyxz = tf.transpose(tf.unravel_index(linear_index[:, 0], (N, H, W, D)))

  if len(nyxz) == 0:
    return [tf.constant([], dtype=tf.float32)], [
        tf.constant([], dtype=tf.float32)]

  faces = tf.gather(unit_cube_faces, linear_index[:, 1], axis=None)

  # Convert unit cube coordinates to voxel grid coordinates
  grid_faces = []
  for d in range(unit_cube_faces.shape[1]):
    xyz = tf.gather(unit_cube_verts, faces[:, d], axis=None)
    permute_idx = tf.constant([1, 0, 2])
    yxz = tf.gather(xyz, permute_idx, axis=1)
    yxz += nyxz[:, 1:]
    grid_faces.append(_ravel_index(yxz, (H + 1, W + 1, D + 1)))

  grid_faces = tf.stack(grid_faces, axis=1)

  x, y, z = tf.meshgrid(tf.range(W + 1), tf.range(H + 1), tf.range(D + 1))

  # alignment, so that the top left corner of each cuboid corresponds to the
  # pixel coordinate of the input grid.
  x = x * 2 / (W - 1) - 1.0
  y = y * 2 / (H - 1) - 1.0
  z = z * 2 / (D - 1) - 1.0

  grid_vertices = tf.reshape(tf.stack((x, y, z), axis=3), shape=(-1, 3))

  # prepare filtering / merge of vertices.
  n_vertices = grid_vertices.shape[0]
  grid_faces += tf.reshape(nyxz[:, 0], shape=(-1, 1)) * n_vertices
  idle_vertices = tf.ones(n_vertices * N, dtype=tf.int32)
  mask_indices, _ = tf.unique(tf.reshape(grid_faces, -1))
  mask_indices = tf.expand_dims(mask_indices, -1)
  idle_vertices = tf.tensor_scatter_nd_update(
      idle_vertices,
      mask_indices,
      tf.zeros((mask_indices.shape[0],), dtype=tf.int32)
  )
  grid_faces -= tf.reshape(nyxz[:, 0], shape=(-1, 1)) * n_vertices
  split_size = tf.math.bincount(nyxz[:, 0], minlength=N)
  faces_list = list(tf.split(grid_faces, split_size.numpy().tolist(), 0))
  idle_vertices = tf.reshape(idle_vertices, (N, n_vertices))
  idle_n = tf.math.cumsum(idle_vertices, axis=1)

  verts_list = [
      tf.gather(grid_vertices,
                tf.where((idle_vertices[n] == 0))[:, 0], axis=0) for n in
      range(N)
  ]

  faces_list = [face_batch - tf.gather(idle_n[n], face_batch, axis=0) for
                n, face_batch in enumerate(faces_list)]

  return verts_list, faces_list
