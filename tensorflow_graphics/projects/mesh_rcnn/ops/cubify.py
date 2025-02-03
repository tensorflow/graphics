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

from tensorflow_graphics.projects.mesh_rcnn.structures.mesh import Meshes
from tensorflow_graphics.util import shape


def _ravel_index(index, dims):
  """Computes a linear index into an array of shape dims.

  Provides the reverse functionality of tf.unravel_index.
  Args:
    index: An int Tensor of shape `[N, 3]`, where each row corresponds to an
      index into an array of dimension `dims`.
    dims: An int tensor of shape `[3,]` denoting shape of array to be indexed.

  Note:

    The shorthands used below are
      `H`: height of input space
      `W`: width of input space
      `D`: depth of input space

    Only supports dims=(H,W,D)

  Returns:
    Integer Tensor containing the index into an array of shape `dims`.
  """
  shape.check_static(tensor=index, has_rank=2)
  if len(dims) != 3:
    raise ValueError('Expects a 3-element list')
  if index.shape[1] != 3:
    raise ValueError('Expects an index tensor of shape Nx3')
  _, width, depth = dims
  linear_index = index[:, 0] * width * depth + index[:, 1] * depth + index[:, 2]
  return linear_index


def cubify(voxel_grid, threshold=0.5):
  """Converts voxel occupancy probability grids into a triangle mesh.

  Each occupied voxel is replaced
  with a cuboid triangle mesh with 8 vertices, 18 edges, and
  12 faces. Shared vertices and edges between adjacent occupied voxels are
  merged, and shared interior faces are eliminated. This results in a
  watertight mesh whose topology depends on the voxel occupancy probabilities.

  The shorthands used below are
    `V`: The number of vertices.
    `[A1, ..., An]`: optional batch dimensions
    `D`: depth of input space (representing z coordinates)
    `H`: height of input space (representing y coordinates)
    `W`: width of input space (representing x coordinates)

    The coordinates assume a Y-up convention.

  Args:
    voxel_grid: A float32 tensor of shape `[A1, ..., An, D, H, W]` containing
      the voxel occupancy probabilities.
    threshold: A float32 scalar denoting the threshold above which a voxel is
      considered occupied. Defaults to 0.5.

  Returns:
    Watertight mesh whose topology depends on the voxel occupancy probabilities.
    The Meshes are wrapped into an
    `tensorflow_graphics.projects.mesh_rcnn.structures.mesh.Meshes` object.

  Raises:
    ValueError: if input is of rank <= 2 or threshold is not a scalar.
  """
  voxel_grid = tf.convert_to_tensor(voxel_grid)
  threshold = tf.convert_to_tensor(threshold)

  shape.check_static(voxel_grid,
                     has_rank_greater_than=2,
                     tensor_name='voxel_grid')
  shape.check_static(threshold,
                     has_rank=0,
                     tensor_name='threshold')

  def _compute_grid_coordinates_fn(face_index):
    """Converts unit cube vertex coordinates to voxel grid coordinates."""
    xyz = tf.gather(unit_cube_verts,
                    face_index,
                    axis=None)
    permute_idx = tf.constant([1, 0, 2])
    yxz = tf.gather(xyz, permute_idx, axis=1)
    yxz += nyxz[:, 1:]
    return _ravel_index(yxz, (height + 1, width + 1, depth + 1))

  # compute batch sizes for output Meshes from input voxel grid.
  batch_sizes = voxel_grid.shape[:-3].as_list()
  if len(batch_sizes) == 0:
    batch_sizes = None

  # Flatten all batch dimensions. A1*A2*...*An = N.
  # New shape: N x D x H x W.
  voxel_grid = tf.reshape(voxel_grid, [-1] + voxel_grid.shape[-3:].as_list())

  batch_size, depth, height, width = voxel_grid.shape
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
    return Meshes([], [])

  kernel_x = tf.constant([.5, .5], shape=(1, 1, 2, 1, 1), dtype=tf.float32)
  kernel_y = tf.constant([.5, .5], shape=(1, 2, 1, 1, 1), dtype=tf.float32)
  kernel_z = tf.constant([.5, .5], shape=(2, 1, 1, 1, 1), dtype=tf.float32)

  x_left_faces, x_right_faces = _create_face_mask(voxel_thresholded, kernel_x,
                                                  -1)
  y_top_faces, y_bottom_faces = _create_face_mask(voxel_thresholded, kernel_y,
                                                  -2)
  z_front_faces, z_back_faces = _create_face_mask(voxel_thresholded, kernel_z,
                                                  -3)

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

  # 12 x batch_size x depth x height x width
  faces_idx = tf.stack(faces_indices, axis=0)

  # batch_size x height x width x depth x 12
  faces_idx = tf.transpose(faces_idx, perm=(1, 3, 4, 2, 0))

  faces_idx = tf.reshape(faces_idx, shape=(-1, unit_cube_faces.shape[0]))
  linear_index = tf.cast(tf.where(tf.not_equal(faces_idx, 0)), tf.int32)
  nyxz = tf.transpose(tf.unravel_index(linear_index[:, 0],
                                       (batch_size, height, width, depth)))

  if len(nyxz) == 0:
    return Meshes([], [])

  unit_cube_faces_per_occupied_voxel = tf.gather(unit_cube_faces,
                                                 linear_index[:, 1],
                                                 axis=None)

  grid_faces = tf.vectorized_map(_compute_grid_coordinates_fn,
                                 tf.transpose(
                                     unit_cube_faces_per_occupied_voxel))

  grid_faces = tf.transpose(grid_faces)

  x, y, z = tf.meshgrid(tf.range(width + 1),
                        tf.range(height + 1),
                        tf.range(depth + 1))

  # align the top left corner of each cuboid to the pixel coordinate of the
  # input grid.
  x = tf.cast(x * 2 / (width - 1) - 1.0, tf.float32)
  y = tf.cast(y * 2 / (height - 1) - 1.0, tf.float32)
  z = tf.cast(z * 2 / (depth - 1) - 1.0, tf.float32)

  grid_vertices = tf.reshape(tf.stack((x, y, z), axis=3), shape=(-1, 3))

  # prepare filtering / merge of vertices.
  n_vertices = grid_vertices.shape[0]
  grid_faces += tf.reshape(nyxz[:, 0], shape=(-1, 1)) * n_vertices
  idle_vertices = tf.ones(n_vertices * batch_size, dtype=tf.int32)
  mask_indices, _ = tf.unique(tf.reshape(grid_faces, -1))
  mask_indices = tf.expand_dims(mask_indices, -1)
  idle_vertices = tf.tensor_scatter_nd_update(
      idle_vertices,
      mask_indices,
      tf.zeros((mask_indices.shape[0],), dtype=tf.int32)
  )
  grid_faces -= tf.reshape(nyxz[:, 0], shape=(-1, 1)) * n_vertices
  split_size = tf.math.bincount(nyxz[:, 0], minlength=batch_size)
  faces_list = list(tf.split(grid_faces, split_size.numpy().tolist(), 0))
  idle_vertices = tf.reshape(idle_vertices, (batch_size, n_vertices))
  idle_n = tf.math.cumsum(idle_vertices, axis=1)

  verts_list = [
      tf.gather(grid_vertices,
                tf.where((idle_vertices[n] == 0))[:, 0], axis=0) for n in
      range(batch_size)
  ]

  faces_list = [face_batch - tf.gather(idle_n[n], face_batch, axis=0) for
                n, face_batch in enumerate(faces_list)]

  return Meshes(verts_list, faces_list, batch_sizes)


def _create_face_mask(voxel_occupancy_grid, kernel, axis):
  """Creates face masks along one axis of a voxel occupancy grid.


  The surfaces of the represented 3D shape are computed using 3D convolutions.

  Example:
    Consider a 1x2x2 fully occupied voxel grid the ordering ZYX. After applying
    this function for dimension 3 (x coordinates), one receives two float32
    tensors indicating for each voxel in the grid whether it is a boundary or
    an inside voxel. Thus, the lower bound along axis 3 (which corresponds to
    the left side) like:
    ```
    [[[1., 0.],
      [1., 0.]]]
    ```
    And the upper bound (i.e. right side) looks like:
    ```
    [[[0., 1.],
      [0., 1.]]]
    ```
    In this representation, 1. means that the voxel is the farthest voxel facing
    this direction. In other words, for the example above a 1. means that this
    voxel is the leftmost (or rightmost, for the upper bound) voxel in a
    subsequence of occupied voxels and thus is considered to be a part of the
    object surface.

  Args:
    voxel_occupancy_grid: float32 tensor of shape
      `[batch_size, depth, height, width]` representing a voxel occupancy grid.
    kernel: A Tensor. Must have the same type as input. Shape
      `[kernel_depth, kernel_height, kernel_width, in_channels,out_channels]`
    axis: int denoting the axis along which to convolve and extract face masks

  Returns:
    Two tensors of shape `[batch_size, depth, height, width]`. The first tensor
    represents faces along the lower bound of the specified axis and the second
    one represents the upper bound. See the Example above for details.
  """

  shape.check_static(voxel_occupancy_grid, has_rank=4)

  grid_shape = voxel_occupancy_grid.shape.as_list()

  # add channel dimension for convolutions, shape is (N, D, H, W, C)
  conv_input = tf.expand_dims(voxel_occupancy_grid, axis=-1)
  # build filters for masks indicating whether the neighboring voxel is occupied
  occupancy_mask = tf.cast(
      tf.math.greater(
          tf.nn.conv3d(conv_input,
                       kernel,
                       strides=[1, 1, 1, 1, 1],
                       padding="VALID"),
          0.5),
      tf.float32)

  grid_shape[axis] -= 1
  reshaped_occupancy = tf.reshape(tf.squeeze(occupancy_mask), shape=grid_shape)
  grid_shape[axis] = 1
  lower_bound = tf.concat([tf.ones(grid_shape, dtype=tf.float32),
                           1 - reshaped_occupancy], axis)

  upper_bound = tf.concat([1 - reshaped_occupancy,
                           tf.ones(grid_shape, dtype=tf.float32)], axis)

  lower_bound *= voxel_occupancy_grid
  upper_bound *= voxel_occupancy_grid
  return lower_bound, upper_bound
