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
""" tensorflow implemetations of the custom ops """

import tensorflow as tf
import numpy as np
from pylib.pc import PointCloud, Grid


def compute_keys_tf(point_cloud: PointCloud, num_cells, cell_size, name=None):
  """ Computes the regular grid cell keys of a point cloud.

    For a point in cell `c` the key is computed as
        \\(key = batch_id * prod_{d=0}^{D} num_cells_{d} + \\)
        \\(sum_{d=0}^{D}( c_{d} prod_{d'=d+1}^{D} num_cells_{d'} ) \\).
    Args:
      point_cloud: A `PointCloud` instance.
      num_cells: An `int` `Tensor` of shape `[D]`, the total number of cells
        per dimension.
      cell_size: An `int` `Tensor` of shape `[D]`, the cell sizes per
        dimension.

    Returns:
      An `int` `Tensor` of shape `[N]`, the keys per point.

  """
  with tf.compat.v1.name_scope(
      name, "compute keys", [point_cloud, num_cells, cell_size]):
    aabb = point_cloud.get_AABB()
    abb_min_per_batch = aabb._aabb_min
    aabb_min_per_point = tf.gather(abb_min_per_batch, point_cloud._batch_ids)
    cell_ind = tf.math.floor(
        (point_cloud._points - aabb_min_per_point) / cell_size)
    cell_ind = tf.cast(cell_ind, tf.int32)
    cell_ind = tf.minimum(
        tf.maximum(cell_ind, tf.zeros_like(cell_ind)),
        num_cells)
    cell_multiplier = tf.math.cumprod(num_cells, reverse=True)
    cell_multiplier = tf.concat((cell_multiplier, [1]), axis=0)
    keys = point_cloud._batch_ids * cell_multiplier[0] + \
        tf.math.reduce_sum(cell_ind * tf.reshape(cell_multiplier[1:], [1, -1]),
                           axis=1)
    return tf.cast(keys, tf.int64)
tf.no_gradient('ComputeKeysTF')


def build_grid_ds_tf(sorted_keys, num_cells, batch_size, name=None):
  """ Method to build a fast access data structure for point clouds.

  Creates a 2D regular grid in the first two dimension, saving the first and
  last index belonging to that cell array.

  Args:
    sorted_keys: An `int` `Tensor` of shape `[N]`, the sorted keys.
    num_cells: An `int` `Tensor` of shape `[D]`, the total number of cells
      per dimension.
    batch_size: An `int`.

  Returns:
    An `int` `Tensor` of shape `[batch_size, num_cells[0], num_cells[1], 2]`.

  """
  with tf.compat.v1.name_scope(
      name, 'build_grid_ds', [sorted_keys, num_cells, batch_size]):
    sorted_keys = tf.cast(tf.convert_to_tensor(value=sorted_keys), tf.int32)
    num_cells = tf.cast(tf.convert_to_tensor(value=num_cells), tf.int32)

    num_keys = sorted_keys.shape[0]
    num_cells_2D = batch_size * num_cells[0] * num_cells[1]
    if num_cells.shape[0] > 2:
        cells_per_2D_cell = tf.reduce_prod(num_cells[2:])
    elif num_cells.shape[0] == 2:
        cells_per_2D_cell = 1

    ds_indices = tf.cast(tf.floor(sorted_keys / cells_per_2D_cell),
                         dtype=tf.int32)
    indices = tf.range(0, num_keys, dtype=tf.int32)
    first_per_cell = tf.math.unsorted_segment_min(
        indices, ds_indices, num_cells_2D)
    last_per_cell = tf.math.unsorted_segment_max(
        indices + 1, ds_indices, num_cells_2D)

    empty_cells = first_per_cell < 0
    first_per_cell = tf.where(
        empty_cells, tf.zeros_like(first_per_cell), first_per_cell)
    last_per_cell = tf.where(
        empty_cells, tf.zeros_like(last_per_cell), last_per_cell)
    empty_cells = first_per_cell > num_keys
    first_per_cell = tf.where(
        empty_cells, tf.zeros_like(first_per_cell), first_per_cell)
    last_per_cell = tf.where(
        empty_cells, tf.zeros_like(last_per_cell), last_per_cell)

    return tf.stack([tf.reshape(first_per_cell,
                                [batch_size, num_cells[0], num_cells[1]]),
                     tf.reshape(last_per_cell,
                                [batch_size, num_cells[0], num_cells[1]])],
                    axis=3)
tf.no_gradient('BuildGridDsTF')


def find_neighbors_tf(grid,
                      point_cloud_centers,
                      radii,
                      max_neighbors=0,
                      name=None):
  """ Method to find the neighbors of a center point cloud in another
  point cloud.

  Args:
    grid: A `Grid` instance, from which the neighbors are chosen.
    point_cloud_centers: A `PointCloud` instance, containing the center points.
    radii: A `float`, the radii to select neighbors from.
    max_neighbors: An `int`, if `0` all neighbors are selected.

  Returns:
  center_neigh_ranges: An `int` `Tensor` of shape `[N]`, end of the ranges per
      center point. You can get the neighbor ids of point `i` (i>0) with
        `neighbors[center_neigh_ranges[i-1]:center_neigh_ranges[i]]`.
  neighbors: An `int` `Tensor` of shape `[M, 2]`, indices of the neighbor
      point and the center for each neighbor. Follows the order of
      `grid._sorted_points`.

  """
  with tf.compat.v1.name_scope(
      name, "find neighbours",
      [grid, point_cloud_centers, radii, max_neighbors]):
    radii = tf.convert_to_tensor(value=radii)
    if radii.shape[0] == [] or radii.shape[0] == 1:
      radii = tf.repeat(radii, grid._point_cloud._dimension)
    # compute keys of center points in neighbors 2D grid
    center_points = point_cloud_centers._points
    center_batch_ids = point_cloud_centers._batch_ids
    aabb = grid._aabb
    abb_min_per_batch_2D = aabb._aabb_min[:, :2]
    aabb_min_per_center_point_2D = tf.gather(
        abb_min_per_batch_2D, center_batch_ids)
    center_cell_ind_2D = tf.math.floor(
        (center_points[:, :2] - aabb_min_per_center_point_2D) / radii[:2])
    center_cell_ind_2D = tf.cast(center_cell_ind_2D, tf.int32)
    center_cell_ind_2D = tf.minimum(
        tf.maximum(center_cell_ind_2D, tf.zeros_like(center_cell_ind_2D)),
        grid._num_cells[:2])
    # find neighbors using fast 2D grid datastructure
    neighbor_points = grid._sorted_points
    neighbor_batch_ids = grid._sorted_batch_ids
    data_structure = grid.get_DS()

    neighbors = []
    center_neigh_ranges = []
    cur_neigh_range = 0
    for i in range(center_points.shape[0]):
      cur_point = center_points[i]
      cur_batch_id = center_batch_ids[i]
      # get cell_ids of adjacent 2D cells (9 in total)
      cur_cell_id_2D = center_cell_ind_2D[i]
      adj_cell_ids_2D = tf.stack(
          (cur_cell_id_2D + [-1, -1],
           cur_cell_id_2D + [-1, 0],
           cur_cell_id_2D + [-1, 1],
           cur_cell_id_2D + [0, 1],
           cur_cell_id_2D,
           cur_cell_id_2D + [0, -1],
           cur_cell_id_2D + [1, -1],
           cur_cell_id_2D + [1, 0],
           cur_cell_id_2D + [1, 1]), axis=0)
      # clip to range between 0 and max num cells
      adj_cell_ids_2D = tf.minimum(
        tf.maximum(adj_cell_ids_2D, tf.zeros_like(adj_cell_ids_2D)),
        grid._num_cells[:2])
      # get min and max point ids of the adjacent cells
      adj_ids = tf.gather_nd(data_structure[cur_batch_id], [adj_cell_ids_2D])
      adj_ids_start = tf.reduce_min(adj_ids[0, :, 0])
      adj_ids_end = tf.reduce_max(adj_ids[0, :, 1])
      # choose points below certain distance and in same batch
      adj_points = neighbor_points[adj_ids_start:adj_ids_end]
      adj_batch_ids = neighbor_batch_ids[adj_ids_start:adj_ids_end]
      distances = tf.linalg.norm(
          adj_points - tf.reshape(cur_point, [1, -1]), axis=1)
      close = (distances < radii[0])
      same_batch = (adj_batch_ids == cur_batch_id)
      close = tf.math.logical_and(close, same_batch)
      close_ids = tf.boolean_mask(tf.range(adj_ids_start, adj_ids_end), close)

      cur_neighbors = tf.stack(
          (close_ids, tf.ones_like(close_ids) * i), axis=1)
      neighbors.append(cur_neighbors)
      cur_neigh_range = cur_neigh_range + cur_neighbors.shape[0]
      center_neigh_ranges.append(cur_neigh_range)

    neighbors = tf.concat(neighbors, axis=0)
    center_neigh_ranges = tf.concat(center_neigh_ranges, axis=0)

    return center_neigh_ranges, neighbors
tf.no_gradient('FindNeighborsTF')


def find_neighbors_no_grid(point_cloud,
                           point_cloud_centers,
                           radius,
                           name=None):
  """ Method to find the neighbors of a center point cloud in another
  point cloud.

  Args:
    point_cloud: A `PointCloud` instance, from which the neighbors are chosen.
    point_cloud_centers: A `PointCloud` instance, containing the center points.
    radius: A `float`, the radius to select neighbors from.

  Returns:
  center_neigh_ranges: An `int` `Tensor` of shape `[N]`, end of the ranges per
      center point. You can get the neighbor ids of point `i` (i>0) with
        `neighbors[center_neigh_ranges[i-1]:center_neigh_ranges[i]]`.
  neighbors: An `int` `Tensor` of shape `[M, 2]`, indices of the neighbor
      point and the center for each neighbor. Follows the order of
      `grid._sorted_points`.

  """
  with tf.compat.v1.name_scope(
      name, 'find neighbors',
      [point_cloud, point_cloud_centers, radius]):
    points = point_cloud._points
    batch_ids = point_cloud._batch_ids
    center_points = point_cloud_centers._points
    center_batch_ids = point_cloud_centers._batch_ids
    num_center_points = center_points.shape[0]

    distances = tf.linalg.norm(tf.expand_dims(points, axis=0) - \
                               tf.expand_dims(center_points, axis=1),
                               axis=-1)
    close = (distances <= radius)
    same_batch = (tf.expand_dims(batch_ids, axis=0) == \
                  tf.expand_dims(center_batch_ids, axis=1))
    close = tf.math.logical_and(same_batch, close)

    neighbors = tf.where(close)
    neighbors = tf.reverse(neighbors, axis=[1])
    num_neighbors = neighbors.shape[0]
    neigh_ranges = tf.math.unsorted_segment_max(
        tf.range(1, num_neighbors + 1),
        neighbors[:, 1],
        num_center_points)
  return neigh_ranges, neighbors
tf.no_gradient('FindNeighborsNoGrid')


def sampling_tf(neighborhood, sample_mode, name=None):
  """ Method to sample the points of a point cloud.

  Args:
    neighborhood: A `Neighborhood` instance, which contains a point cloud with
      its neighbors.
    sample_mode: An `int`specifiying the sample mode,
      `0` for average, `1` for poisson.

  Returns:
    sampled_points: A `float` `Tensor` of shape [S, D], the sampled points.
    sampled_batch_ids: An `int` `Tensor` of shape [S], the batch ids.
    sampled_indices: An `int` `Tensor` of shape [S], the indices to the
      unsampled points.
      Following the order of neighborhood._grid._sorted_points.

  """
  with tf.compat.v1.name_scope(name, "sampling", [neighborhood, sample_mode]):
    points = neighborhood._grid._sorted_points
    batch_ids = neighborhood._grid._sorted_batch_ids
    num_points = points.shape[0]
    if sample_mode == 0:
      # poisson sampling
      nb_ranges = tf.concat(([0], neighborhood._samples_neigh_ranges), axis=0)
      neighbors = neighborhood._neighbors
      num_points = neighborhood._grid._sorted_points.shape[0]
      log_probabilities = tf.ones([num_points])
      sampled_indices = tf.zeros([0], dtype=tf.int64)
      # to set log prob to -inf <=> prob to zero
      tf_neg_inf = tf.constant(-np.inf)

      #sample points until all log probabilites are set to -inf
      while not tf.reduce_all(tf.math.is_inf(log_probabilities)):
        choice = tf.random.categorical(
            tf.expand_dims(log_probabilities, axis=0), 1)[0]
        # add choice to sampled indices
        sampled_indices = tf.concat((sampled_indices, choice), axis=0)
        # set log probability of neighbors to -inf
        sample_neighbors = \
            neighbors[nb_ranges[choice[0]]:nb_ranges[choice[0] + 1], 0]
        num_neighbors = sample_neighbors.shape[0]
        log_probabilities = tf.tensor_scatter_nd_update(
            log_probabilities,
            tf.expand_dims(sample_neighbors, axis=1),
            tf.repeat(tf_neg_inf, num_neighbors))
      sampled_points = tf.gather(points, sampled_indices)
      sampled_batch_ids = tf.gather(batch_ids, sampled_indices)

    elif sample_mode == 1:
      # cell average sampling
      keys = neighborhood._grid._sorted_keys
      # replace keys with numbers 0 to num_unique keys
      unique, _, counts = tf.unique_with_counts(keys)
      num_unique_keys = unique.shape[0]
      new_keys = tf.repeat(tf.range(0, num_unique_keys), counts)
      # average over points with same cell key
      sampled_points = tf.math.segment_mean(points, new_keys)
      # get batch of a point in the same cell
      sampled_indices = tf.math.segment_min(tf.range(0, num_points), new_keys)
      sampled_batch_ids = tf.gather(batch_ids, sampled_indices)

    return sampled_points, sampled_batch_ids, sampled_indices

tf.no_gradient('samplingTF')


_pi = tf.constant(np.pi)


def compute_pdf_inside_neighborhoods_tf(neighborhood,
                                        bandwidth,
                                        mode,
                                        name=None):
  """ Method to compute the density distribution inside the neighborhoods of a
  point cloud in euclidean space using kernel density estimation (KDE).

  Args:
    neighborhood: A `Neighborhood` instance.
    bandwidth: An `int` `Tensor` of shape `[D]`, the bandwidth of the KDE.
    mode: A `KDEMode` value.

  Returns:
    A `float` `Tensor` of shape `[N]`, the estimated density per center point.

  """
  with tf.compat.v1.name_scope(
      name, "compute pdf with point gradients",
      [neighborhood, bandwidth, mode]):
    bandwidth = tf.convert_to_tensor(value=bandwidth)
    points = neighborhood._grid._sorted_points
    neighbors = neighborhood._neighbors
    nbh_ranges = neighborhood._samples_neigh_ranges

    # compute difference vectors inside neighborhoods
    num_adjacencies = neighbors.shape[0]
    nbh_start_ind = tf.concat(([0], nbh_ranges[0:-1]), axis=0)
    nbh_sizes = nbh_ranges - nbh_start_ind
    max_num_neighbors = tf.reduce_max(nbh_sizes)
    nbh_sizes_per_nb = tf.repeat(nbh_sizes, nbh_sizes)

    nb_indices_1 = tf.repeat(neighbors[:, 0], nbh_sizes_per_nb)

    mask = tf.sequence_mask(nbh_sizes_per_nb, max_num_neighbors)
    mask_indices = tf.cast(tf.compat.v1.where(mask), tf.int32)
    indices_tensor = tf.repeat(tf.reshape(tf.range(0, max_num_neighbors),
                                          [1, max_num_neighbors]),
                               num_adjacencies, axis=0)
    nbh_start_per_nb = tf.repeat(nbh_start_ind, nbh_sizes)
    indices_tensor = indices_tensor + \
        tf.reshape(nbh_start_per_nb, [num_adjacencies, 1])
    indices_2 = tf.gather_nd(params=indices_tensor, indices=mask_indices)
    nb_indices_2 = tf.gather(neighbors[:, 0], indices_2)

    nb_diff = tf.gather(points, nb_indices_1) - tf.gather(points, nb_indices_2)
    # kernel density estimation using the distances
    rel_bandwidth = tf.reshape(bandwidth * neighborhood._radii, [1, -1])
    kernel_input = nb_diff / rel_bandwidth
    # gaussian kernel
    nb_kernel_value = tf.exp(-tf.pow(kernel_input, 2) / 2) / tf.sqrt(2 * _pi)
    nb_kernel_value = tf.reduce_prod(nb_kernel_value, axis=1)
    nb_id_per_nb_pair = tf.repeat(tf.range(0, num_adjacencies),
                                  nbh_sizes_per_nb)
    # sum over influence inside neighborhood
    pdf = tf.math.unsorted_segment_sum(nb_kernel_value,
                                       nb_id_per_nb_pair,
                                       num_adjacencies) /\
        tf.reduce_prod(bandwidth)
    return pdf


def compute_pdf_tf(neighborhood, bandwidth, mode, name=None):
  """ Method to compute the density distribution using neighborhood information
  in euclidean space using kernel density estimation (KDE).

  Args:
    neighborhood: A `Neighborhood` instance of the pointcloud to itself.
    bandwidth: An `int` `Tensor` of shape `[D]`, the bandwidth of the KDE.
    mode: A `KDEMode` value.

  Returns:
    A `float` `Tensor` of shape `[N]`, the estimated density per point,
      with respect to the sorted points of the grid in `neighborhood`.

  """
  with tf.compat.v1.name_scope(
      name, "compute pdf with point gradients",
      [neighborhood, bandwidth, mode]):
    bandwidth = tf.convert_to_tensor(value=bandwidth)

    rel_bandwidth = tf.reshape(bandwidth * neighborhood._radii, [1, -1])
    points = neighborhood._grid._sorted_points / rel_bandwidth
    num_points = points.shape[0]
    neighbors = neighborhood._neighbors
    # point differences
    nb_diff = tf.gather(points, neighbors[:, 0]) - \
        tf.gather(points, neighbors[:, 1])
    # kde on point differences
    # gaussian kernel, note division by bandwidth was already done above
    nb_kernel_value = tf.exp(-tf.pow(nb_diff, 2) / 2) / tf.sqrt(2 * _pi)
    nb_kernel_value = tf.reduce_prod(nb_kernel_value, axis=1)
    # sum over influence of neighbors
    pdf = tf.math.unsorted_segment_sum(nb_kernel_value,
                                       neighbors[:, 1],
                                       num_points) / \
        tf.reduce_prod(bandwidth)
    return pdf


def basis_proj_tf(neigh_basis, features, neighborhood, name=None):
  """ Method to aggregate the features*basis for different neighborhoods.

  Args:
    neigh_basis: A `float` `Tensor` of shape `[M, H]`, the projection of
        each neighbor to the different basis.
    features: A `float` `Tensor` of shape `[N_in, C]`, the input features.
    neighborhood: A `Neighborhood` instance.

  Returns:
    A `float` `Tensor` of shape ``[N_out, C, H]`, the weighted latent features.
  """
  with tf.compat.v1.name_scope(
        name,
        'basis_projection',
        [neigh_basis, neighborhood, features]
                              ):
    neigh_basis = tf.convert_to_tensor(value=neigh_basis, dtype=tf.float32)
    features = tf.convert_to_tensor(value=features, dtype=tf.float32)
    # get input in correct shapes
    num_nbh = neighborhood._point_cloud_sampled._points.shape[0]
    features_per_nb = tf.gather(features,
                                neighborhood._original_neigh_ids[:, 0])
    # Monte-Carlo Integration
    weighted_features_per_nb = tf.expand_dims(features_per_nb, 2) *\
        tf.expand_dims(neigh_basis, 1)
    weighted_latent_per_center = tf.math.unsorted_segment_sum(
        weighted_features_per_nb, neighborhood._neighbors[:, 1], num_nbh)
    return weighted_latent_per_center
