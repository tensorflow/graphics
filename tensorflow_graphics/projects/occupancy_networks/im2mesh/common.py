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
""" NO COMMENT NOW"""

# import multiprocessing

import tensorflow as tf
import numpy as np
from im2mesh.utils.libkdtree import KDTree


MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]
"""
TODO: pytorch codes -> tensorflow
chamfer distance -> can use the one in TFG?
"""


def compute_iou(occ1, occ2):
  """ Computes the Intersection over Union (IoU) value for two sets of
  occupancy values.

  Args:
      occ1 (tensor): first set of occupancy values
      occ2 (tensor): second set of occupancy values
  """
  occ1 = np.asarray(occ1)
  occ2 = np.asarray(occ2)

  # Put all data in second dimension
  # Also works for 1-dimensional data
  if occ1.ndim >= 2:
    occ1 = occ1.reshape(occ1.shape[0], -1)
  if occ2.ndim >= 2:
    occ2 = occ2.reshape(occ2.shape[0], -1)

  # Convert to boolean values
  occ1 = occ1 >= 0.5
  occ2 = occ2 >= 0.5

  # Compute IOU
  area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
  area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

  iou = area_intersect / area_union

  return iou


def chamfer_distance(points1, points2, use_kdtree=True, give_id=False):
  """ Returns the chamfer distance for the sets of points.

  Args:
      points1 (tensor): first point set
      points2 (tensor): second point set
      use_kdtree (bool): whether to use a kdtree
      give_id (bool): whether to return the IDs of nearest points
  """
  # points1 and points2 might be torch.tensor....?
  if use_kdtree:
    return chamfer_distance_kdtree(points1, points2, give_id=give_id)
  else:
    return chamfer_distance_naive(points1, points2)


def chamfer_distance_naive(points1, points2):
  """ Naive implementation of the Chamfer distance.

  Args:
      points1 (numpy array): first point set
      points2 (numpy array): second point set
  """

  # points1 and points2 might be torch.tensor....?
  assert points1.shape == points2.shape
  batch_size, t, _ = points1.shape

  points1 = tf.reshape(points1, [batch_size, t, 1, 3])
  points2 = tf.reshape(points2, [batch_size, 1, t, 3])

  distances = tf.reduce_sum(tf.math.pow(points1 - points2, 2), axis=-1)

  chamfer1 = tf.reduce_mean(tf.reduce_min(distances, axis=1)[0], axis=1)
  chamfer2 = tf.reduce_mean(tf.reduce_min(distances, axis=2)[0], axis=1)

  chamfer = chamfer1 + chamfer2
  return chamfer


# TODO for psgn, dmc, pix2mesh
def chamfer_distance_kdtree(points1, points2, give_id=False):
  """ KD-tree based implementation of the Chamfer distance.

  Args:
      points1 (numpy array): first point set
      points2 (numpy array): second point set
      give_id (bool): whether to return the IDs of the nearest points
  """
  # Points have size batch_size x T x 3
  batch_size = points1.shape[0]

  # First convert points to numpy
  points1_np = points1.detach().numpy()
  points2_np = points2.detach().numpy()

  # Get list of nearest neighbors indieces
  idx_nn_12, _ = get_nearest_neighbors_indices_batch(points1_np, points2_np)
  idx_nn_12 = tf.constant(idx_nn_12, dtype=tf.int64)
  # Expands it as batch_size x 1 x 3
  idx_nn_12_expand = tf.broadcast_to(tf.reshape(
      idx_nn_12, shape=[batch_size, -1, 1]), shape=points1.shape)

  # Get list of nearest neighbors indieces
  idx_nn_21, _ = get_nearest_neighbors_indices_batch(points2_np, points1_np)
  idx_nn_21 = tf.constant(idx_nn_21, dtype=tf.int64)
  # Expands it as batch_size x T x 3
  idx_nn_21_expand = tf.broadcast_to(tf.reshape(
      idx_nn_21, shape=[batch_size, -1, 1]), shape=points2.shape)

  # Compute nearest neighbors in points2 to points in points1
  # points_12[i, j, k] = points2[i, idx_nn_12_expand[i, j, k], k]
  points_12 = tf.gather(points2, indices=idx_nn_12_expand,
                        axis=None, batch_dims=1)

  # Compute nearest neighbors in points1 to points in points2
  # points_21[i, j, k] = points2[i, idx_nn_21_expand[i, j, k], k]
  points_21 = tf.gather(points1, indices=idx_nn_21_expand,
                        axis=None, batch_dims=1)

  # Compute chamfer distance
  chamfer1 = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.pow(
      points1 - points_12, 2), axis=2), axis=1)
  chamfer2 = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.pow(
      points2 - points_21, 2), axis=2), axis=1)

  # Take sum
  chamfer = chamfer1 + chamfer2

  # If required, also return nearest neighbors
  if give_id:
    return chamfer1, chamfer2, idx_nn_12, idx_nn_21

  return chamfer


def get_nearest_neighbors_indices_batch(points_src, points_tgt, k=1):
  """ Returns the nearest neighbors for point sets batchwise.

  Args:
      points_src (numpy array): source points
      points_tgt (numpy array): target points
      k (int): number of nearest neighbors to return
  """
  indices = []
  distances = []

  for (p1, p2) in zip(points_src, points_tgt):
    kdtree = KDTree(p2)
    dist, idx = kdtree.query(p1, k=k)
    indices.append(idx)
    distances.append(dist)

  return indices, distances


# tensorflow
def normalize_imagenet(x):
  """ Normalize input images according to ImageNet standards.

  Args:
      x (tensor): input images
  """
  offset = tf.constant(MEAN_RGB, shape=[1, 1, 3])
  x -= offset

  scale = tf.constant(STDDEV_RGB, shape=[1, 1, 3])
  x /= scale

  return x


def make_3d_grid(bb_min, bb_max, shape):
  """ Makes a 3D grid.

  Args:
      bb_min (tuple): bounding box minimum
      bb_max (tuple): bounding box maximum
      shape (tuple): output shape
  """
  size = shape[0] * shape[1] * shape[2]

  pxs = tf.linspace(bb_min[0], bb_max[0], shape[0])
  pys = tf.linspace(bb_min[1], bb_max[1], shape[1])
  pzs = tf.linspace(bb_min[2], bb_max[2], shape[2])

  pxs = tf.reshape(
      tf.tile(tf.reshape(pxs, [-1, 1, 1]), [1, shape[1], shape[2]]), [size]
  )
  pys = tf.reshape(
      tf.tile(tf.reshape(pys, [1, -1, 1]), [shape[0], 1, shape[2]]), [size]
  )
  pzs = tf.reshape(
      tf.tile(tf.reshape(pzs, [1, 1, -1]), [shape[0], shape[1], 1]), [size]
  )
  p = tf.stack([pxs, pys, pzs], axis=1)

  return p


def transform_points(points, transform):
  """ Transforms points with regard to passed camera information.

  Args:
      points (tensor): points tensor
      transform (tensor): transformation matrices
  """
  assert points.shape[2] == 3
  assert transform.shape[1] == 3
  assert points.shape[0] == transform.shape[0]

  if transform.shape[2] == 4:
    r = transform[:, :, :3]
    t = transform[:, :, 3:]
    points_out = points @ tf.transpose(r, perm=[0, 2, 1]) + tf.transpose(
        t, perm=[0, 2, 1]
    )
  elif transform.shape[2] == 3:
    k = transform
    points_out = points @ tf.transpose(k, perm=[0, 2, 1])

  return points_out


def b_inv(b_mat):
  """ Performs batch matrix inversion.

  Arguments:
      b_mat: the batch of matrices that should be inverted
  """

  eye = tf.broadcast_to(tf.linalg.diag(
      tf.ones(shape=b_mat.shape[-1], dtype=b_mat.dtype)), shape=b_mat.shape)
  b_inv = tf.linalg.solve(b_mat, eye)
  return b_inv


def transform_points_back(points, transform):
  """ Inverts the transformation.

  Args:
      points (tensor): points tensor
      transform (tensor): transformation matrices
  """
  assert points.shape[2] == 3
  assert transform.shape[1] == 3
  assert points.shape[0] == transform.shape[0]

  if transform.shape[2] == 4:
    r = transform[:, :, :3]
    t = transform[:, :, 3:]
    points_out = points - tf.transpose(t, perm=[0, 2, 1])
    points_out = points_out @ b_inv(tf.transpose(r, perm=[0, 2, 1]))
  elif transform.shape[2] == 3:
    k = transform
    points_out = points @ b_inv(tf.transpose(k, perm=[0, 2, 1]))

  return points_out


def project_to_camera(points, transform):
  """ Projects points to the camera plane.

  Args:
      points (tensor): points tensor
      transform (tensor): transformation matrices
  """
  p_camera = transform_points(points, transform)
  p_camera = p_camera[..., :2] / p_camera[..., 2:]
  return p_camera


def get_camera_args(data, loc_field=None, scale_field=None):
  """ Returns dictionary of camera arguments.

  Args:
      data (dict): data dictionary
      loc_field (str): name of location field
      scale_field (str): name of scale field
  """
  rt = data["inputs.world_mat"]
  k = data["inputs.camera_mat"]

  if loc_field is not None:
    loc = data[loc_field]
  else:
    loc = tf.zeros([k.shape[0], 3], dtype=k.dtype)

  if scale_field is not None:
    scale = data[scale_field]
  else:
    scale = tf.zeros(K.shape[0], dtype=k.dtype)

  rt = fix_rt_camera(rt, loc, scale)
  k = fix_k_camera(k, img_size=137.0)
  kwargs = {"Rt": rt, "K": k}
  return kwargs


def fix_rt_camera(rt, loc, scale):
  """ Fixes Rt camera matrix.

  Args:
      rt (tensor): rt camera matrix
      loc (tensor): location
      scale (float): scale
  """
  # Rt is B x 3 x 4
  # loc is B x 3 and scale is B
  batch_size = rt.shape[0]
  r = rt[:, :, :3]
  t = rt[:, :, 3:]

  scale = tf.reshape(scale, [batch_size, 1, 1])
  r_new = r * scale
  t_new = t + r @ tf.expand_dims(loc, axis=2)

  rt_new = tf.concat([r_new, t_new], axis=2)

  assert rt_new.shape == (batch_size, 3, 4)
  return rt_new


def fix_k_camera(k, img_size=137):
  """Fix camera projection matrix.

  This changes a camera projection matrix that maps to
  [0, img_size] x [0, img_size] to one that maps to [-1, 1] x [-1, 1].

  Args:
      k (np.ndarray):     Camera projection matrix.
      img_size (float):   Size of image plane k projects to.
  """
  # Unscale and recenter
  scale_mat = tf.constant(
      [[2.0 / img_size, 0, -1], [0, 2.0 / img_size, -1], [0, 0, 1.0]],
      dtype=k.dtype,
  )

  k_new = tf.reshape(scale_mat, [1, 3, 3]) @ K
  return k_new
