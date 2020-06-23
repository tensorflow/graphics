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
# Lint as: python3
"""Utility modules for reconstructing scenes.
"""

import os
import numpy as np
from skimage import measure
import tensorflow.compat.v1 as tf

from tensorflow_graphics.projects.local_implicit_grid.core import evaluator
from tensorflow_graphics.projects.local_implicit_grid.core import local_implicit_grid_layer as lig
from tensorflow_graphics.projects.local_implicit_grid.core import point_utils as pt


class LIGOptimizer(object):
  """Class for using optimization to acquire feature grid."""

  def __init__(self, ckpt, origin, grid_shape, part_size, occ_idx,
               indep_pt_loss=True, overlap=True, alpha_lat=1e-2, npts=2048,
               init_std=1e-2, learning_rate=1e-3, var_prefix='', nows=False):
    self.ckpt = ckpt
    self.ckpt_dir = os.path.dirname(ckpt)
    self.params = self._load_params(self.ckpt_dir)
    self.origin = origin
    self.grid_shape = grid_shape
    self.part_size = part_size
    self.occ_idx = occ_idx
    self.init_std = init_std
    self.learning_rate = learning_rate
    self.var_prefix = var_prefix
    self.nows = nows

    self.xmin = self.origin
    if overlap:
      true_shape = (np.array(grid_shape) - 1) / 2.0
      self.xmax = self.origin + true_shape * part_size
    else:
      self.xmax = self.origin + (np.array(grid_shape) - 1) * part_size

    _, sj, sk = self.grid_shape
    self.occ_idx_flat = (self.occ_idx[:, 0]*(sj*sk)+
                         self.occ_idx[:, 1]*sk+self.occ_idx[:, 2])

    self.indep_pt_loss = indep_pt_loss
    self.overlap = overlap
    self.alpha_lat = alpha_lat
    self.npts = int(npts)
    self._init_graph()

  def _load_params(self, ckpt_dir):
    param_file = os.path.join(ckpt_dir, 'params.txt')
    params = evaluator.parse_param_file(param_file)
    return params

  def _init_graph(self):
    """Initialize computation graph for tensorflow.
    """
    self.graph = tf.Graph()
    with self.graph.as_default():
      self.point_coords_ph = tf.placeholder(
          tf.float32,
          shape=[1, self.npts, 3])  # placeholder
      self.point_values_ph = tf.placeholder(
          tf.float32,
          shape=[1, self.npts, 1])  # placeholder

      self.point_coords = self.point_coords_ph
      self.point_values = self.point_values_ph
      self.liggrid = lig.LocalImplicitGrid(
          size=self.grid_shape,
          in_features=self.params['codelen'],
          out_features=1,
          num_filters=self.params['refiner_nf'],
          net_type='imnet',
          method='linear' if self.overlap else 'nn',
          x_location_max=(1.0 if self.overlap else 2.0),
          name='lig',
          interp=(not self.indep_pt_loss),
          min_grid_value=self.xmin,
          max_grid_value=self.xmax)

      si, sj, sk = self.grid_shape
      self.occ_idx_flat_ = tf.convert_to_tensor(
          self.occ_idx_flat[:, np.newaxis])
      self.shape_ = tf.constant([si*sj*sk, self.params['codelen']],
                                dtype=tf.int64)
      self.feat_sparse_ = tf.Variable(
          (tf.random.normal(shape=[self.occ_idx.shape[0],
                                   self.params['codelen']]) *
           self.init_std),
          trainable=True,
          name='feat_sparse')
      self.feat_grid = tf.scatter_nd(self.occ_idx_flat_,
                                     self.feat_sparse_,
                                     self.shape_)
      self.feat_grid = tf.reshape(self.feat_grid,
                                  [1, si, sj, sk, self.params['codelen']])
      self.feat_norm = tf.norm(self.feat_sparse_, axis=-1)

      if self.indep_pt_loss:
        self.preds, self.weights = self.liggrid(self.feat_grid,
                                                self.point_coords,
                                                training=True)
        # preds: [b, n, 8, 1], weights: [b, n, 8]
        self.preds_interp = tf.reduce_sum(
            tf.expand_dims(self.weights, axis=-1)*self.preds,
            axis=2)  # [b, n, 1]
        self.preds = tf.concat([self.preds,
                                self.preds_interp[:, :, tf.newaxis, :]],
                               axis=2)  # preds: [b, n, 9, 1]
        self.point_values = tf.broadcast_to(
            self.point_values[:, :, tf.newaxis, :],
            shape=self.preds.shape)  # [b, n, 9, 1]
      else:
        self.preds = self.liggrid(self.feat_grid,
                                  self.point_coords,
                                  training=True)  # [b, n, 1]

      self.labels_01 = (self.point_values+1) / 2  # turn labels to 0, 1 labels
      self.loss_pt = tf.losses.sigmoid_cross_entropy(
          self.labels_01,
          logits=self.preds,
          reduction=tf.losses.Reduction.NONE)
      self.loss_lat = tf.reduce_mean(self.feat_norm) * self.alpha_lat
      self.loss = tf.reduce_mean(self.loss_pt) + self.loss_lat

      # compute accuracy metric
      if self.indep_pt_loss:
        self.pvalue = tf.sign(self.point_values[:, :, -1, 0])
        self.ppred = tf.sign(self.preds[:, :, -1, 0])
      else:
        self.pvalue = tf.sign(self.point_values[..., 0])
        self.ppred = tf.sign(self.preds[:, :, 0])
      self.accu = tf.reduce_sum(tf.cast(
          tf.logical_or(tf.logical_and(self.pvalue > 0, self.ppred > 0),
                        tf.logical_and(self.pvalue < 0, self.ppred < 0)),
          tf.float32)) / float(self.npts)

      # get optimizer
      self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
      self.fgrid_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='feat_sparse')
      self.train_op = self.optimizer.minimize(
          self.loss,
          global_step=tf.train.get_or_create_global_step(),
          var_list=[self.fgrid_vars])

      self.map_dict = self._get_var_mapping(model=self.liggrid,
                                            scope=self.var_prefix)
      self.sess = tf.Session()
      if not self.nows:
        self.saver = tf.train.Saver(self.map_dict)
        self.saver.restore(self.sess, self.ckpt)
      self._initialize_uninitialized(self.sess)

  def _get_var_mapping(self, model, scope=''):
    vars_ = model.trainable_variables
    varnames = [v.name for v in vars_]  # .split(':')[0]
    varnames = [scope+v.replace('lig/', '').strip(':0') for v in varnames]
    map_dict = dict(zip(varnames, vars_))
    return map_dict

  def _initialize_uninitialized(self, sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run(
        [tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars,
                                                is_not_initialized) if not f]

    if not_initialized_vars:
      sess.run(tf.variables_initializer(not_initialized_vars))

  def optimize_feat_grid(self, point_coords, point_vals, steps=10000,
                         print_every_n_steps=1000):
    """Optimize feature grid.

    Args:
      point_coords: [npts, 3] point coordinates.
      point_vals: [npts, 1] point values.
      steps: int, number of steps for gradient descent.
      print_every_n_steps: int, print every n steps.
    Returns:

    """
    print_every_n_steps = int(print_every_n_steps)

    point_coords = point_coords.copy()
    point_vals = np.sign(point_vals.copy())

    if point_coords.ndim == 3:
      point_coords = point_coords[0]
    if point_vals.ndim == 3:
      point_vals = point_vals[0]
    elif point_vals.ndim == 1:
      point_vals = point_vals[:, np.newaxis]

    # clip
    point_coords = np.clip(point_coords, self.xmin, self.xmax)

    # shuffle points
    seq = np.random.permutation(point_coords.shape[0])
    point_coords = point_coords[seq]
    point_vals = point_vals[seq]

    point_coords = point_coords[np.newaxis]
    point_vals = point_vals[np.newaxis]

    # random point sampling function
    def random_point_sample():
      sid = np.random.choice(point_coords.shape[1]-self.npts+1)
      eid = sid + self.npts
      return point_coords[:, sid:eid], point_vals[:, sid:eid]

    with self.graph.as_default():
      for i in range(steps):
        pc, pv = random_point_sample()
        accu_, loss_, _ = self.sess.run([self.accu, self.loss, self.train_op],
                                        feed_dict={
                                            self.point_coords_ph: pc,
                                            self.point_values_ph: pv})
        if i % print_every_n_steps == 0:
          print('Step [{:6d}] Accu: {:5.4f} Loss: {:5.4f}'.format(i,
                                                                  accu_, loss_))

  @property
  def feature_grid(self):
    with self.graph.as_default():
      return self.sess.run(self.feat_grid)


def occupancy_sparse_to_dense(occ_idx, grid_shape):
  dense = np.zeros(grid_shape, dtype=np.bool).ravel()
  occ_idx_f = (occ_idx[:, 0] * grid_shape[1] * grid_shape[2] +
               occ_idx[:, 1] * grid_shape[2] + occ_idx[:, 2])
  dense[occ_idx_f] = True
  dense = np.reshape(dense, grid_shape)
  return dense


def get_in_out_from_samples(mesh, npoints, sample_factor=10, std=0.01):
  """Get in/out point samples from a given mesh.

  Args:
    mesh: trimesh mesh. Original mesh to sample points from.
    npoints: int, number of points to sample on the mesh surface.
    sample_factor: int, number of samples to pick per surface point.
    std: float, std of samples to generate.
  Returns:
    surface_samples: [npoints, 6], where first 3 dims are xyz, last 3 dims are
    normals (nx, ny, nz).
  """
  surface_point_samples, fid = mesh.sample(int(npoints), return_index=True)
  surface_point_normals = mesh.face_normals[fid]
  offsets = np.random.randn(int(npoints), sample_factor, 1) * std
  near_surface_samples = (surface_point_samples[:, np.newaxis, :] +
                          surface_point_normals[:, np.newaxis, :] * offsets)
  near_surface_samples = np.concatenate([near_surface_samples, offsets],
                                        axis=-1)
  near_surface_samples = near_surface_samples.reshape([-1, 4])
  surface_samples = np.concatenate([surface_point_samples,
                                    surface_point_normals], axis=-1)
  return surface_samples, near_surface_samples


def get_in_out_from_ray(points_from_ray, sample_factor=10, std=0.01):
  """Get sample points from points from ray.

  Args:
    points_from_ray: [npts, 6], where first 3 dims are xyz, last 3 are ray dir.
    sample_factor: int, number of samples to pick per surface point.
    std: float, std of samples to generate.
  Returns:
    near_surface_samples: [npts*sample_factor, 4], where last dimension is
    distance to surface point.
  """
  surface_point_samples = points_from_ray[:, :3]
  surface_point_normals = points_from_ray[:, 3:]
  # make sure normals are normalized to unit length
  n = surface_point_normals
  surface_point_normals = n / (np.linalg.norm(n, axis=1, keepdims=True)+1e-8)
  npoints = points_from_ray.shape[0]
  offsets = np.random.randn(npoints, sample_factor, 1) * std
  near_surface_samples = (surface_point_samples[:, np.newaxis, :] +
                          surface_point_normals[:, np.newaxis, :] * offsets)
  near_surface_samples = np.concatenate([near_surface_samples, offsets],
                                        axis=-1)
  near_surface_samples = near_surface_samples.reshape([-1, 4])
  return near_surface_samples


def intrinsics_from_matrix(int_mat):
  return (int_mat[0, 0], int_mat[1, 1], int_mat[0, 2], int_mat[1, 2])


def encode_decoder_one_scene(near_surface_samples, ckpt_dir, part_size,
                             overlap, indep_pt_loss,
                             xmin=np.zeros(3),
                             xmax=np.ones(3),
                             res_per_part=16, npts=4096, init_std=1e-4,
                             learning_rate=1e-3, steps=10000, nows=False,
                             verbose=False):
  """Wrapper function for encoding and decoding one scene.

  Args:
    near_surface_samples: [npts*sample_factor, 4], where last dimension is
    distance to surface point.
    ckpt_dir: str, path to checkpoint directory to use.
    part_size: float, size of each part to use when autodecoding.
    overlap: bool, whether to use overlapping encoding.
    indep_pt_loss: bool, whether to use independent point loss in optimization.
    xmin: np.array of len 3, lower coordinates of the domain bounds.
    xmax: np.array of len 3, upper coordinates of the domain bounds.
    res_per_part: int, resolution of output evaluation per part.
    npts: int, number of points to use per step when doing gradient descent.
    init_std: float, std to use when initializing seed.
    learning_rate: float, learning rate for doing gradient descent.
    steps: int, number of optimization steps to take.
    nows: bool, no warmstarting from checkpoint. use random codebook.
    verbose: bool, verbose mode.
  Returns:
    v: float32 np.array, vertices of reconstructed mesh.
    f: int32 np.array, faces of reconstructed mesh.
    feat_grid: float32 np.array, feature grid.
    mask: bool np.array, mask of occupied cells.
  """
  ckpt = tf.train.latest_checkpoint(ckpt_dir)
  np.random.shuffle(near_surface_samples)
  param_file = os.path.join(ckpt_dir, 'params.txt')
  params = evaluator.parse_param_file(param_file)

  _, occ_idx, grid_shape = pt.np_get_occupied_idx(
      near_surface_samples[:100000, :3],
      xmin=xmin-0.5*part_size, xmax=xmax+0.5*part_size, crop_size=part_size,
      ntarget=1, overlap=overlap, normalize_crops=False, return_shape=True)
  npts = min(npts, near_surface_samples.shape[0])
  if verbose: print('LIG shape: {}'.format(grid_shape))
  if verbose: print('Optimizing latent codes in LIG...')
  goptim = LIGOptimizer(
      ckpt, origin=xmin, grid_shape=grid_shape, part_size=part_size,
      occ_idx=occ_idx, indep_pt_loss=indep_pt_loss, overlap=overlap,
      alpha_lat=params['alpha_lat'], npts=npts, init_std=init_std,
      learning_rate=learning_rate, var_prefix='', nows=nows)

  goptim.optimize_feat_grid(near_surface_samples[:, :3],
                            near_surface_samples[:, 3:], steps=steps)

  mask = occupancy_sparse_to_dense(occ_idx, grid_shape)

  # evaluate mesh for the current crop
  if verbose: print('Extracting mesh from LIG...')
  svg = evaluator.SparseLIGEvaluator(
      ckpt, num_filters=params['refiner_nf'],
      codelen=params['codelen'], origin=xmin,
      grid_shape=grid_shape, part_size=part_size,
      overlap=overlap, scope='')
  feat_grid = goptim.feature_grid[0]
  out_grid = svg.evaluate_feature_grid(feat_grid,
                                       mask=mask,
                                       res_per_part=res_per_part)

  v, f, _, _ = measure.marching_cubes_lewiner(out_grid, 0)
  v *= (part_size / float(res_per_part) *
        float(out_grid.shape[0]) / (float(out_grid.shape[0])-1))
  v += xmin

  return v, f, feat_grid, mask
