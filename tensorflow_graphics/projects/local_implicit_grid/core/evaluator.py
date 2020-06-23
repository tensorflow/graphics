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
"""Utility modules for evaluating model from checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.io import gfile

from tensorflow_graphics.projects.local_implicit_grid.core import implicit_nets as im
from tensorflow_graphics.projects.local_implicit_grid.core import local_implicit_grid_layer as lig
from tensorflow_graphics.projects.local_implicit_grid.core import model_g2g as g2g
from tensorflow_graphics.projects.local_implicit_grid.core import model_g2v as g2v


tf.logging.set_verbosity(tf.logging.ERROR)


def parse_param_file(param_file):
  """Parse parameter file for parameters."""
  with gfile.GFile(param_file, 'r') as fh:
    lines = fh.readlines()
  d = {}
  for l in lines:
    l = l.rstrip('\n')
    splits = l.split(':')
    key = splits[0]
    val_ = splits[1].strip()
    if not val_:
      val = ''
    else:
      try:
        val = ast.literal_eval(val_)
      except (ValueError, SyntaxError):
        val = str(val_)
    d[key] = val
  return d


class RefinerEvaluator(object):
  """Load pretrained refiner and evaluate for a given code.
  """

  def __init__(self, ckpt, codelen, dim=3, out_features=1, num_filters=128,
               point_batch=20000):
    self.ckpt = ckpt
    self.codelen = codelen
    self.dim = dim
    self.out_features = out_features
    self.num_filters = num_filters
    self.point_batch = point_batch
    self.graph = tf.Graph()
    self._init_graph()
    self.global_step_ = self.global_step.eval(session=self.sess)

  def _init_graph(self):
    """Initialize computation graph for tensorflow.
    """
    with self.graph.as_default():
      self.refiner = im.ImNet(dim=self.dim,
                              in_features=self.codelen,
                              out_features=self.out_features,
                              num_filters=self.num_filters)
      self.global_step = tf.get_variable('global_step', shape=[],
                                         dtype=tf.int64)

      self.pts_ph = tf.placeholder(tf.float32, shape=[self.point_batch, 3])
      self.lat_ph = tf.placeholder(tf.float32, shape=[self.codelen])

      lat = tf.broadcast_to(self.lat_ph[tf.newaxis],
                            [self.point_batch, self.codelen])
      code = tf.concat((self.pts_ph, lat), axis=-1)  # [pb, 3+c]

      vals = self.refiner(code, training=False)  # [pb, 1]
      self.vals = tf.squeeze(vals, axis=1)  # [pb]
      self.saver = tf.train.Saver()
      self.sess = tf.Session()
      self.saver.restore(self.sess, self.ckpt)

  def _get_grid_points(self, xmin, xmax, res):
    x = np.linspace(xmin, xmax, res)
    xyz = np.meshgrid(*tuple([x] * self.dim), indexing='ij')
    xyz = np.stack(xyz, axis=-1)
    xyz = xyz.reshape([-1, self.dim])
    return xyz

  def eval_points(self, lat, points):
    """Evaluate network at locations specified by points.

    Args:
      lat: [self.codelen,] np array, latent code.
      points: [#v, self.dim] np array, point locations to evaluate.
    Returns:
      all_vals: [#v] np array, function values at locations.
    """
    npt = points.shape[0]
    npb = int(np.ceil(float(npt)/self.point_batch))
    all_vals = np.zeros([npt], dtype=np.float32)

    for idx in range(npb):
      sid = int(idx * self.point_batch)
      eid = int(min(npt, sid+self.point_batch))
      pts = points[sid:eid]
      pad_w = self.point_batch - (eid - sid)
      pts = np.pad(pts, ((0, pad_w), (0, 0)), mode='constant')
      with self.graph.as_default():
        val = self.sess.run(self.vals, feed_dict={self.pts_ph: pts,
                                                  self.lat_ph: lat})
      all_vals[sid:eid] = val[:(eid-sid)]
    return all_vals

  def eval_grid(self, lat, xmin=-1.0, xmax=1.0, res=64):
    """Evaluate network on a grid.

    Args:
      lat: [self.codelen,] np array, latent code.
      xmin: float, minimum coordinate value for grid.
      xmax: float, maximum coordinate value for grid.
      res: int, resolution (per dimension) of grid.
    Returns:
      grid_val: [res, res, res] np.float32 array, grid of values from query.
    """
    grid_points = self._get_grid_points(xmin=xmin, xmax=xmax, res=res)
    point_val = self.eval_points(lat, grid_points)
    grid_val = point_val.reshape([res, res, res])
    return grid_val


class EncoderEvaluator(object):
  """Load pretrained grid encoder and evaluate single crops."""

  def __init__(self,
               ckpt,
               in_grid_res=32,
               encoder_nf=32,
               codelen=32,
               grid_batch=128):
    """Initialization function.

    Args:
      ckpt: str, path to checkpoint.
      in_grid_res: int, resolution of grid to feed to encoder.
      encoder_nf: int, number of base filters for encoder.
      codelen: int, length of output latent code.
      grid_batch: int, batch size of cut-out grid to evaluate at a time.
    """
    self.ckpt = ckpt
    self.codelen = codelen
    self.grid_batch = grid_batch
    self.in_grid_res = in_grid_res
    self.encoder_nf = encoder_nf
    self.graph = tf.Graph()
    self._init_graph()  # creates self.sess

  def _init_graph(self):
    """Initialize computation graph for tensorflow.
    """
    with self.graph.as_default():
      self.encoder = g2v.GridEncoder(in_grid_res=self.in_grid_res,
                                     num_filters=self.encoder_nf,
                                     codelen=self.codelen,
                                     name='g2v')
      self.grid_ph = tf.placeholder(
          tf.float32,
          shape=[None, self.in_grid_res, self.in_grid_res, self.in_grid_res, 1])

      self.lats = self.encoder(self.grid_ph, training=False)  # [gb, codelen]
      self.saver = tf.train.Saver()
      self.sess = tf.Session()
      self.saver.restore(self.sess, self.ckpt)

  def eval_grid(self, grid):
    """Strided evaluation of full grid into feature grid.

    Args:
      grid: [batch, gres, gres, gres, 1] input feature grid.
    Returns:
      codes: [batch, codelen] output feature gird.
    """
    # initialize output feature grid
    niters = int(np.ceil(grid.shape[0] / self.grid_batch))
    codes = []
    for idx in range(niters):
      sid = idx * self.grid_batch
      eid = min(sid+self.grid_batch, grid.shape[0])

      c = self.sess.run(self.lats,
                        feed_dict={self.grid_ph: grid[sid:eid]})
      codes.append(c)
    codes = np.concatenate(codes, axis=0)
    return codes.astype(np.float32)


class FullGridEncoderEvaluator(object):
  """Load pretrained grid encoder and evaluate a full input grid.

  Performs windowed encoding and outputs an encoded feature grid.
  """

  def __init__(self,
               ckpt,
               in_grid_res=32,
               num_filters=32,
               codelen=128,
               grid_batch=128,
               gres=256,
               overlap=True):
    """Initialization function.

    Args:
      ckpt: str, path to checkpoint.
      in_grid_res: int, resolution of grid to feed to encoder.
      num_filters: int, number of base filters for encoder.
      codelen: int, length of output latent code.
      grid_batch: int, batch size of cut-out grid to evaluate at a time.
      gres: int, resolution of the full grid.
      overlap: bool, whether to do overlapping or non-overlapping cutout
        evaluations.
    """
    self.ckpt = ckpt
    self.codelen = codelen
    self.grid_batch = grid_batch
    self.in_grid_res = in_grid_res
    self.gres = gres
    self.num_filters = num_filters
    self.graph = tf.Graph()
    self._init_graph()
    self.global_step_ = self.global_step.eval(session=self.sess)
    if overlap:
      ijk = np.arange(0, gres-int(in_grid_res/2), int(in_grid_res/2))
      self.out_grid_res = ijk.shape[0]
    else:
      ijk = np.arange(0, gres, in_grid_res)
      self.out_grid_res = ijk.shape[0]
    self.ijk = np.meshgrid(ijk, ijk, ijk, indexing='ij')
    self.ijk = np.stack(self.ijk, axis=-1).reshape([-1, 3])

  def _init_graph(self):
    """Initialize computation graph for tensorflow."""
    with self.graph.as_default():
      self.encoder = g2v.GridEncoder(
          in_grid_res=self.in_grid_res,
          num_filters=self.num_filters,
          codelen=self.codelen,
          name='g2v')
      self.global_step = tf.get_variable(
          'global_step', shape=[], dtype=tf.int64)
      self.grid_ph = tf.placeholder(
          tf.float32, shape=[self.gres, self.gres, self.gres])
      self.start_ph = tf.placeholder(tf.int32, shape=[self.grid_batch, 3])
      self.ingrid = self._batch_slice(self.grid_ph, self.start_ph,
                                      self.in_grid_res, self.grid_batch)
      self.ingrid = self.ingrid[..., tf.newaxis]
      self.lats = self.encoder(self.ingrid, training=False)  # [gb, codelen]
      self.saver = tf.train.Saver()
      self.sess = tf.Session()
      self.saver.restore(self.sess, self.ckpt)

  def _batch_slice(self, ary, start_ijk, w, batch_size):
    """Batched slicing of original grid.

    Args:
      ary: tensor, rank = 3.
      start_ijk: [batch_size, 3] tensor, starting index.
      w: width of cube to extract.
      batch_size: int, batch size.

    Returns:
      batched_slices: [batch_size, w, w, w] tensor, batched slices of ary.
    """
    batch_size = start_ijk.shape[0]
    ijk = tf.range(w, dtype=tf.int32)
    slice_idx = tf.meshgrid(ijk, ijk, ijk, indexing='ij')
    slice_idx = tf.stack(
        slice_idx, axis=-1)  # [in_grid_res, in_grid_res, in_grid_res, 3]
    slice_idx = tf.broadcast_to(slice_idx[tf.newaxis], [batch_size, w, w, w, 3])
    offset = tf.broadcast_to(
        start_ijk[:, tf.newaxis, tf.newaxis, tf.newaxis, :],
        [batch_size, w, w, w, 3])
    slice_idx += offset
    # [batch_size, in_grid_res, in_grid_res, in_grid_res, 3]
    batched_slices = tf.gather_nd(ary, slice_idx)
    # [batch_size, in_grid_res, in_grid_res, in_grid_res]
    return batched_slices

  def eval_grid(self, grid):
    """Strided evaluation of full grid into feature grid.

    Args:
      grid: [gres, gres, gres] input feature grid.

    Returns:
      ogrid: [out_grid_res, out_grid_res, out_grid_res, codelen] output feature
      gird.
    """
    # initialize output feature grid
    ogrid = np.zeros([self.ijk.shape[0], self.codelen])
    niters = np.ceil(self.ijk.shape[0] / self.grid_batch).astype(np.int)
    for idx in range(niters):
      sid = idx * self.grid_batch
      eid = min(sid + self.grid_batch, self.ijk.shape[0])
      start_ijk = self.ijk[sid:eid]
      # pad if last iteration does not have a full batch
      pad_w = self.grid_batch - start_ijk.shape[0]
      start_ijk = np.pad(start_ijk, ((0, pad_w), (0, 0)), mode='constant')
      lats = self.sess.run(
          self.lats, feed_dict={
              self.grid_ph: grid,
              self.start_ph: start_ijk
          })
      ogrid[sid:eid] = lats[:eid - sid]
    ogrid = ogrid.reshape(
        [self.out_grid_res, self.out_grid_res, self.out_grid_res, self.codelen])
    return ogrid.astype(np.float32)


class LIGEvaluator(object):
  """Load pretrained grid refiner and evaluate a feature grid.
  """

  def __init__(self,
               ckpt,
               size=(15, 15, 15),
               in_features=32,
               out_features=1,
               x_location_max=1,
               num_filters=32,
               min_grid_value=(0., 0., 0.),
               max_grid_value=(1., 1., 1.),
               net_type='imnet',
               method='linear',
               point_batch=20000,
               scope=''):
    """Initialization function.

    Args:
      ckpt: str, path to checkpoint.
      size: list or tuple of ints, grid dimension in each dimension.
      in_features: int, number of input channels.
      out_features: int, number of output channels.
      x_location_max: float, relative coordinate range for one voxel.
      num_filters: int, number of filters for refiner.
      min_grid_value: tuple, lower bound of query points.
      max_grid_value: tuple, upper bound of query points.
      net_type: str, one of occnet/deepsdf.
      method: str, one of linear/nn.
      point_batch: int, pseudo batch size for evaluating points.
      scope: str, scope of imnet layer.
    """
    self.dim = 3  # hardcode for dim = 3
    self.ckpt = ckpt
    self.size = size
    self.x_location_max = x_location_max
    self.num_filters = num_filters
    self.in_features = in_features
    self.out_features = out_features
    self.net_type = net_type
    self.method = method
    self.point_batch = point_batch
    self.scope = scope
    self.min_grid_value = min_grid_value
    self.max_grid_value = max_grid_value
    self.graph = tf.Graph()
    self._init_graph()

  def _init_graph(self):
    """Initialize computation graph for tensorflow.
    """
    with self.graph.as_default():
      self.lig = lig.LocalImplicitGrid(size=self.size,
                                       in_features=self.in_features,
                                       out_features=self.out_features,
                                       num_filters=self.num_filters,
                                       net_type=self.net_type,
                                       method=self.method,
                                       x_location_max=self.x_location_max,
                                       min_grid_value=self.min_grid_value,
                                       max_grid_value=self.max_grid_value,
                                       name='lig')

      self.pts_ph = tf.placeholder(tf.float32, shape=[self.point_batch, 3])
      self.latgrid_ph = tf.placeholder(tf.float32,
                                       shape=[self.size[0],
                                              self.size[1],
                                              self.size[2],
                                              self.in_features])
      self.latgrid = self.latgrid_ph[tf.newaxis]
      self.points = self.pts_ph[tf.newaxis]
      vals = self.lig(self.latgrid, self.points, training=False)  # [1,npts,1]
      self.vals = tf.squeeze(vals, axis=[0, 2])  # [npts]
      self.map_dict = self._get_var_mapping(model=self.lig)
      self.saver = tf.train.Saver(self.map_dict)
      self.sess = tf.Session()
      self.saver.restore(self.sess, self.ckpt)

  def _get_grid_points(self, xmin, xmax, res):
    x = np.linspace(xmin, xmax, res)
    xyz = np.meshgrid(*tuple([x] * self.dim), indexing='ij')
    xyz = np.stack(xyz, axis=-1)
    xyz = xyz.reshape([-1, self.dim])
    return xyz

  def eval_points(self, latgrid, points):
    """Evaluate network at locations specified by points.

    Args:
      latgrid: [size0, size1, size2, self.codelen] np array, latent code.
      points: [#v, self.dim] np array, point locations to evaluate.
    Returns:
      all_vals: [#v] np array, function values at locations.
    """
    npt = points.shape[0]
    npb = int(np.ceil(float(npt)/self.point_batch))
    all_vals = np.zeros([npt], dtype=np.float32)

    for idx in range(npb):
      sid = int(idx * self.point_batch)
      eid = int(min(npt, sid+self.point_batch))
      pts = points[sid:eid]
      pad_w = self.point_batch - (eid - sid)
      if pts.shape[0] < self.point_batch:
        pts_pad = np.tile(pts[0:1], (pad_w, 1))
        # repeat the first point in the batch
        pts = np.concatenate([pts, pts_pad], axis=0)
      with self.graph.as_default():
        val = self.sess.run(self.vals, feed_dict={self.pts_ph: pts,
                                                  self.latgrid_ph: latgrid})
      all_vals[sid:eid] = val[:(eid-sid)]
    return all_vals

  def eval_grid(self, latgrid, xmin=0.0, xmax=1.0, res=128):
    """Evaluate network on a grid.

    Args:
      latgrid: [size0, size1, size2, self.codelen] np array, latent code.
      xmin: float, minimum coordinate value for grid.
      xmax: float, maximum coordinate value for grid.
      res: int, resolution (per dimension) of grid.
    Returns:
      grid_val: [res, res, res] np.float32 array, grid of values from query.
    """
    grid_points = self._get_grid_points(xmin=xmin, xmax=xmax, res=res)
    point_val = self.eval_points(latgrid, grid_points)
    grid_val = point_val.reshape([res, res, res])
    return grid_val

  def _get_var_mapping(self, model):
    vars_ = model.trainable_variables
    varnames = [v.name for v in vars_]  # .split(':')[0]
    varnames = [self.scope+v.replace('lig/', '').strip(':0') for v in varnames]
    map_dict = dict(zip(varnames, vars_))
    return map_dict


class UNetEvaluator(object):
  """Load pretrained UNet for generating feature grid for coarse voxel inputs."""

  def __init__(self,
               ckpt,
               in_grid_res,
               out_grid_res,
               num_filters,
               max_filters,
               out_features,
               sph_norm=0.):
    self.ckpt = ckpt
    self.in_grid_res = in_grid_res
    self.out_grid_res = out_grid_res
    self.num_filters = num_filters
    self.max_filters = max_filters
    self.out_features = out_features
    self.sph_norm = sph_norm
    self.graph = tf.Graph()
    self._init_graph()

  def _init_graph(self):
    """Initialize computation graph for tensorflow."""
    with self.graph.as_default():
      self.unet = g2g.UNet3D(in_grid_res=self.in_grid_res,
                             out_grid_res=self.out_grid_res,
                             num_filters=self.num_filters,
                             max_filters=self.max_filters,
                             out_features=self.out_features)
      self.input_grid_ph = tf.placeholder(
          tf.float32,
          [None, None, None])
      self.input_grid = self.input_grid_ph[tf.newaxis, ..., tf.newaxis]
      self.feat_grid = self.unet(self.input_grid)
      self.saver = tf.train.Saver()
      self.sess = tf.Session()
      self.saver.restore(self.sess, self.ckpt)

  def eval_grid(self, input_grid):
    """Evaluate input grid (no batching).

    Args:
      input_grid: [in_grid_res, in_grid_res, in_grid_res] tensor.
    Returns:
      [out_grid_res, out_grid_res, out_grid_res, out_features]
    """
    with self.graph.as_default():
      feat_grid = self.sess.run(self.feat_grid,
                                feed_dict={self.input_grid_ph: input_grid})
    feat_grid = feat_grid[0]
    if self.sph_norm > 0:
      feat_grid = (feat_grid /
                   np.linalg.norm(feat_grid, axis=-1, keepdims=True) *
                   self.sph_norm)
    return feat_grid


class SparseLIGEvaluator(object):
  """Evaluate sparse encoded feature grids."""

  def __init__(self, ckpt, num_filters, codelen, origin, grid_shape,
               part_size, overlap=True, scope=''):
    self.scope = scope
    self.overlap = overlap
    self.ckpt = ckpt
    self.num_filters = num_filters
    self.codelen = codelen
    if overlap:
      self.res = (np.array(grid_shape) - 1) / 2.0
    else:
      self.res = np.array(grid_shape) - 1
    self.res = self.res.astype(np.int32)
    self.xmin = np.array(origin)
    self.xmax = self.xmin + self.res * part_size
    self.part_size = part_size

    self.lvg = LIGEvaluator(ckpt=ckpt,
                            size=grid_shape,
                            in_features=codelen,
                            out_features=1,
                            x_location_max=2-float(overlap),
                            num_filters=num_filters,
                            min_grid_value=self.xmin,
                            max_grid_value=self.xmax,
                            net_type='imnet',
                            method='linear' if overlap else 'nn',
                            scope=scope)

  def evaluate_feature_grid(self, feature_grid, mask, res_per_part=4,
                            conservative=False):
    """Evaluate feature grid.

    Args:
      feature_grid: [*grid_size, codelen] np.array, feature grid to evaluate.
      mask: [*grid_size] bool np.array, mask for feature locations.
      res_per_part: int, resolution of output evaluation per part.
      conservative: bool, whether to do conservative evaluations.
      If true, evalutes a cell if either neighbor is masked. Else, evaluates a
      cell if all neighbors are masked.
    Returns:
      output grid.
    """
    # setup grid
    eps = 1e-6
    s = self.res
    l = [np.linspace(self.xmin[i]+eps, self.xmax[i]-eps, res_per_part*s[i])
         for i in range(3)]
    xyz = np.stack(np.meshgrid(l[0], l[1], l[2],
                               indexing='ij'), axis=-1).reshape(-1, 3)
    output_grid = np.ones([res_per_part*s[0],
                           res_per_part*s[1],
                           res_per_part*s[2]], dtype=np.float32).reshape(-1)
    mask = mask.astype(np.bool)
    if self.overlap:
      mask = np.stack([mask[:-1, :-1, :-1],
                       mask[:-1, :-1, 1:],
                       mask[:-1, 1:, :-1],
                       mask[:-1, 1:, 1:],
                       mask[1:, :-1, :-1],
                       mask[1:, :-1, 1:],
                       mask[1:, 1:, :-1],
                       mask[1:, 1:, 1:]], axis=-1)
      if conservative:
        mask = np.any(mask, axis=-1)
      else:
        mask = np.all(mask, axis=-1)

    g = np.stack(np.meshgrid(np.arange(mask.shape[0]),
                             np.arange(mask.shape[1]),
                             np.arange(mask.shape[2]),
                             indexing='ij'), axis=-1).reshape(-1, 3)
    g = g[:, 0]*(mask.shape[1]*mask.shape[2]) + g[:, 1]*mask.shape[2] + g[:, 2]
    g_valid = g[mask.ravel()]

    if self.overlap:
      ijk = np.floor((xyz - self.xmin) / self.part_size * 2).astype(np.int32)
    else:
      ijk = np.floor((xyz - self.xmin +
                      0.5 * self.part_size) / self.part_size).astype(np.int32)
    ijk_idx = (ijk[:, 0]*(mask.shape[1] * mask.shape[2]) +
               ijk[:, 1]*mask.shape[2] + ijk[:, 2])
    pt_mask = np.isin(ijk_idx, g_valid)

    output_grid[pt_mask] = self.lvg.eval_points(feature_grid, xyz[pt_mask])
    output_grid = output_grid.reshape(res_per_part*s[0],  # pylint: disable=too-many-function-args
                                      res_per_part*s[1],
                                      res_per_part*s[2])
    return output_grid
