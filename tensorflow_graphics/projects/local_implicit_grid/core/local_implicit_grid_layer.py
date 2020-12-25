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
"""Local Implicit Grid layer implemented in Tensorflow.
"""

import tensorflow.compat.v1 as tf

from tensorflow_graphics.projects.local_implicit_grid.core import implicit_nets
from tensorflow_graphics.projects.local_implicit_grid.core import regular_grid_interpolation

layers = tf.keras.layers


class LocalImplicitGrid(layers.Layer):
  """Local Implicit Grid layer.
  """

  def __init__(self,
               size=(32, 32, 32),
               in_features=16,
               out_features=1,
               x_location_max=1,
               num_filters=128,
               net_type="imnet",
               method="linear",
               interp=True,
               min_grid_value=(0, 0, 0),
               max_grid_value=(1, 1, 1),
               name="lvoxgrid"):
    """Initialization function.

    Args:
      size: list or tuple of ints, grid dimension in each dimension.
      in_features: int, number of input channels.
      out_features: int, number of output channels.
      x_location_max: float, relative coordinate range for one voxel.
      num_filters: int, number of filters for refiner.
      net_type: str, one of occnet/deepsdf.
      method: str, one of linear/nn.
      interp: bool, interp final results across neighbors (only in linear mode).
      min_grid_value: tuple, lower bound of query points.
      max_grid_value: tuple, upper bound of query points.
      name: str, name of the layer.
    """
    super(LocalImplicitGrid, self).__init__(name=name)
    # Print warning if x_location_max and method do not match
    if not ((x_location_max == 1 and method == "linear") or
            (x_location_max == 2 and method == "nn")):
      raise ValueError("Bad combination of x_location_max and method.")
    self.cin = in_features
    self.cout = out_features
    self.dim = len(size)
    self.x_location_max = x_location_max
    self.interp = interp
    self.min_grid_value = min_grid_value
    self.max_grid_value = max_grid_value
    self.num_filters = num_filters
    if self.dim not in [2, 3]:
      raise ValueError("`size` must be tuple or list of len 2 or 3.")
    if net_type == "imnet":
      self.net = implicit_nets.ImNet(
          in_features=in_features, num_filters=num_filters)
    elif net_type == "deepsdf":
      self.net = implicit_nets.DeepSDF(
          in_features=in_features, num_filters=num_filters)
    else:
      raise NotImplementedError
    if method not in ["linear", "nn"]:
      raise ValueError("`method` must be `linear` or `nn`.")
    self.method = method
    self.size_tensor = tf.constant(size)
    self.size = size
    if size[0] == size[1] == size[2] == 1:
      self.cubesize = None
    else:
      self.cubesize = tf.constant([1/(r-1) for r in size], dtype=tf.float32)

  def call(self, grid, pts, training=False):
    """Forward method for Learnable Voxel Grid.

    Args:
      grid: `[batch_size, *self.size, in_features]` tensor, input feature grid.
      pts: `[batch_size, num_points, dim]` tensor, coordinates of points that
      are within the range (0, 1).
      training: bool, flag indicating training phase.
    Returns:
      outputs: `[batch_size, num_points, out_features]` tensor, continuous
      function field value at locations specified at pts.
    Raises:
      RuntimeError: dimensions of grid does not match that of self.
    """
    # assert that dimensions match
    grid = tf.ensure_shape(grid, (None, self.size[0], self.size[1],
                                  self.size[2], self.cin))
    pts = tf.ensure_shape(pts, (None, None, self.dim))

    lat, weights, xloc = self._interp(grid, pts)
    outputs = self._eval_net(lat, weights, xloc, training=training)

    return outputs

  def _interp(self, grid, pts):
    """Interpolation function to get local latent code, weights & relative loc.

    Args:
      grid: `[batch_size, *self.size, in_features]` tensor, input feature grid.
      pts: `[batch_size, num_points, dim]` tensor, coordinates of points that
      are within the range (0, 1).
    Returns:
      lat: `[batch_size, num_points, 2**dim, in_features]` tensor, neighbor
      latent codes for each input point.
      weights: `[batch_size, num_points, 2**dim]` tensor, bi/tri-linear
      interpolation weights for each neighbor.
      xloc: `[batch_size, num_points, 2**dim, dim]`tensor, relative coordinates.
    """
    lat, weights, xloc = regular_grid_interpolation.get_interp_coefficients(
        grid,
        pts,
        min_grid_value=self.min_grid_value,
        max_grid_value=self.max_grid_value)
    xloc *= self.x_location_max

    return lat, weights, xloc

  def _eval_net(self, lat, weights, xloc, training=False):
    """Evaluate function values by querying shared dense network.

    Args:
      lat: `[batch_size, num_points, 2**dim, in_features]` tensor, neighbor
      latent codes for each input point.
      weights: `[batch_size, num_points, 2**dim]` tensor, bi/tri-linear
      interpolation weights for each neighbor.
      xloc: `[batch_size, num_points, 2**dim, dim]`tensor, relative coordinates.
      training: bool, flag indicating training phase.
    Returns:
      values: `[batch_size, num_point, out_features]` tensor, query values.
    """
    nb, np, nn, nc = lat.get_shape().as_list()
    nd = self.dim
    if self.method == "linear":
      inputs = tf.concat([xloc, lat], axis=-1)
      # `[batch_size, num_points, 2**dim, dim+in_features]`
      inputs = tf.reshape(inputs, [-1, nc+nd])
      values = self.net(inputs, training=training)
      values = tf.reshape(values, [nb, np, nn, self.cout])
      # `[batch_size, num_points, 2**dim, out_features]`
      if self.interp:
        values = tf.reduce_sum(tf.expand_dims(weights, axis=-1)*values, axis=2)
        # `[batch_size, num_points out_features]`
      else:
        values = (values, weights)
    else:  # nearest neighbor
      nid = tf.cast(tf.argmax(weights, axis=-1), tf.int32)
      # [batch_size, num_points]
      bid = tf.broadcast_to(tf.range(nb, dtype=tf.int32)[:, tf.newaxis],
                            [nb, np])
      pid = tf.broadcast_to(tf.range(np, dtype=tf.int32)[tf.newaxis, :],
                            [nb, np])
      gather_id = tf.stack((bid, pid, nid), axis=-1)
      lat_ = tf.gather_nd(lat, gather_id)  # [batch_size, num_points, in_feat]
      xloc_ = tf.gather_nd(xloc, gather_id)  # [batch_size, num_points, dim]
      inputs = tf.concat([xloc_, lat_], axis=-1)
      inputs = tf.reshape(inputs, [-1, nc+nd])
      values = self.net(inputs, training=training)
      values = tf.reshape(values, [nb, np, self.cout])
      # `[batch_size, num_points, out_features]`

    return values
