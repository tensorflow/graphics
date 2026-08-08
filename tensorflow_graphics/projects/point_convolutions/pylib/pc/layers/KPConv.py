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
"""Class for kernel point cloud convolutions"""

import tensorflow as tf
from pylib.pc.utils import _flatten_features

from pylib.pc import PointCloud
from pylib.pc import Grid
from pylib.pc import Neighborhood
from pylib.pc import KDEMode

from pylib.pc.custom_ops import basis_proj
from pylib.pc.layers.utils import _format_output, spherical_kernel_points, \
    random_rotation


def _linear_weighting(values, sigma):
  """ Linear kernel weights for KP Conv.

  Args:
    values: A `float` `Tensor` of shape `[K, M]`, the normalized distances to
      the kernel points, in `[0,1]`.
    sigma: A `float`, the normalized influence distance of the kernel points.

  Returns:
    A `float` `Tensor` of shape `[K, M]`.

  """
  return tf.nn.relu(1 - values / sigma)


def _gaussian_weighting(values, sigma):
  """ Gaussian kernel weights for KP Conv.

  Args:
    values: A `float` `Tensor` of shape `[K, M]`, the normalized distances to
      the kernel points, in `[0,1]`.
    sigma: A `float`, the normalized influence distance of the kernel points.

  Returns:
    A `float` `Tensor` of shape `[K, M]`.

  """
  sigma = sigma / 3
  return tf.exp(-(values / sigma)**2)

kernel_interpolation = {'linear': _linear_weighting,
                        'gaussian': _gaussian_weighting}


class KPConv(tf.Module):
  """ A Kernel Point Convolution for 3D point clouds.

  Based on the paper [KPConv: Flexible and Deformable Convolution for Point
  Clouds. Thomas et al., 2019](https://arxiv.org/abs/1904.08889).

  Note: To use this layer for point clouds with arbitrary dimension `D`,
    pass initial kernel points of dimension `D` using `custom_kernel_points`.

  Args:
    num_features_in: An `int`, `C_in`, the number of features per input point.
    num_features_out: An `int`, `C_out`, the number of features to compute.
    num_kernel_points: An ìnt`, the number of points for representing the
      kernel, default is `15`. (optional)
    num_dims: An `int`, the dimensionality of the point cloud. Defaults to `3`.
      (optional)
    deformable: A 'bool', indicating whether to use rigid or deformable kernel
      points, default is `False`. (optional)
    kp_interpolation: A `string`, either `'linear'`(default) or `'gaussian'`.
      (optional)
    custom_kernel_points: A `float` `Tensor` of shape `[K, D]`, to pass custom
      kernel points. (optional)
    initializer_weights: A `tf.initializer` for the weights,
      default `TruncatedNormal`. (optional)

  Raises:
    ValueError, if no custom kernel points are passed for dimension not equal
      to 3.

  """

  def __init__(self,
               num_features_in,
               num_features_out,
               num_kernel_points=15,
               num_dims=3,
               deformable=False,
               kp_interpolation='linear',
               custom_kernel_points=None,
               initializer_weights=None,
               name=None):

    super().__init__(name=name)

    self._num_features_in = num_features_in
    self._num_features_out = num_features_out
    self._num_kernel_points = num_kernel_points
    self._deformable = deformable
    self._weighting = kernel_interpolation[kp_interpolation]
    self._num_dims = num_dims
    if name is None:
      self._name = 'KPConv'
    else:
      self._name = name

    if num_dims != 3 and custom_kernel_points is None:
      raise ValueError(
        "For dimension not 3 custom kernel points must be provided!")

    # initialize kernel points
    if custom_kernel_points is None:
      self._kernel_points = spherical_kernel_points(num_kernel_points,
                                                    rotate=True)
    else:
      self._kernel_points = tf.convert_to_tensor(value=custom_kernel_points,
                                                 dtype=tf.float32)

    # Reposition the points at radius 0.75.
    self._kernel_points = self._kernel_points * 0.75

    # initialize variables
    if initializer_weights is None:
      initializer_weights = tf.initializers.GlorotNormal

    weights_init_obj = initializer_weights()

    if deformable:
      self._kernel_offsets_weights = \
        tf.Variable(
            weights_init_obj(shape=[
                        self._num_kernel_points * self._num_features_in,
                        self._num_kernel_points * self._num_dims],
                        dtype=tf.float32),
            trainable=True,
            name=self._name + "/weights_deformable")
      self._get_offsets = self._kernel_offsets
    else:
      def _zero(*args, **kwargs):
        """ Replaces `_get_offsets` with zeros for rigid KPConv.
        """
        return tf.constant(0.0, dtype=tf.float32)
      self._get_offsets = _zero

    self._weights = \
        tf.Variable(
            weights_init_obj(shape=[
                        self._num_kernel_points * self._num_features_in,
                        self._num_features_out],
                        dtype=tf.float32),
            trainable=True,
            name=self._name + "/conv_weights")

  def _kp_conv(self,
               kernel_input,
               neighborhood,
               features):
    """ Method to compute a kernel point convolution using linear interpolation
    of the kernel weights.

    Note: In the following
      `D` is the dimensionality of the points cloud (=3)
      `M` is the number of neighbor pairs
      'C1`is the number of input features
      `C2` is the number of output features
      `N1' is the number of input points
      `N2' is the number of ouput points

    Args:
      kernel_inputs: A `float` `Tensor` of shape `[M, D]`, the input to the
        kernel, i.e. the distances between neighbor pairs.
      neighborhood: A `Neighborhood` instance.
      features: A `float` `Tensor` of shape `[N1, C1]`, the input features.

    Returns:
      A `float` `Tensor` of shape `[N2, C2]`, the output features.

    """
    # neighbor pairs ids
    neighbors = neighborhood._original_neigh_ids
    # kernel weights from distances, shape [M, K]
    kernel_offsets = self._get_offsets(kernel_input, neighborhood, features)
    points_diff = tf.expand_dims(kernel_input, 1) - \
        (tf.expand_dims(self._kernel_points, 0) + kernel_offsets)
    points_dist = tf.linalg.norm(points_diff, axis=2)
    kernel_weights = self._weighting(points_dist, self._sigma)

    # Pad zeros to fullfil requirements of the basis_proj custom op,
    # 8, 16, 32, or 64 basis are allowed.
    if self._num_kernel_points < 8:
        kernel_weights = tf.pad(kernel_weights,
                                [[0, 0], [0, 8 - self._num_kernel_points]])
    elif self._num_kernel_points > 8 and self._num_kernel_points < 16:
        kernel_weights = tf.pad(kernel_weights,
                                [[0, 0], [0, 16 - self._num_kernel_points]])
    elif self._num_kernel_points > 16 and self._num_kernel_points < 32:
        kernel_weights = tf.pad(kernel_weights,
                                [[0, 0], [0, 32 - self._num_kernel_points]])
    elif self._num_kernel_points > 32 and self._num_kernel_points < 64:
        kernel_weights = tf.pad(kernel_weights,
                                [[0, 0], [0, 64 - self._num_kernel_points]])

    # save values for regularization loss computation
    self._cur_point_dist = points_dist
    self._cur_neighbors = neighbors

    # Compute the projection to the samples.
    weighted_features = basis_proj(
        kernel_weights,
        features,
        neighborhood)
    # remove padding
    weighted_features = weighted_features[:, :, 0:self._num_kernel_points]

    #Compute convolution - hidden layer to output (linear)
    convolution_result = tf.matmul(
        tf.reshape(weighted_features,
                   [-1, self._num_features_in * self._num_kernel_points]),
        self._weights)

    return convolution_result

  def __call__(self,
               features,
               point_cloud_in: PointCloud,
               point_cloud_out: PointCloud,
               conv_radius,
               neighborhood=None,
               kernel_influence_dist=None,
               return_sorted=False,
               return_padded=False,
               name=None):
    """ Computes the Kernel Point Convolution between two point clouds.

    Note:
      In the following, `A1` to `An` are optional batch dimensions.
      `C_in` is the number of input features.
      `C_out` is the number of output features.

    Args:
      features: A `float` `Tensor` of shape `[N1, C_in]` or
        `[A1, ..., An,V, C_in]`.
      point_cloud_in: A 'PointCloud' instance, on which the features are
        defined.
      point_cloud_out: A `PointCloud` instance, on which the output
        features are defined.
      conv_radius: A `float`, the convolution radius.
      neighborhood: A `Neighborhood` instance, defining the neighborhood
        with centers from `point_cloud_out` and neighbors in `point_cloud_in`.
        If `None` it is computed internally. (optional)
      kernel_influence_dist = A `float`, the influence distance of the kernel
        points. If `None` uses `conv_radius / 2.5`, as suggested in Section 3.3
        of the paper. (optional)
      return_sorted: A `boolean`, if `True` the output tensor is sorted
        according to the batch_ids. (optional)
      return_padded: A `bool`, if 'True' the output tensor is sorted and
        zero padded. (optional)

    Returns:
      A `float` `Tensor` of shape
        `[N2, C_out]`, if `return_padded` is `False`
      or
        `[A1, ..., An, V_out, C_out]`, if `return_padded` is `True`.

    """

    features = tf.cast(tf.convert_to_tensor(value=features),
                       dtype=tf.float32)
    features = _flatten_features(features, point_cloud_in)
    self._num_output_points = point_cloud_out._points.shape[0]

    if kernel_influence_dist is None:
      # normalized
      self._sigma = tf.constant(1.0)
    else:
      self._sigma = tf.convert_to_tensor(
        value=kernel_influence_dist / conv_radius, dtype=tf.float32)
    #Create the radii tensor.
    radii_tensor = tf.cast(tf.repeat([conv_radius], self._num_dims),
                           dtype=tf.float32)

    if neighborhood is None:
      #Compute the grid
      grid = Grid(point_cloud_in, radii_tensor)
      #Compute the neighborhoods
      neigh = Neighborhood(grid, radii_tensor, point_cloud_out)
    else:
      neigh = neighborhood

    #Compute kernel inputs.
    neigh_point_coords = tf.gather(
        point_cloud_in._points, neigh._original_neigh_ids[:, 0])
    center_point_coords = tf.gather(
        point_cloud_out._points, neigh._original_neigh_ids[:, 1])
    points_diff = (neigh_point_coords - center_point_coords) / \
        tf.reshape(radii_tensor, [1, self._num_dims])
    #Compute Monte-Carlo convolution
    convolution_result = self._kp_conv(points_diff, neigh, features)
    return _format_output(convolution_result,
                          point_cloud_out,
                          return_sorted,
                          return_padded)

  def _kernel_offsets(self,
                      kernel_input,
                      neighborhood,
                      features):
    """ Method to compute the kernel offsets for deformable KPConv
    using a rigid KPConv.

    As described in Section 3.2 of [KPConv: Flexible and Deformable Convolution
    for Point Clouds. Thomas et al., 2019](https://arxiv.org/abs/1904.08889).

    Note: In the following
      `D` is the dimensionality of the point cloud (=3)
      `M` is the number of neighbor pairs
      'C1`is the number of input features
      `N1' is the number of input points
      `N2' is the number of ouput points
      `K` is the number of kernel points

    Args:
      kernel_inputs: A `float` `Tensor` of shape `[M, D]`, the input to the
        kernel, i.e. the distances between neighbor pairs.
      neighborhood: A `Neighborhood` instance.
      features: A `float` `Tensor` of shape `[N1, C1]`, the input features.

    Returns:
      A `float` `Tensor` of shape `[K, M, D]`, the offsets.

    """
    # neighbor pairs ids
    neighbors = neighborhood._original_neigh_ids
    # kernel weights from distances, shape [M, K]
    points_diff = tf.expand_dims(kernel_input, 1) - \
        tf.expand_dims(self._kernel_points, 0)
    points_dist = tf.linalg.norm(points_diff, axis=2)
    kernel_weights = self._weighting(points_dist, self._sigma)

    # Pad zeros to fullfil requirements of the basis_proj custom op,
    # 8, 16, or 32 basis are allowed.
    if self._num_kernel_points < 8:
        kernel_weights = tf.pad(kernel_weights,
                                [[0, 0], [0, 8 - self._num_kernel_points]])
    elif self._num_kernel_points > 8 and self._num_kernel_points < 16:
        kernel_weights = tf.pad(kernel_weights,
                                [[0, 0], [0, 16 - self._num_kernel_points]])
    elif self._num_kernel_points > 16 and self._num_kernel_points < 32:
        kernel_weights = tf.pad(kernel_weights,
                                [[0, 0], [0, 32 - self._num_kernel_points]])
    elif self._num_kernel_points > 32 and self._num_kernel_points < 64:
        kernel_weights = tf.pad(kernel_weights,
                                [[0, 0], [0, 64 - self._num_kernel_points]])

    # Compute the projection to the samples.
    weighted_features = basis_proj(
        kernel_weights,
        features,
        neighborhood)
    # remove padding
    weighted_features = weighted_features[:, :, 0:self._num_kernel_points]

    # Compute convolution - hidden layer to output (linear)
    offset_per_center = tf.matmul(
        tf.reshape(weighted_features,
                   [-1, self._num_features_in * self._num_kernel_points]),
        self._kernel_offsets_weights)

    # save for regularization loss computation
    self._offsets = tf.reshape(offset_per_center, [self._num_output_points,
                                                   self._num_kernel_points,
                                                   self._num_dims])
    # project back onto neighbor pairs, shape [M, D*K]
    offset_per_nb = tf.gather(offset_per_center, neighbors[:, 1])
    # reshape to shape [M, K, self._num_dims]
    return  tf.reshape(offset_per_nb,
                       [neighbors.shape[0],
                        self._num_kernel_points,
                        self._num_dims])

  def regularization_loss(self, name=None):
    """ The regularization loss for deformable kernel points.

    As described in Section 3.2 of [KPConv: Flexible and Deformable Convolution
    for Point Clouds. Thomas et al., 2019](https://arxiv.org/abs/1904.08889).

    Returns:
      A 'float`, the sum of repulsory and fitting loss.

    """

    # attractive loss, distances to closest points in each neighborhood
    neighbors = self._cur_neighbors

    # reshape to [M, K]
    point_dist = (self._cur_point_dist / self._sigma)**2
    min_dists_per_nbh = tf.math.unsorted_segment_min(point_dist,
                                                     neighbors[:, 1],
                                                     self._num_output_points)
    loss_fit = tf.reduce_sum(min_dists_per_nbh)

    # repulsive loss for kernel points with overlapping influence area.
    kernel_offsets = self._offsets
    # shape [N2, K, D]
    kernel_points = tf.expand_dims(self._kernel_points, 0) + kernel_offsets
    kernel_dists = tf.linalg.norm(tf.expand_dims(kernel_points, 1) - \
                                  tf.expand_dims(kernel_points, 2),
                                  axis=3)
    kernel_weights = self._weighting(kernel_dists, self._sigma)
    # set weight between same kernel point to zero
    kernel_weights = tf.linalg.set_diag(kernel_weights,
                                        tf.zeros([self._num_output_points,
                                                  self._num_kernel_points]))
    loss_rep = tf.reduce_sum(kernel_weights)

    return loss_fit + loss_rep
