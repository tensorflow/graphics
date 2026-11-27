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
"""Classes to Monte-Carlo point cloud convolutions"""

import tensorflow as tf
from pylib.pc.utils import _flatten_features

from pylib.pc import PointCloud
from pylib.pc import Grid
from pylib.pc import Neighborhood
from pylib.pc import KDEMode

from pylib.pc.custom_ops import basis_proj
from pylib.pc.layers.utils import _format_output

non_linearity_types = {'relu': tf.nn.relu,
                       'lrelu': tf.nn.leaky_relu,
                       'leakyrelu': tf.nn.leaky_relu,
                       'leaky_relu': tf.nn.leaky_relu,
                       'elu': tf.nn.elu}


class MCConv(tf.Module):
  """ Monte-Carlo convolution for point clouds.

  Based on the paper [Monte Carlo Convolution for Learning on Non-Uniformly
  Sampled Point Clouds. Hermosilla et al., 2018]
  (https://arxiv.org/abs/1806.01759).
  Uses a multiple MLPs as convolution kernels.

  Args:
    num_features_in: An `int`, `C_in`, the number of features per input point.
    num_features_out: An `int`, `C_out`, the number of features to compute.
    num_dims: An `int`, the input dimension to the kernel MLP. Should be the
      dimensionality of the point cloud.
    num_mlps: An `int`, number of MLPs used to compute the output features.
      Warning: num_features_out should be divisible by num_mlps.
    mlp_size: An ìnt list`, list with the number of layers and hidden neurons
      of the MLP used as kernel, defaults to `[8]`. (optional).
    non_linearity_type: An `string`, specifies the type of the activation
      function used inside the kernel MLP.
      Possible: `'ReLU', 'lReLU', 'ELU'`, defaults to leaky ReLU. (optional)
    initializer_weights: A `tf.initializer` for the kernel MLP weights,
      default `TruncatedNormal`. (optional)
    initializer_biases: A `tf.initializer` for the kernel MLP biases,
      default: `zeros`. (optional)

  """

  def __init__(self,
               num_features_in,
               num_features_out,
               num_dims,
               num_mlps=4,
               mlp_size=[8],
               non_linearity_type='leaky_relu',
               initializer_weights=None,
               initializer_biases=None,
               name=None):

    super().__init__(name=name)

    self._num_features_in = num_features_in
    self._num_features_out = num_features_out
    self._num_mlps = num_mlps
    self._mlp_size = mlp_size
    self._num_dims = num_dims
    self._non_linearity_type = non_linearity_type

    if num_features_out % num_mlps != 0:
      raise ValueError(
          "The number of output features must be divisible by the number" +
          " of kernel MLPs")

    if name is None:
      self._name = 'MCConv'
    else:
      self._name = name

    # initialize variables
    if initializer_weights is None:
      initializer_weights = tf.initializers.TruncatedNormal
    if initializer_biases is None:
      initializer_biases = tf.initializers.zeros

    self._weights_tf = []
    self._bias_tf = []
    prev_num_inut = self._num_dims
    for cur_layer_iter, cur_layer in enumerate(self._mlp_size):

      if cur_layer_iter:
        std_dev = tf.math.sqrt(1.0 / float(prev_num_inut))
      else:
        std_dev = tf.math.sqrt(2.0 / float(prev_num_inut))

      weights_init_obj = initializer_weights(stddev=std_dev)
      self._weights_tf.append(tf.Variable(
          weights_init_obj(
              shape=[self._num_mlps, prev_num_inut, cur_layer],
              dtype=tf.float32),
          trainable=True,
          name=self._name + "/weights_" + str(cur_layer_iter)))

      bias_init_obj = initializer_biases()
      self._bias_tf.append(tf.Variable(
          bias_init_obj(shape=[self._num_mlps, 1, cur_layer],
                        dtype=tf.float32),
          trainable=True,
          name=self._name + "/bias_" + str(cur_layer_iter)))
      prev_num_inut = cur_layer

    std_dev = tf.math.sqrt(2.0 / \
                           float(cur_layer * self._num_features_in))

    weights_init_obj = initializer_weights(stddev=std_dev)
    self._final_weights_tf = tf.Variable(
        weights_init_obj(
            shape=[
                self._num_mlps,
                cur_layer * self._num_features_in,
                self._num_features_out // self._num_mlps],
            dtype=tf.float32),
        trainable=True,
        name=self._name + "/final_weights_" + str(cur_layer_iter))

  def _monte_carlo_conv(self,
                        kernel_inputs,
                        neighborhood,
                        pdf,
                        features,
                        non_linearity_type='leaky_relu'):
    """ Method to compute a Monte-Carlo integrated convolution using multiple
    MLPs as implicit convolution kernel functions.

    Args:
      kernel_inputs: A `float` `Tensor` of shape `[M, L]`, the input to the
        kernel MLP.
      neighborhood: A `Neighborhood` instance.
      pdf: A `float` `Tensor` of shape `[M]`, the point densities.
      features: A `float` `Tensor` of shape `[N, C1]`, the input features.
      non_linearity_type: An `string`, specifies the type of the activation
        function used inside the kernel MLP.
        Possible: `'ReLU', 'leaky_ReLU', 'ELU'`, defaults to leaky ReLU.
        (optional)

    Returns:
      A `float` `Tensor` of shape `[N,C2]`, the output features.

    """

    # Compute the hidden layer MLP
    cur_inputs = tf.tile(tf.reshape(kernel_inputs, [1, -1, self._num_dims]),
                         [self._num_mlps, 1, 1])
    for cur_layer_iter in range(len(self._weights_tf)):
      cur_inputs = tf.matmul(cur_inputs, self._weights_tf[cur_layer_iter]) + \
        self._bias_tf[cur_layer_iter]
      cur_inputs = non_linearity_types[non_linearity_type.lower()](cur_inputs)
    cur_inputs = tf.reshape(tf.transpose(cur_inputs, [1, 0, 2]),
                            [-1, self._mlp_size[-1] * self._num_mlps]) \
        / tf.reshape(pdf, [-1, 1])

    # Compute the projection to the samples.
    weighted_features = basis_proj(
        cur_inputs,
        features,
        neighborhood)

    # Reshape features
    weighted_features = tf.transpose(tf.reshape(weighted_features,
                                                [-1, self._num_features_in,
                                                 self._num_mlps,
                                                 self._mlp_size[-1]]),
                                     [2, 0, 1, 3])

    #Compute convolution - hidden layer to output (linear)
    convolution_result = tf.matmul(
        tf.reshape(
            weighted_features,
            [self._num_mlps, -1, self._num_features_in * self._mlp_size[-1]]),
        self._final_weights_tf)

    return tf.reshape(tf.transpose(convolution_result, [1, 0, 2]),
                      [-1, self._num_features_out])

  def __call__(self,
               features,
               point_cloud_in: PointCloud,
               point_cloud_out: PointCloud,
               radius,
               neighborhood=None,
               bandwidth=0.2,
               return_sorted=False,
               return_padded=False,
               name=None):
    """ Computes the Monte-Carlo Convolution between two point clouds.

    Note:
      In the following, `A1` to `An` are optional batch dimensions.
      `C_in` is the number of input features.
      `C_out` is the number of output features.

    Args:
      features: A `float` `Tensor` of shape `[N_in, C_in]` or
        `[A1, ..., An,V, C_in]`.
      point_cloud_in: A 'PointCloud' instance, on which the features are
        defined.
      point_cloud_out: A `PointCloud` instance, on which the output
        features are defined.
      radius: A `float`, the convolution radius.
      neighborhood: A `Neighborhood` instance, defining the neighborhood
        with centers from `point_cloud_out` and neighbors in `point_cloud_in`.
        If `None` it is computed internally. (optional)
      bandwidth: A `float`, the bandwidth used in the kernel density
        estimation on the input point cloud. (optional)
      return_sorted: A `boolean`, if `True` the output tensor is sorted
        according to the batch_ids. (optional)
      return_padded: A `bool`, if 'True' the output tensor is sorted and
        zero padded. (optional)

    Returns:
      A `float` `Tensor` of shape
        `[N_out, C_out]`, if `return_padded` is `False`
      or
        `[A1, ..., An, V_out, C_out]`, if `return_padded` is `True`.

    """

    features = tf.cast(tf.convert_to_tensor(value=features),
                       dtype=tf.float32)
    features = _flatten_features(features, point_cloud_in)

    #Create the radii tensor.
    radii_tensor = tf.cast(tf.repeat([radius], self._num_dims),
                           dtype=tf.float32)
    #Create the badnwidth tensor.
    bwTensor = tf.repeat(bandwidth, self._num_dims)

    if neighborhood is None:
      #Compute the grid
      grid = Grid(point_cloud_in, radii_tensor)
      #Compute the neighborhoods
      neigh = Neighborhood(grid, radii_tensor, point_cloud_out)
    else:
     neigh = neighborhood
    pdf = neigh.get_pdf(bandwidth=bwTensor, mode=KDEMode.constant)

    #Compute kernel inputs.
    neigh_point_coords = tf.gather(
        point_cloud_in._points, neigh._original_neigh_ids[:, 0])
    center_point_coords = tf.gather(
        point_cloud_out._points, neigh._original_neigh_ids[:, 1])
    points_diff = (neigh_point_coords - center_point_coords) / \
        tf.reshape(radii_tensor, [1, self._num_dims])

    #Compute Monte-Carlo convolution
    convolution_result = self._monte_carlo_conv(
        points_diff, neigh, pdf, features, self._non_linearity_type)

    return _format_output(convolution_result,
                          point_cloud_out,
                          return_sorted,
                          return_padded)
