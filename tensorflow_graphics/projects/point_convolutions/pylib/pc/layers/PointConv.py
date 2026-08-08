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
"""Classes to for PointConv point cloud convolutions"""

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


class PointConv(tf.Module):
  """ Monte-Carlo convolution for point clouds.

  Based on the paper [PointConv: Deep Convolutional Networks on 3D Point
  Clouds. Wu et al., 2019](https://arxiv.org/abs/1811.07246).
  Uses a single MLP with one hidden layer as convolution kernel.

  Args:
    num_features_in: An `int`, C_in, the number of features per input point.
    num_features_out: An `int`, C_out, the number of features to compute.
    num_dims: An `int`, the input dimension to the kernel MLP. Should be the
      dimensionality of the point cloud.
    size_hidden: An ìnt`, the number of neurons in the hidden layer of the
        kernel MLP, must be in `[8, 16, 32]`, defaults to `32`. (optional).
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
               size_hidden=32,
               non_linearity_type='relu',
               initializer_weights=None,
               initializer_biases=None,
               name=None):

    super().__init__(name=name)

    self._num_features_in = num_features_in
    self._num_features_out = num_features_out
    self._size_hidden = size_hidden
    self._num_dims = num_dims
    self._non_linearity_type = non_linearity_type

    if name is None:
      self._name = 'PointConv'
    else:
      self._name = name

    # initialize variables
    if initializer_weights is None:
      initializer_weights = tf.initializers.GlorotNormal
    if initializer_biases is None:
      initializer_biases = tf.initializers.zeros

    # Hidden layer of the kernel.
    weights_init_obj = initializer_weights()
    self._basis_axis_tf = tf.Variable(
          weights_init_obj(
              shape=[self._num_dims, self._size_hidden],
              dtype=tf.float32),
          trainable=True,
          name=self._name + "/hidden_vectors")

    bias_init_obj = initializer_biases()
    self._basis_bias_tf = tf.Variable(
        bias_init_obj(
            shape=[1, self._size_hidden],
            dtype=tf.float32),
        trainable=True,
        name=self._name + "/hidden_bias")

    # Convolution weights.
    self._weights = tf.Variable(
          weights_init_obj(
              shape=[
                  self._size_hidden * self._num_features_in,
                  self._num_features_out],
              dtype=tf.float32),
          trainable=True,
          name=self._name + "/conv_weights")

    # Weights of the non-linear transform of the pdf.
    self._weights_pdf = \
        [tf.Variable(
          weights_init_obj(
              shape=[1, 16],
              dtype=tf.float32),
          trainable=True,
          name=self._name + "/pdf_weights_1"),
         tf.Variable(
          weights_init_obj(
              shape=[16, 1],
              dtype=tf.float32),
          trainable=True,
          name=self._name + "/pdf_weights_2")]

    self._biases_pdf = \
        [tf.Variable(
          bias_init_obj(
              shape=[1, 16],
              dtype=tf.float32),
          trainable=True,
          name=self._name + "/pdf_biases_1"),
         tf.Variable(
          bias_init_obj(
              shape=[1, 1],
              dtype=tf.float32),
          trainable=True,
          name=self._name + "/pdf_biases_2")]

  def _point_conv(self,
                  kernel_inputs,
                  neighborhood,
                  pdf,
                  features,
                  non_linearity_type='relu'):
    """ Method to compute a PointConv convolution using a single
    MLP with one hidden layer as implicit convolution kernel function.

    Args:
      kernel_inputs: A `float` `Tensor` of shape `[M, L]`, the input to the
        kernel MLP.
      neighborhood: A `Neighborhood` instance.
      pdf: A `float` `Tensor` of shape `[M]`, the point densities.
      features: A `float` `Tensor` of shape `[N, C1]`, the input features.
      non_linearity_type: An `string`, specifies the type of the activation
        function used inside the kernel MLP.
        Possible: `'ReLU', 'lReLU', 'ELU'`, defaults to leaky ReLU. (optional)

    Returns:
      A `float` `Tensor` of shape `[N,C2]`, the output features.

    """

    # Compute the hidden layer MLP
    basis_neighs = tf.matmul(kernel_inputs, self._basis_axis_tf) + \
        self._basis_bias_tf
    basis_neighs = \
        non_linearity_types[non_linearity_type.lower()](basis_neighs)

    # Normalize the pdf
    max_pdf = tf.math.unsorted_segment_max(
        pdf,
        neighborhood._original_neigh_ids[:, 1],
        tf.shape(neighborhood._samples_neigh_ranges)[0])
    neigh_max_pdfs = tf.gather(max_pdf, neighborhood._original_neigh_ids[:, 1])
    cur_pdf = pdf / neigh_max_pdfs
    cur_pdf = tf.reshape(cur_pdf, [-1, 1])

    # Non-linear transform pdf
    cur_pdf = tf.nn.relu(tf.matmul(cur_pdf, self._weights_pdf[0]) +\
                         self._biases_pdf[0])
    cur_pdf = tf.matmul(cur_pdf, self._weights_pdf[1]) + self._biases_pdf[1]

    # Scale features
    basis_neighs = basis_neighs / cur_pdf

    # Compute the projection to the samples.
    weighted_features = basis_proj(
        basis_neighs,
        features,
        neighborhood)

    #Compute convolution - hidden layer to output (linear)
    convolution_result = tf.matmul(
        tf.reshape(weighted_features,
                   [-1, self._num_features_in * self._size_hidden]),
        self._weights)

    return convolution_result

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
    points_diff = (neigh_point_coords - center_point_coords)

    #Compute PointConv convolution
    convolution_result = self._point_conv(
        points_diff, neigh, pdf, features, self._non_linearity_type)

    return _format_output(convolution_result,
                          point_cloud_out,
                          return_sorted,
                          return_padded)
