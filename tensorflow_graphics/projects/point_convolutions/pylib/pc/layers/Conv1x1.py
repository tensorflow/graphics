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
"""Class to represent point cloud 1x1 convolution"""

import tensorflow as tf
from pylib.pc.utils import _flatten_features
from pylib.pc.layers.utils import _format_output

from pylib.pc import PointCloud


class Conv1x1(tf.Module):
  """ A 1x1 convolution on the point features. This op reshapes the arguments
  to pass them to `tf.keras.layers.Conv1D` to perform the equivalent
  convolution operation.

  Note: This uses less memory than a point cloud convolution layer with a 1x1
    neighborhood, but might be slower for large feature dimensions.

  Args:
    num_features_in: An `int` `C_in`, the number of input features.
    num_features_out: An `int` `C_out`, the number of output features.
    name: An `string` with the name of the module.
  """

  def __init__(self, num_features_in, num_features_out, name=None):

    super().__init__(name=name)

    if not(name is None):
        weigths_name = name + "/weights"
        bias_name = name + "/bias"
    else:
        weigths_name = "Conv1x1/weights"
        bias_name = "Conv1x1/bias"

    std_dev = tf.math.sqrt(2.0 / float(num_features_in))
    weights_init_obj = tf.initializers.TruncatedNormal(stddev=std_dev)
    self._weights_tf = tf.Variable(
        weights_init_obj(
            shape=[num_features_in, num_features_out],
            dtype=tf.float32),
        trainable=True,
        name=weigths_name)

    bias_init_obj = tf.initializers.zeros()
    self._bias_tf = tf.Variable(
        bias_init_obj(
            shape=[1, num_features_out],
            dtype=tf.float32),
        trainable=True,
        name=bias_name)

  def __call__(self,
               features,
               point_cloud,
               return_sorted=False,
               return_padded=False):
    """ Computes the 1x1 convolution on a point cloud.

    Note:
      In the following, `A1` to `An` are optional batch dimensions.
      `C_in` is the number of input features.
      `C_out` is the number of output features.

    Args:
      features: A `float` `Tensor` of shape `[N_in, C_in]` or
        `[A1, ..., An, V, C_in]`.
      point_cloud: A 'PointCloud' instance, on which the features are
        defined.
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
    features = _flatten_features(features, point_cloud)
    features = tf.matmul(features, self._weights_tf) + self._bias_tf
    return _format_output(features,
                          point_cloud,
                          return_sorted,
                          return_padded)
