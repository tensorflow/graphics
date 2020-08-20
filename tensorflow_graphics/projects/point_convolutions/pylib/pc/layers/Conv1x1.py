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


class Conv1x1:
  """ A 1x1 convolution on the point features. This op reshapes the arguments
  to pass them to `tf.keras.layers.Conv1D` to perform the equivalent
  convolution operation.

  Note: This uses less memory than a point cloud convolution layer with a 1x1
    neighborhood, but might be slower for large feature dimensions.

  Args:
    num_features_in: An `int` `C_in`, the number of input features.
    num_features_out: An `int` `C_out`, the number of output features.
    **kwargs: Additional keyword arguments to be passed to
      `tf.keras.layers.Conv1D`. (optional)

  """

  def __init__(self, num_features_in, num_features_out, **kwargs):
    self.conv_layer = tf.keras.layers.Conv1D(
        filters=num_features_out,
        kernel_size=1,
        input_shape=[None, 1, num_features_in],
        **kwargs)
    self._num_features_out = num_features_out

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
    features = tf.expand_dims(features, 1)
    features = self.conv_layer(features)
    features = tf.reshape(features, [-1, self._num_features_out])
    return _format_output(features,
                          point_cloud,
                          return_sorted,
                          return_padded)
