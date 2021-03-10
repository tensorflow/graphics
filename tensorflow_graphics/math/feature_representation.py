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
"""This module implements methods for feature representation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import tensorflow as tf

from tensorflow_graphics.util import export_api


def positional_encoding(features: tf.Tensor,
                        num_frequencies: int,
                        name="positional_encoding") -> tf.Tensor:
  """Positional enconding of a tensor as described in the NeRF paper (https://arxiv.org/abs/2003.08934).

  Args:
    features: A tensor of shape `[A1, ..., An, M]` where M is the dimension
       of the features.
    num_frequencies: Number N of frequencies for the positional encoding.
    name: A name for this op that defaults to "positional_encoding".

  Returns:
    A tensor of shape `[A1, ..., An, 2*N*M + M]`.
  """
  with tf.name_scope(name):
    features = tf.convert_to_tensor(value=features)

    output = [features]
    for i in range(num_frequencies):
      for fn in [tf.sin, tf.cos]:
        output.append(fn(2. ** i * features))
    return tf.concat(output, -1)

# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
