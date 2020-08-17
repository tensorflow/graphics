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
"""This module implements Tensorflow sRGB color space utility functions.

More details about sRGB can be found on [this page.]
(https://en.wikipedia.org/wiki/SRGB)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf
from tensorflow_graphics.image.color_space import constants
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape

# Conversion constants following the naming convention from the 'theory of the
# transformation' section at https://en.wikipedia.org/wiki/SRGB.
_A = constants.srgb_gamma["A"]
_PHI = constants.srgb_gamma["PHI"]
_K0 = constants.srgb_gamma["K0"]
_GAMMA = constants.srgb_gamma["GAMMA"]


def from_linear_rgb(linear_rgb, name=None):
  """Converts linear RGB to sRGB colors.

  Note:
      In the following, A1 to An are optional batch dimensions.

  Args:
    linear_rgb: A Tensor of shape `[A_1, ..., A_n, 3]`, where the last dimension
      represents RGB values in the range [0, 1] in linear color space.
    name: A name for this op that defaults to "srgb_from_linear_rgb".

  Raises:
    ValueError: If `linear_rgb` has rank < 1 or has its last dimension not
      equal to 3.

  Returns:
    A tensor of shape `[A_1, ..., A_n, 3]`, where the last dimension represents
    sRGB values.
  """
  with tf.compat.v1.name_scope(name, "srgb_from_linear_rgb", [linear_rgb]):
    linear_rgb = tf.convert_to_tensor(value=linear_rgb)

    shape.check_static(
        tensor=linear_rgb,
        tensor_name="linear_rgb",
        has_rank_greater_than=0,
        has_dim_equals=(-1, 3))
    linear_rgb = asserts.assert_all_in_range(linear_rgb, 0., 1.)

    # Adds a small eps to avoid nan gradients from the second branch of
    # tf.where.
    linear_rgb += sys.float_info.epsilon
    return tf.compat.v1.where(linear_rgb <= _K0 / _PHI, linear_rgb * _PHI,
                              (1 + _A) * (linear_rgb**(1 / _GAMMA)) - _A)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
