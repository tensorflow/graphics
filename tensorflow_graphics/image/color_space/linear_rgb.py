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
"""This module implements utility functions for the linear RGB color space."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_graphics.image.color_space import constants
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape
from tensorflow_graphics.util import type_alias

# Conversion constants following the naming convention from the 'theory of the
# transformation' section at https://en.wikipedia.org/wiki/SRGB.
_A = constants.srgb_gamma["A"]
_PHI = constants.srgb_gamma["PHI"]
_K0 = constants.srgb_gamma["K0"]
_GAMMA = constants.srgb_gamma["GAMMA"]


def from_srgb(srgb: type_alias.TensorLike,
              name: str = "linear_rgb_from_srgb") -> tf.Tensor:
  """Converts sRGB colors to linear colors.

  Note:
      In the following, A1 to An are optional batch dimensions.

  Args:
    srgb: A tensor of shape `[A_1, ..., A_n, 3]`, where the last dimension
      represents sRGB values.
    name: A name for this op that defaults to "linear_rgb_from_srgb".

  Raises:
    ValueError: If `srgb` has rank < 1 or has its last dimension not equal to 3.

  Returns:
    A tensor of shape `[A_1, ..., A_n, 3]`, where the last dimension represents
    RGB values in linear color space.
  """
  with tf.name_scope(name):
    srgb = tf.convert_to_tensor(value=srgb)

    shape.check_static(
        tensor=srgb,
        tensor_name="srgb",
        has_rank_greater_than=0,
        has_dim_equals=(-1, 3))

    srgb = asserts.assert_all_in_range(srgb, 0., 1.)
    return tf.where(srgb <= _K0, srgb / _PHI, ((srgb + _A) / (1 + _A))**_GAMMA)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
