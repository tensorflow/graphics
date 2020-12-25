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
"""3D Pose feature."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_datasets import features


class Pose(features.FeaturesDict):
  """`FeatureConnector` for 3d pose representations.

  During `_generate_examples`, the feature connector accepts as input any of:

    * `dict:` A dictionary containing the rotation and translation of the
    object (see output format below).

  Output:
    A dictionary containing:

    * 'R': A `float32` tensor with shape `[3, 3]` denoting the
    3D rotation matrix.
    * 't': A `float32` tensor with shape `[3,]` denoting the
    translation vector.

  """

  def __init__(self):
    super(Pose, self).__init__({
        'R': features.Tensor(shape=(3, 3), dtype=tf.float32),
        't': features.Tensor(shape=(3,), dtype=tf.float32),
    })

  def encode_example(self, example_dict):
    """Convert the given pose into a dict convertible to tf example."""

    if not all(key in example_dict for key in ['R', 't']):
      raise ValueError(
          f'Missing keys in provided dictionary! Expecting \'R\' and \'t\', '
          f'but {example_dict.keys()} were given.')

    return super(Pose, self).encode_example(example_dict)

  @classmethod
  def from_json_content(cls, value) -> 'Pose':
    return cls()

  def to_json_content(self):
    return {}
