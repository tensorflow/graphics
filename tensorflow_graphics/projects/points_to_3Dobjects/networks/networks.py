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
"""Network related functionalities."""
# python3
import tensorflow as tf


class NormType:
  """Class to abstract the normalization that is used."""

  def __init__(self, norm_name, norm_params):
    if norm_name == 'batchnorm':
      self.norm = tf.keras.layers.BatchNormalization
    else:
      raise ValueError(f'Norm type {norm_name} not recognized.')

    # To finetune without modifying the BN 'trainable': False
    self.params = {
        **norm_params,
        **{
            'name': 'batchnorm',
            'epsilon': 1e-5,
            'momentum': 0.1
        }
    }

  def __call__(self):
    norm_layer = self.norm() if not self.params else self.norm(**self.params)
    return norm_layer


class Regularization:
  """"Regularization class."""

  def __init__(self, regularization_name, wd=0.004):
    self.wd = wd
    if regularization_name == 'l1':
      self.regularizer = tf.keras.regularizers.l1
    elif regularization_name == 'l2':
      self.regularizer = tf.keras.regularizers.l2
    elif regularization_name is None:
      self.regularizer = None
    else:
      raise ValueError(
          f'Regularization type {regularization_name} not recognized.')

  def __call__(self):
    regularizer = self.regularizer(
        self.wd) if self.regularizer is not None else None
    return regularizer
