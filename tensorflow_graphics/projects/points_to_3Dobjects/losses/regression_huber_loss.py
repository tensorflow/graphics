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
"""Huber loss."""
import tensorflow as tf


class HuberLoss:
  """Class loss."""

  def __init__(self, reduce_batch=False, order=2):
    self.reduce_batch = reduce_batch
    self.ord = order

  def __call__(self, prediction, gt, sample):

    loss = tf.norm(prediction - gt, ord=self.ord, axis=3)

    if self.reduce_batch:
      loss = tf.reduce_mean(loss)
    else:
      loss = tf.reduce_mean(loss, axis=[1, 2])
    return loss
