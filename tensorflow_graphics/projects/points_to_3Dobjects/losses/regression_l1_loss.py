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
"""Regression L1 loss."""
import tensorflow as tf


class RegL1Loss:
  """Loss class."""

  def __init__(self, reduce_batch=False, sparse=False):
    self.reduce_batch = reduce_batch
    self.sparse = sparse

  def __call__(self, prediction, gt, sample):
    indices = sample['indices']
    b, n = tf.shape(prediction)[0], tf.shape(prediction)[-1]
    mask = tf.tile(
        tf.expand_dims(
            sample['valid_boxes_mask'],
            axis=-1), (1, 1, n))
    mask = tf.cast(mask, tf.float32)
    if self.sparse:
      prediction_centers = prediction
      # permutation = tf.argsort(indices)
      # gt = tf.gather(gt, permutation, batch_dims=1)
    else:
      prediction_centers = tf.gather(
          tf.reshape(prediction, [b, -1, n]), indices, batch_dims=1)
    loss = tf.math.abs((gt * mask) - (prediction_centers * mask))
    if self.reduce_batch:
      loss = tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-4)
    else:
      loss = tf.reduce_sum(
          loss, axis=[1, 2]) / (
              tf.reduce_sum(mask, axis=[1, 2]) + 1e-4)
    return loss
