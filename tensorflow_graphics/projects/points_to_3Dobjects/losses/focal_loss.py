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
"""Focal loss."""
import tensorflow as tf


class FocalLoss:
  """Loss class."""

  def __init__(self, reduce_batch=False):
    self.reduce_batch = reduce_batch

  def __call__(self, prediction, gt, sample):
    prediction = tf.clip_by_value(tf.math.sigmoid(prediction), 1e-4, 1 - 1e-4)

    positive_mask = tf.cast(tf.equal(gt, 1.0), tf.float32)
    negative_mask = tf.cast(tf.less(gt, 1.0), tf.float32)

    negative_weights = tf.math.pow(1 - gt, 4)
    positive_loss = tf.math.log(prediction) * tf.math.pow(1 - prediction,
                                                          2) * positive_mask
    negative_loss = tf.math.log(1 - prediction) * tf.math.pow(
        prediction, 2) * negative_weights * negative_mask

    if self.reduce_batch:
      positive_loss = tf.reduce_sum(positive_loss)
      negative_loss = tf.reduce_sum(negative_loss)
      number_positive = tf.reduce_sum(positive_mask)
      if number_positive == 0:
        loss = -positive_loss
      else:
        loss = -(positive_loss + negative_loss) / number_positive
    else:
      positive_loss = tf.reduce_sum(positive_loss, axis=[1, 2, 3])
      negative_loss = tf.reduce_sum(negative_loss, axis=[1, 2, 3])
      number_positive = tf.reduce_sum(positive_mask, axis=[1, 2, 3])
      number_positive_mask = tf.cast(tf.equal(number_positive, 0), tf.float32)
      number_positive = tf.where(
          tf.math.equal(number_positive, 0), tf.ones_like(number_positive),
          number_positive)
      loss_1 = -positive_loss
      loss_2 = -(positive_loss + negative_loss) / number_positive
      loss = loss_1 * number_positive_mask + loss_2 * (1 - number_positive_mask)

    return loss
