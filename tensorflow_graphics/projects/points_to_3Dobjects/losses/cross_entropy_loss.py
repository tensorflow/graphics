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
"""Cross entropy loss."""
import tensorflow as tf


class CrossEntropyLoss:
  """This class implements the cross entropy loss."""

  def __init__(self, reduce_batch=False, label_smoothing=0.1,
               soft_shape_labels=False):
    self.reduce_batch = reduce_batch
    self.label_smoothing = label_smoothing
    self.soft_shape_labels = soft_shape_labels

  def __call__(self, prediction, gt, sample):
    indices = sample['indices']
    b, num_classes = tf.shape(prediction)[0], tf.shape(prediction)[-1]
    predicted_centers = tf.gather(
        tf.reshape(prediction, [b, -1, num_classes]), indices, batch_dims=1)

    if not self.soft_shape_labels:
      gt = tf.one_hot(tf.cast(gt, tf.int32), num_classes)
      cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(
          from_logits=True, label_smoothing=self.label_smoothing,
          reduction=tf.keras.losses.Reduction.NONE)
      loss = cross_entropy_loss(gt, predicted_centers)
    else:
      gt = tf.cast(sample['shapes_soft'], dtype=tf.float32)
      gt = tf.reshape(gt, [b, -1, num_classes])
      loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=predicted_centers,
                                                     labels=gt)
    if self.reduce_batch:
      loss = tf.reduce_mean(loss)
    else:
      loss = tf.reduce_mean(loss, axis=[1])
    return loss
