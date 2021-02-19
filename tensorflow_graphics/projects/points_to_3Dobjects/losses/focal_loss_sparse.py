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


class SparseFocalLoss:
  """This class implements the cross entropy loss."""

  def __init__(self, reduce_batch=False, gamma=2.0, alpha=0.25):
    self.reduce_batch = reduce_batch
    self.gamma = gamma
    self.alpha = alpha

  def __call__(self, prediction, gt, sample):
    indices = sample['indices']
    b, num_classes = tf.shape(prediction)[0], tf.shape(prediction)[-1]
    logits = tf.gather(
        tf.reshape(prediction, [b, -1, num_classes]), indices, batch_dims=1)

    targets = tf.one_hot(tf.cast(gt, tf.int32), num_classes)
    positive_label_mask = tf.equal(targets, 1.0)

    neg_logits = -1.0 * logits
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets,
                                                            logits=logits)
    modulator = tf.exp(self.gamma * targets * neg_logits -
                       self.gamma * tf.math.softplus(neg_logits))
    loss = modulator * cross_entropy
    weighted_loss = tf.where(positive_label_mask, self.alpha * loss,
                             (1.0 - self.alpha) * loss)

    if self.reduce_batch:
      loss = tf.reduce_mean(weighted_loss)
    else:
      loss = tf.reduce_mean(weighted_loss, axis=[1, 2])
    return loss
