#Copyright 2020 Google LLC
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
"""
Implementation of the PointNet networks from:

@inproceedings{qi2017pointnet,
  title={Pointnet: Deep learning on point sets
         for3d classification and segmentation},
  author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern
             recognition},
  pages={652--660},
  year={2017}}

NOTE: scheduling of batchnorm momentum currently not available in keras. However
experimentally, using the batch norm from Keras resulted in better test accuracy
(+1.5%) than the author's [custom batch norm version](https://github.com/charlesq34/pointnet/blob/master/utils/tf_util.py)
even when coupled with batchnorm momentum decay. Further, note the author's
version is actually performing a "global normalization", as mentioned in the
[tf.nn.moments documentation] (https://www.tensorflow.org/api_docs/python/tf/nn/moments).
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class PointNetConv2Layer(Layer):
  def __init__(self, channels, momentum):
    super(PointNetConv2Layer, self).__init__()
    self.momentum = momentum
    self.channels = channels

  def build(self, input_shape):
    self.conv = layers.Conv2D(self.channels, (1, 1), input_shape=input_shape)
    self.bn = layers.BatchNormalization(momentum=self.momentum)

  def call(self, x, training):
    return tf.nn.relu(self.bn(self.conv(x), training))


class PointNetDenseLayer(Layer):
  def __init__(self, channels, momentum):
    super(PointNetDenseLayer, self).__init__()
    self.momentum = momentum
    self.channels = channels

  def build(self, input_shape):
    self.dense = layers.Dense(self.channels, input_shape=input_shape)
    self.bn = layers.BatchNormalization(momentum=self.momentum)

  def call(self, x, training):
    return tf.nn.relu(self.bn(self.dense(x), training))


class VanillaEncoder(Layer):
  def __init__(self, momentum=.5):
    super(VanillaEncoder, self).__init__()
    self.conv1 = PointNetConv2Layer(64, momentum)
    self.conv2 = PointNetConv2Layer(64, momentum)
    self.conv3 = PointNetConv2Layer(64, momentum)
    self.conv4 = PointNetConv2Layer(128, momentum)
    self.conv5 = PointNetConv2Layer(1024, momentum)

  def call(self, x, training):
    x = tf.expand_dims(x, axis=2)      # BxNx1xD
    x = self.conv1(x, training)        # BxNx1x64
    x = self.conv2(x, training)        # BxNx1x64
    x = self.conv3(x, training)        # BxNx1x64
    x = self.conv4(x, training)        # BxNx1x128
    x = self.conv5(x, training)        # BxNx1x1024
    x = tf.math.reduce_max(x, axis=1)  # Bx1x1024
    return tf.squeeze(x)               # Bx1024


class ClassificationHead(tf.keras.layers.Layer):
  def __init__(self, num_classes, momentum):
    super(ClassificationHead, self).__init__()
    self.dense1 = PointNetDenseLayer(512, momentum)
    self.dense2 = PointNetDenseLayer(256, momentum)
    self.dropout = layers.Dropout(0.3)
    self.dense3 = layers.Dense(num_classes, activation="linear")

  def call(self, x, training):
    x = self.dense1(x, training)
    x = self.dense2(x, training)
    x = self.dropout(x, training)
    x = self.dense3(x)
    return x  # Bx1


class PointNetVanillaClassifier(tf.keras.layers.Layer):

  def __init__(self, num_classes, momentum=.5):
    super(PointNetVanillaClassifier, self).__init__()
    self.encoder = VanillaEncoder(momentum)
    self.classifier = ClassificationHead(num_classes, momentum)

  def call(self, points, training):
    features = self.encoder(points, training)  # Bx1024
    logits = self.classifier(features, training)  # Bx40
    return logits

  @staticmethod
  def loss(labels, logits):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits
    residual = cross_entropy(labels=labels, logits=logits)
    loss = tf.reduce_mean(residual)
    return loss
