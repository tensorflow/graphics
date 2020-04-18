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
  title={Pointnet: Deep learning on point sets for 3d classification and segmentation},
  author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={652--660},
  year={2017}
}

NOTE: scheduling of batchnorm momentum currently not available in keras. However, experimentally, using the batch norm from Keras resulted in better test accuracy (+1.5%) than the author's custom batch norm version
(https://github.com/charlesq34/pointnet/blob/master/utils/tf_util.py), even when coupled with batch-norm momentum decay. Further, note the author's version is actually performing a "global normalization", as mentioned in the tf.nn.moments documentation (https://www.tensorflow.org/api_docs/python/tf/nn/moments).
"""

import tensorflow as tf
from tensorflow.keras import models #< TODO remove
from tensorflow.keras import layers

class PointNetConv2Layer(tf.keras.layers.Layer):
  def __init__(self, channels, momentum):
    super(PointNetConv2Layer, self).__init__()
    self.momentum = momentum
    self.channels = channels

  def build(self, input_shape):
    self.conv = layers.Conv2D(self.channels, (1, 1), input_shape=input_shape)
    self.bn = layers.BatchNormalization(momentum=self.momentum)

  def call(self, x, training=False):
    return tf.nn.relu(self.bn(self.conv(x), training))

class VanillaEncoder(tf.keras.models.Model):
  def __init__(self, batchnorm_momentum):
    super(tf.keras.models.Model, self).__init__()
    # self.conv1 = layers.Conv2D(64, (1, 1), input_shape=(num_points, 1, 3))
    self.conv1 = PointNetConv2Layer(64, momentum=batchnorm_momentum, input_shape=(2048, 1, 3))
    self.conv2 = PointNetConv2Layer(64, momentum=batchnorm_momentum)
    self.conv3 = PointNetConv2Layer(64, momentum=batchnorm_momentum)
    self.conv4 = PointNetConv2Layer(128, momentum=batchnorm_momentum)
    self.conv5 = PointNetConv2Layer(1024, momentum=batchnorm_momentum)
  
  def call(self, x, training=False):
    num_points = x.shape[-2]
    x = tf.expand_dims(x, axis=2)     #< BxNx1xD
    x = self.conv1(x, training)       #< BxNx1x64
    x = self.conv2(x, training)       #< BxNx1x64
    x = self.conv3(x, training)       #< BxNx1x64
    x = self.conv4(x, training)       #< BxNx1x128
    x = self.conv5(x, training)       #< BxNx1x1024
    x = tf.math.reduce_max(x, axis=1) #< Bx1x1024
    return tf.squeeze(x)              #< Bx1024

class VanillaEncoder_LEGACY(tf.keras.layers.Layer):
  def __init__(self, num_points, batchnorm_momentum=.5):
    super(VanillaEncoder_LEGACY, self).__init__()

    print("TODO: still calling legacy encoder")
    self.model = models.Sequential()
    self.model.add(PointNetConv2Layer(64, momentum=batchnorm_momentum)) 
    self.model.add(PointNetConv2Layer(64, momentum=batchnorm_momentum))
    self.model.add(PointNetConv2Layer(64, momentum=batchnorm_momentum))
    self.model.add(PointNetConv2Layer(128, momentum=batchnorm_momentum))
    self.model.add(PointNetConv2Layer(1024, momentum=batchnorm_momentum))

  def build(self, input_shape):
    B,N,C = input_shape
    self.model.build(input_shape=(B,N,1,C))

  def call(self, x, training):
    x = tf.expand_dims(x, axis=2) #< BxNx1xD (prep for Conv2D)
    x = self.model(x, training) #< BxNx1x1024
    x = tf.math.reduce_max(x, axis=1) #< Bx1x1024
    return tf.squeeze(x) #< Bx1024


class ClassificationHead(object):
  def __init__(self, num_classes=40, n_features=1024, batchnorm_momentum=.5):
    self.model = models.Sequential()
    self.model.add(layers.Dense(512, input_shape=(n_features,)))
    self.model.add(layers.BatchNormalization(momentum=batchnorm_momentum))
    self.model.add(layers.Activation("relu"))
    self.model.add(layers.Dense(256))
    self.model.add(layers.BatchNormalization(momentum=batchnorm_momentum))
    self.model.add(layers.Activation("relu"))
    self.model.add(layers.Dropout(0.3))
    self.model.add(layers.Dense(num_classes, activation="linear"))
    self.trainable_variables = self.model.trainable_variables

  def __call__(self, features, training):
    return self.model(features, training) #< Bx1


class PointNetVanillaClassifier(object):
  
  def __init__(self, num_points, num_classes, batchnorm_momentum=.5):
    self.encoder = VanillaEncoder_LEGACY(num_points=num_points, batchnorm_momentum=batchnorm_momentum)
    self.classifier = ClassificationHead(num_classes=num_classes, batchnorm_momentum=batchnorm_momentum)
    self.trainable_variables = self.encoder.trainable_variables + self.classifier.trainable_variables 
  
  def __call__(self, points, training):
    features = self.encoder(points, training) #< Bx1024
    logits = self.classifier(features, training) #< Bx40
    return logits

  @staticmethod
  def loss(labels, logits):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits
    residual = cross_entropy(labels=labels, logits=logits)
    loss = tf.reduce_mean(residual)
    return loss

if __name__ == "__main__":
  # x = tf.random.uniform((32,2048,3))
  # encoder = VanillaEncoder(2048, momentum=.5)
  # print(encoder(x).shape)

  x = tf.random.uniform((32,2048,1,3))
  layer = PointNetConv2Layer(64, .5)
  # layer.build(x.shape)
  y = layer(x)
  print(y.shape)
  # print([var.name for var in layer.trainable_variables])