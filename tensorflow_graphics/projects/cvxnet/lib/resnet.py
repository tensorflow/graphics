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
"""ResNet Architecture."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
keras = tf.keras


class Resnet18(keras.Model):
  """ResNet-18 (V1)."""

  def __init__(self, feature_dims):
    super(Resnet18, self).__init__()
    self.conv1 = keras.layers.Conv2D(
        64, 7, strides=2, padding='same', use_bias=False)
    self.bn1 = keras.layers.BatchNormalization()
    self.relu1 = keras.layers.ReLU()
    self.maxpool = keras.layers.MaxPooling2D(3, strides=2, padding='same')
    layers = [2, 2, 2, 2]

    self.layer1 = ResLayer(BasicBlock, 64, 64, layers[0])
    self.layer2 = ResLayer(BasicBlock, 64, 128, layers[1], stride=2)
    self.layer3 = ResLayer(BasicBlock, 128, 256, layers[2], stride=2)
    self.layer4 = ResLayer(BasicBlock, 256, 512, layers[3], stride=2)

    self.fc = keras.layers.Dense(feature_dims, activation=None)

  def call(self, x, training=False):
    x = self.conv1(x)
    x = self.bn1(x, training=training)
    x = self.relu1(x)
    x = self.maxpool(x)

    x = self.layer1(x, training=training)
    x = self.layer2(x, training=training)
    x = self.layer3(x, training=training)
    x = self.layer4(x, training=training)

    x = tf.reduce_mean(x, axis=(1, 2))
    x = self.fc(x)

    return x


class ResLayer(keras.Model):
  """Residual Layer."""

  def __init__(self, block, inplanes, planes, blocks, stride=1):
    super(ResLayer, self).__init__()
    if stride != 1 or inplanes != planes:
      downsample = True
    else:
      downsample = False

    self.conv_layers = []
    self.conv_layers.append(block(planes, stride, downsample=downsample))
    for unused_i in range(1, blocks):
      self.conv_layers.append(block(planes))

  def call(self, x, training=True):
    for layer in self.conv_layers:
      x = layer(x, training=training)
    return x


class BasicBlock(keras.Model):
  """Building block of resnet."""

  def __init__(self, planes, stride=1, downsample=False):
    super(BasicBlock, self).__init__()

    self.conv1 = keras.layers.Conv2D(
        planes, 3, strides=stride, padding='same', use_bias=False)
    self.bn1 = keras.layers.BatchNormalization()
    self.conv2 = keras.layers.Conv2D(planes, 3, padding='same', use_bias=False)
    self.bn2 = keras.layers.BatchNormalization()
    if downsample:
      self.downsample = downsample
      self.dconv1 = keras.layers.Conv2D(
          planes, 1, strides=stride, padding='same', use_bias=False)
      self.dbn1 = keras.layers.BatchNormalization()
    else:
      self.downsample = downsample

  def call(self, x, training=True):
    residual = x
    if self.downsample:
      residual = self.dconv1(residual)
      residual = self.dbn1(residual, training=training)

    x = self.conv1(x)
    x = self.bn1(x, training=training)
    x = tf.nn.relu(x)
    x = self.conv2(x)
    x = self.bn2(x, training=training)

    x += residual
    x = tf.nn.relu(x)

    return x
