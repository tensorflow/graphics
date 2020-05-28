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
"""Tests for pointnet layers."""

# pylint: disable=invalid-name

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_graphics.nn.layer.pointnet import ClassificationHead
from tensorflow_graphics.nn.layer.pointnet import PointNetConv2Layer
from tensorflow_graphics.nn.layer.pointnet import PointNetDenseLayer
from tensorflow_graphics.nn.layer.pointnet import PointNetVanillaClassifier
from tensorflow_graphics.nn.layer.pointnet import VanillaEncoder
from tensorflow_graphics.util import test_case


class RandomForwardExecutionTest(test_case.TestCase):

  @parameterized.parameters(
      ((32, 2048, 1, 3), (32), (.5), True),
      ((32, 2048, 1, 3), (32), (.5), False),
      ((32, 2048, 1, 2), (16), (.99), True),
  )
  def test_conv2(self, input_shape, channels, momentum, training):
    B, N, X, _ = input_shape
    inputs = tf.random.uniform(input_shape)
    layer = PointNetConv2Layer(channels, momentum)
    outputs = layer(inputs, training=training)
    assert outputs.shape == (B, N, X, channels)

  @parameterized.parameters(
      ((32, 1024), (40), (.5), True),
      ((32, 2048), (20), (.5), False),
      ((32, 512), (10), (.99), True),
  )
  def test_dense(self, input_shape, channels, momentum, training):
    B, _ = input_shape
    inputs = tf.random.uniform(input_shape)
    layer = PointNetDenseLayer(channels, momentum)
    outputs = layer(inputs, training=training)
    assert outputs.shape == (B, channels)

  @parameterized.parameters(
      ((32, 2048, 3), (.9), True),
      ((32, 2048, 2), (.5), False),
      ((32, 2048, 3), (.99), True),
  )
  def test_vanilla_encoder(self, input_shape, momentum, training):
    B = input_shape[0]
    inputs = tf.random.uniform(input_shape)
    encoder = VanillaEncoder(momentum)
    outputs = encoder(inputs, training=training)
    assert outputs.shape == (B, 1024)

  @parameterized.parameters(
      ((16, 1024), (20), (.9), True),
      ((8, 2048), (40), (.5), False),
      ((32, 512), (10), (.99), True),
  )
  def test_classification_head(self, input_shape, num_classes, momentum,
                               training):
    B = input_shape[0]
    inputs = tf.random.uniform(input_shape)
    head = ClassificationHead(num_classes, momentum)
    outputs = head(inputs, training=training)
    assert outputs.shape == (B, num_classes)

  @parameterized.parameters(
      ((32, 1024, 3), 40, True),
      ((32, 1024, 2), 40, False),
      ((16, 2048, 3), 20, True),
      ((16, 2048, 2), 20, False),
  )
  def test_vanilla_classifier(self, input_shape, num_classes, training):
    B = input_shape[0]
    C = num_classes
    inputs = tf.random.uniform(input_shape)
    model = PointNetVanillaClassifier(num_classes, momentum=.5)
    logits = model(inputs, training)
    assert logits.shape == (B, C)
    labels = tf.random.uniform((B,), minval=0, maxval=C, dtype=tf.int64)
    PointNetVanillaClassifier.loss(labels, logits)


if __name__ == "__main__":
  test_case.main()
