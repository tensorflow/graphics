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

These shorthands are used throughout this module:
  `B`: Number of elements in a batch.
  `N`: The number of points.
  `D`: Number of dimensions (2 for 2D, 3 for 3D).
  `C`: The number of feature channels.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class PointNetConv2Layer(Layer):
  """The 2D convolution layer used by the feature encoder in PointNet."""

  def __init__(self, channels, momentum):
    """Constructs a Conv2 layer.

    Note:
      Differently from the standard Keras Conv2 layer, the order of ops is:
      1. fully connected layer
      2. batch normalization layer
      3. ReLU activation unit
    
    Args:
      channels: the number of generated feature.
      momentum: the momentum of the batch normalization layer.
    """
    super(PointNetConv2Layer, self).__init__()
    self.channels = channels
    self.momentum = momentum

  def build(self, input_shape):
    """Builds the layer with a specified input_shape."""
    self.conv = layers.Conv2D(self.channels, (1, 1), input_shape=input_shape)
    self.bn = layers.BatchNormalization(momentum=self.momentum)

  # pylint: disable=arguments-differ
  def call(self, inputs, training):
    """Executes the convolution.

    Args:
      inputs: a dense tensor of size `[B, N, 1, D]`.
      training: flag to control batch normalization update statistics.

    Returns: 
      Tensor with shape `[B, N, 1, C]`.
    """
    return tf.nn.relu(self.bn(self.conv(inputs), training))


class PointNetDenseLayer(Layer):
  """The fully connected layer used by the classification head in pointnet.
  
  Note:
    Differently from the standard Keras Conv2 layer, the order of ops is:
      1. fully connected layer
      2. batch normalization layer
      3. ReLU activation unit
  """
  def __init__(self, channels, momentum):
    super(PointNetDenseLayer, self).__init__()
    self.momentum = momentum
    self.channels = channels

  def build(self, input_shape):
    """Builds the layer with a specified input_shape."""
    self.dense = layers.Dense(self.channels, input_shape=input_shape)
    self.bn = layers.BatchNormalization(momentum=self.momentum)

  # pylint: disable=arguments-differ
  def call(self, inputs, training):
    """Executes the convolution.

    Args:
      inputs: a dense tensor of size `[B, D]`.
      training: flag to control batch normalization update statistics.

    Returns: 
      Tensor with shape `[B, C]`.
    """
    return tf.nn.relu(self.bn(self.dense(inputs), training))


class VanillaEncoder(Layer):
  """The Vanilla PointNet feature encoder.
  
  Consists of five conv2 layers with (64,64,64,128,1024) output channels.

  Note:
    PointNetConv2Layer are used instead of tf.keras.layers.Conv2D.

  https://github.com/charlesq34/pointnet/blob/master/models/pointnet_cls_basic.py
  """

  def __init__(self, momentum=.5):
    """Constructs a VanillaEncoder keras layer.

    Args: 
      momentum: the momentum used for the batch normalization layer.
    """
    super(VanillaEncoder, self).__init__()
    self.conv1 = PointNetConv2Layer(64, momentum)
    self.conv2 = PointNetConv2Layer(64, momentum)
    self.conv3 = PointNetConv2Layer(64, momentum)
    self.conv4 = PointNetConv2Layer(128, momentum)
    self.conv5 = PointNetConv2Layer(1024, momentum)

  # pylint: disable=arguments-differ
  def call(self, inputs, training):
    """Computes the PointNet features. 

    Args:
      inputs: a dense tensor of size `[B, N, D]`.
      training: flag to control batch normalization update statistics.

    Returns:
      Tensor with shape `[B, N, C=1024]`
    """
    x = tf.expand_dims(inputs, axis=2) # BxNx1xD
    x = self.conv1(x, training)        # BxNx1x64
    x = self.conv2(x, training)        # BxNx1x64
    x = self.conv3(x, training)        # BxNx1x64
    x = self.conv4(x, training)        # BxNx1x128
    x = self.conv5(x, training)        # BxNx1x1024
    x = tf.math.reduce_max(x, axis=1)  # Bx1x1024
    return tf.squeeze(x)               # Bx1024


class ClassificationHead(tf.keras.layers.Layer):
  """The PointNet classification head.

  The classifier consists of 2x PointNetDenseLayer layers (with 512 and 256 channels),
  followed by a dropout layer (drop rate=30%) a dense linear layer producing the 
  logits of the num_classes classes. 

  Args:
    num_classes: the number of classes to classify
    momentum: the momentum used for the batch normalization layer.
  """
  def __init__(self, num_classes, momentum):
    super(ClassificationHead, self).__init__()
    self.dense1 = PointNetDenseLayer(512, momentum)
    self.dense2 = PointNetDenseLayer(256, momentum)
    self.dropout = layers.Dropout(0.3)
    self.dense3 = layers.Dense(num_classes, activation="linear")

  # pylint: disable=arguments-differ
  def call(self, inputs, training):
    """Computes the classiciation logits (without softmax).

    Args:
      inputs: a dense tensor of size `[B, D]`.
      training: flag to control batch normalization update statistics.

    Returns:
      Tensor with shape `[B, 1]`
    """
    x = self.dense1(inputs, training) # Bx512
    x = self.dense2(x, training)      # Bx256
    x = self.dropout(x, training)
    return self.dense3(x)             # Bx1


class PointNetVanillaClassifier(tf.keras.layers.Layer):
  # TODO documentation

  def __init__(self, num_classes, momentum=.5):
    super(PointNetVanillaClassifier, self).__init__()
    self.encoder = VanillaEncoder(momentum)
    self.classifier = ClassificationHead(num_classes, momentum)

  # pylint: disable=arguments-differ
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
