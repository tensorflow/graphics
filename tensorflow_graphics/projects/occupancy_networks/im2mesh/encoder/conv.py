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
""" NO COMMENT NOW"""

import tensorflow as tf
from im2mesh.common import normalize_imagenet


class ConvEncoder(tf.keras.Model):
  r''' Simple convolutional encoder network.

  It consists of 5 convolutional layers, each downsampling the input by a
  factor of 2, and a final fully-connected layer projecting the output to
  c_dim dimenions.

  Args:
      c_dim (int): output dimension of latent embedding
  '''

  def __init__(self, c_dim=128):
    super().__init__()
    self.conv0 = tf.keras.layers.Conv2D(32, 3, strides=2)
    self.conv1 = tf.keras.layers.Conv2D(64, 3, strides=2)
    self.conv2 = tf.keras.layers.Conv2D(128, 3, strides=2)
    self.conv3 = tf.keras.layers.Conv2D(256, 3, strides=2)
    self.conv4 = tf.keras.layers.Conv2D(512, 3, strides=2)
    self.fc_out = tf.keras.layers.Dense(c_dim)
    self.actvn = tf.keras.layers.ReLU()

  def call(self, x):
    batch_size = x.shape[0]

    net = self.conv0(x)
    net = self.conv1(self.actvn(net))
    net = self.conv2(self.actvn(net))
    net = self.conv3(self.actvn(net))
    net = self.conv4(self.actvn(net))
    # net = net.view(batch_size, 512, -1).mean(2)
    # net = tf.reduce_mean(tf.reshape(net, [batch_size, 512, -1]), axis=-1)
    net = tf.reshape(net, [batch_size, 512, -1])
    net = tf.math.reduce_mean(net, axis=-1)

    out = self.fc_out(self.actvn(net))

    return out


class BasicBlock(tf.keras.layers.Layer):
  r'''
  Args:
  '''
  # TODO:

  def __init__(self, filter_num, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                        kernel_size=(3, 3),
                                        strides=stride,
                                        padding="same")
    self.bn1 = tf.keras.layers.BatchNormalization(
        momentum=0.1, epsilon=1e-05)

    self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                        kernel_size=(3, 3),
                                        strides=1,
                                        padding="same")
    self.bn2 = tf.keras.layers.BatchNormalization(
        momentum=0.1, epsilon=1e-05)

    if stride != 1:
      self.downsample = tf.keras.Sequential()
      self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                 kernel_size=(1, 1),
                                                 strides=stride))
      self.downsample.add(tf.keras.layers.BatchNormalization(
          momentum=0.1, epsilon=1e-05))
    else:
      self.downsample = lambda x: x

  def call(self, inputs, training=False, **kwargs):
    residual = self.downsample(inputs)

    x = self.conv1(inputs)
    x = self.bn1(x, training=training)
    x = tf.nn.relu(x)
    x = self.conv2(x)
    x = self.bn2(x, training=training)

    output = tf.nn.relu(tf.keras.layers.add([residual, x]))

    return output


def make_basic_block_layer(filter_num, blocks, stride=1):
  res_block = tf.keras.Sequential()
  res_block.add(BasicBlock(filter_num, stride=stride))

  for _ in range(1, blocks):
    res_block.add(BasicBlock(filter_num, stride=1))

  return res_block


class Resnet18(tf.keras.Model):
  r''' ResNet-18 encoder network for image input.
  Args:
      c_dim (int): output dimension of the latent embedding
      normalize (bool): whether the input images should be normalized
      use_linear (bool): whether a final linear layer should be used
  '''

  def __init__(self, c_dim, normalize=True, use_linear=True):
    super().__init__()
    self.normalize = normalize
    self.use_linear = use_linear

    self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                        kernel_size=(7, 7),
                                        strides=2,
                                        padding="same")
    self.bn1 = tf.keras.layers.BatchNormalization(
        momentum=0.1, epsilon=1e-05)

    self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                           strides=2,
                                           padding="same")

    self.layer1 = make_basic_block_layer(filter_num=64,
                                         blocks=2)
    self.layer2 = make_basic_block_layer(filter_num=128,
                                         blocks=2,
                                         stride=2)
    self.layer3 = make_basic_block_layer(filter_num=256,
                                         blocks=2,
                                         stride=2)
    self.layer4 = make_basic_block_layer(filter_num=512,
                                         blocks=2,
                                         stride=2)

    self.avgpool = tf.keras.layers.GlobalAveragePooling2D()

    if use_linear:
      self.fc = tf.keras.layers.Dense(c_dim)
    elif c_dim == 512:
      # self.fc = nn.Sequential() # original
      self.fc = tf.keras.Sequential()  # CHECK
    else:
      raise ValueError('c_dim must be 512 if use_linear is False')

  def call(self, x, training=False):
    if self.normalize:
      x = normalize_imagenet(x)
    x = self.conv1(x)
    x = self.bn1(x, training=training)
    x = tf.nn.relu(x)
    x = self.pool1(x)
    x = self.layer1(x, training=training)
    x = self.layer2(x, training=training)
    x = self.layer3(x, training=training)
    x = self.layer4(x, training=training)
    x = self.avgpool(x)
    out = self.fc(x)
    return out

# class Resnet34(nn.Module):
#     r''' ResNet-34 encoder network.

#     Args:
#         c_dim (int): output dimension of the latent embedding
#         normalize (bool): whether the input images should be normalized
#         use_linear (bool): whether a final linear layer should be used
#     '''
#     def __init__(self, c_dim, normalize=True, use_linear=True):
#         super().__init__()
#         self.normalize = normalize
#         self.use_linear = use_linear
#         self.features = models.resnet34(pretrained=True)
#         self.features.fc = nn.Sequential()
#         if use_linear:
#             self.fc = nn.Linear(512, c_dim)
#         elif c_dim == 512:
#             self.fc = nn.Sequential()
#         else:
#             raise ValueError('c_dim must be 512 if use_linear is False')

#     def forward(self, x):
#         if self.normalize:
#             x = normalize_imagenet(x)
#         net = self.features(x)
#         out = self.fc(net)
#         return out


class Resnet50(tf.keras.Model):
  r''' ResNet-50 encoder network.

  Args:
      c_dim (int): output dimension of the latent embedding
      normalize (bool): whether the input images should be normalized
      use_linear (bool): whether a final linear layer should be used
  '''

  def __init__(self, c_dim, normalize=True, use_linear=True):
    super().__init__()
    self.normalize = normalize
    self.use_linear = use_linear
    self.features = tf.keras.applications.ResNet50(
        include_top=False)  # feature_extractor

    if use_linear:
      self.fc = tf.keras.layers.Dense(c_dim)
    elif c_dim == 2048:
      # self.fc = nn.Sequential() # original
      self.fc = tf.keras.Sequential()  # CHECK
    else:
      raise ValueError('c_dim must be 2048 if use_linear is False')

    print("resnet50")

  def call(self, x, training=False):
    if self.normalize:
      x = normalize_imagenet(x)
    net = self.features(x, training=training)
    out = self.fc(net)
    return out


class Resnet101(tf.keras.Model):
  r''' ResNet-101 encoder network.

  Args:
      c_dim (int): output dimension of the latent embedding
      normalize (bool): whether the input images should be normalized
      use_linear (bool): whether a final linear layer should be used
  '''

  def __init__(self, c_dim, normalize=True, use_linear=True):
    super().__init__()
    self.normalize = normalize
    self.use_linear = use_linear
    self.features = tf.keras.applications.ResNet50(
        include_top=False)  # feature_extractor

    if use_linear:
      self.fc = tf.keras.layers.Dense(c_dim)
    elif c_dim == 2048:
      # self.fc = nn.Sequential() # original
      self.fc = tf.keras.Sequential()  # CHECK
    else:
      raise ValueError('c_dim must be 2048 if use_linear is False')

  def call(self, x, training=False):
    if self.normalize:
      x = normalize_imagenet(x)
    net = self.features(x, training=training)
    out = self.fc(net)
    return out
