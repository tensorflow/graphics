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
# python3
"""Create Hourglass network architecture."""
import gin
import tensorflow as tf

from tensorflow_graphics.projects.points_to_3Dobjects.networks import networks
import tensorflow_graphics.projects.points_to_3Dobjects.networks.custom_blocks as network_cb


class InputDownsampleBlock(tf.keras.layers.Layer):
  """Block for the initial feature downsampling.

  Attributes:
    input_shape: Tuple with the input shape i.e. (None, 512, 512, 3).
    out_dims: Integer tuple, number of filter in the first and second
      convolutions.
    norm_type: Object type of the norm to use.
  """

  def __init__(self, input_shape, out_dims, norm_type, kernel_regularizer):

    super(InputDownsampleBlock, self).__init__(name='InputDownsampleBlock')
    self.conv_block = network_cb.ConvolutionalBlock(
        7,
        out_dims[0],
        norm_type=norm_type,
        kernel_regularizer=kernel_regularizer,
        stride=2,
        input_shape=input_shape,
        padding='valid')

    self.residual_block = network_cb.ResidualBlock(
        out_dims[1], norm_type, kernel_regularizer, stride=2, skip=True)

  def call(self, inputs, **kwargs):
    x = self.conv_block(inputs)
    x = self.residual_block(x)
    return x


class EncoderDecoderBlock(tf.keras.layers.Layer):
  """Block that creates recursively the Encode-Decoder structure of one Hourglass.

    Attributes:
      depth: Integer, depth of the encoder
      features_dims: Integer list, feature dimensions in each depth.
      blocks_per_stage: Integer list, number of residual blocks in each stage.
      norm_type: Object type of the norm to use.
  """

  def __init__(self, depth, features_dims, blocks_per_stage, norm_type,
               kernel_regularizer):

    super(EncoderDecoderBlock, self).__init__(name='EncoderDecoderBlock')
    cur_feature_dim = features_dims[0]
    next_feature_dim = features_dims[1]

    self.encoder_block1 = self.make_encoder_hourglass_block(
        cur_feature_dim, norm_type, kernel_regularizer, blocks_per_stage[0])
    self.encoder_block2 = self.make_encoder_hourglass_block(
        next_feature_dim,
        norm_type,
        kernel_regularizer,
        blocks_per_stage[0],
        do_stride=True)

    if depth > 1:
      self.inner_block = [
          EncoderDecoderBlock(depth - 1, features_dims[1:],
                              blocks_per_stage[1:], norm_type,
                              kernel_regularizer)
      ]
    else:
      self.inner_block = self.make_encoder_hourglass_block(
          next_feature_dim, norm_type, kernel_regularizer, blocks_per_stage[1])

    self.decoder_block = self.make_decoder_hourglass_block(
        next_feature_dim, cur_feature_dim, norm_type, kernel_regularizer,
        blocks_per_stage[0])
    self.upsample = tf.keras.layers.UpSampling2D(2)

    self.merge_features = tf.keras.layers.Add()

  @staticmethod
  def make_encoder_hourglass_block(out_dim,
                                   norm_type,
                                   kernel_regularizer,
                                   num_blocks,
                                   do_stride=False):
    hourglass_block = []
    for ii in range(num_blocks):
      stride = 2 if ii == 0 and do_stride else 1
      hourglass_block.append(
          network_cb.ResidualBlock(
              out_dim,
              norm_type,
              kernel_regularizer,
              stride=stride,
              skip=stride == 2))
    return hourglass_block

  @staticmethod
  def make_decoder_hourglass_block(in_dim, out_dim, norm_type,
                                   kernel_regularizer, num_blocks):
    hourglass_block = []
    for _ in range(num_blocks - 1):
      hourglass_block.append(
          network_cb.ResidualBlock(in_dim, norm_type, kernel_regularizer))
    skip = in_dim != out_dim
    hourglass_block.append(
        network_cb.ResidualBlock(
            out_dim, norm_type, kernel_regularizer, skip=skip))
    return hourglass_block

  def call(self, inputs, **kwargs):
    x = inputs
    for block in self.encoder_block1:
      x = block(x)
    x_s = inputs
    for block in self.encoder_block2:
      x_s = block(x_s)
    for block in self.inner_block:
      x_s = block(x_s)
    for block in self.decoder_block:
      x_s = block(x_s)
    x_s = self.upsample(x_s)
    merge_features = self.merge_features([x, x_s])
    return merge_features


class Head(tf.keras.layers.Layer):
  """Block that creates a convolutional head.

    Attributes:
      in_dim: Integer, number of filter in the first convolution.
      dim: Integer, number of output classes.
      name: String, name of the block .
  """

  def __init__(self,
               in_dim,
               dim,
               norm_type,
               kernel_regularizer,
               name='head',
               return_features=False):

    super(Head, self).__init__(name=name)
    self.return_features = return_features
    self.pad = tf.keras.layers.ZeroPadding2D((1, 1))
    self.conv = tf.keras.layers.Conv2D(
        in_dim, 3, kernel_regularizer=kernel_regularizer())
    self.relu = tf.keras.layers.ReLU()
    self.out_conv = tf.keras.layers.Conv2D(
        dim, 1, kernel_regularizer=kernel_regularizer())
    if self.name == 'centers':
      self.out_conv.bias_initializer = tf.keras.initializers.constant(-2.19)

  def call(self, inputs, **kwargs):
    x = self.pad(inputs)
    x = self.conv(x)
    x_ = self.relu(x)
    x = self.out_conv(x_)
    if self.return_features:
      out_x = {self.name: x, f'{self.name}_features': x_}
    else:
      out_x = {self.name: x}
    return out_x


class ClassificationHead(tf.keras.layers.Layer):
  """Block that creates a classification head."""

  def __init__(self,
               norm_type,
               kernel_regularizer,
               channels_per_block=(256, 512, 1024),
               num_classes=1000,
               name='classification_head'):
    super(ClassificationHead, self).__init__(name=name)
    self._encoder = []
    self._logits = []
    for i in range(len(channels_per_block)):
      self._encoder.append(
          network_cb.ResidualBlock(
              channels_per_block[i],
              norm_type,
              kernel_regularizer,
              skip=True,
              stride=2))
    self._logits = tf.keras.layers.Dense(
        num_classes,
        kernel_initializer='zeros',
        kernel_regularizer=kernel_regularizer())

  def call(self, inputs, **kwargs):
    x = inputs
    for block in self._encoder:
      x = block(x)
    x = tf.reduce_mean(x, axis=[1, 2])
    return {self.name: self._logits(x)}


class Hourglass(tf.keras.Model):
  """Hourglass network architecture.

  Attributes:
    heads: Dict, keys are head names, value for each key is a dict with {'dim':
      dim} where dim is the output dimensionality of the last convolution.
    features_dims: Integer list, feature dimensionality at the output of the
      input downsampling convolution block and at the output of each encoder
      stage.
    number_hourglasses: Integer, number of hourglasses to stack.
    depth: Integer, depth of the encoder.
    blocks_per_stage: Integer list, number of residual blocks at each encoder
      stage.
    name: String, name of the block.
  """

  def __init__(self,
               input_shape=gin.REQUIRED,
               heads=None,
               features_dims=(128, 256, 256, 384, 384, 384, 512),
               number_hourglasses=2,
               depth=5,
               blocks_per_stage=(2, 2, 2, 2, 2, 4),
               kernel_regularization=None,
               return_output_features=False,
               name='hourglass'):
    super(Hourglass, self).__init__(name=name)
    norm_type = networks.NormType('batchnorm', {})
    kernel_regularizer = networks.Regularization(kernel_regularization)

    self.output_stride = 4
    self.heads = heads
    self.number_hourglasses = number_hourglasses
    self.return_output_features = return_output_features

    self.downsample_input = InputDownsampleBlock(input_shape, features_dims[:2],
                                                 norm_type, kernel_regularizer)
    self.hourglass_network = []
    self.output_conv = []
    for _ in range(self.number_hourglasses):
      self.hourglass_network.append(
          EncoderDecoderBlock(depth, features_dims[1:], blocks_per_stage,
                              norm_type, kernel_regularizer))
      self.output_conv.append(
          network_cb.ConvolutionalBlock(3, features_dims[1], norm_type,
                                        kernel_regularizer))
    self.intermediate_conv1 = []
    self.intermediate_conv2 = []
    self.intermediate_residual = []
    for _ in range(self.number_hourglasses - 1):
      self.intermediate_conv1.append(
          network_cb.ConvolutionalBlock(
              1, features_dims[1], norm_type, kernel_regularizer, relu=False))
      self.intermediate_conv2.append(
          network_cb.ConvolutionalBlock(
              1, features_dims[1], norm_type, kernel_regularizer, relu=False))
      self.intermediate_residual.append(
          network_cb.ResidualBlock(features_dims[1], norm_type,
                                   kernel_regularizer))
    self.intermediate_relu = tf.keras.layers.ReLU()

    if self.heads is not None:
      for name, val in self.heads.items():
        self.__setattr__(name, [])
        for _ in range(self.number_hourglasses):
          if 'classification' in name:
            self.__getattribute__(name).append(
                ClassificationHead(
                    norm_type, kernel_regularizer, name=name, **val))
          else:
            self.__getattribute__(name).append(
                Head(features_dims[1], name=name, norm_type=norm_type,
                     kernel_regularizer=kernel_regularizer, **val))

  def call(self, inputs, training=None, mask=None):

    x = self.downsample_input(inputs)
    output = []
    for ii in range(self.number_hourglasses):
      x_hourglass = self.hourglass_network[ii](x)

      x_out = self.output_conv[ii](x_hourglass)

      if self.heads is not None:
        output_heads = {}
        for name in self.heads:
          head = self.__getattribute__(name)[ii]
          head_out = head(x_out)
          output_heads = {**output_heads, **head_out}
        if self.return_output_features:
          output_heads['output_features'] = x_out
        new_output = output_heads
      else:
        new_output = {'output_features': x_out}
      output.append(new_output)

      if ii < self.number_hourglasses - 1:
        x = self.intermediate_conv1[ii](x) + self.intermediate_conv2[ii](x_out)
        x = self.intermediate_relu(x)

        intermediate_residual = self.intermediate_residual[ii]
        x = intermediate_residual(x)

    return output


def _top_scores_heatmaps(centers, k):
  """Get top scores from heatmaps."""
  b, h, w, c = centers.shape
  centers_t = tf.transpose(centers, [0, 3, 1, 2])
  scores, indices = tf.math.top_k(tf.reshape(centers_t, [b, c, -1]), k)
  topk_inds = indices % (h * w)
  topk_ys = tf.cast(tf.cast((indices / w), tf.int32), tf.float32)
  topk_xs = tf.cast(tf.cast((indices % w), tf.int32), tf.float32)
  scores, indices = tf.math.top_k(tf.reshape(scores, [b, -1]), k)
  topk_classes = tf.cast(indices / k, tf.int32)
  topk_inds = tf.gather(
      tf.reshape(topk_inds, [b, -1]), indices, axis=1, batch_dims=1)
  ys = tf.gather(tf.reshape(topk_ys, [b, -1]), indices, axis=1, batch_dims=1)
  xs = tf.gather(tf.reshape(topk_xs, [b, -1]), indices, axis=1, batch_dims=1)

  return xs, ys, topk_classes, topk_inds, scores


def _get_offsets(offset, topk_inds):
  b, _, _, n = offset.shape
  offset = tf.reshape(offset, [b, -1, n])
  offset = tf.gather(offset, topk_inds, batch_dims=1)
  return offset


def hourglass_104(*args, **kwargs):
  return Hourglass(*args, **kwargs)


if __name__ == '__main__':
  print('hellow')
