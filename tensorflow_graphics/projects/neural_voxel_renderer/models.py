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
"""Definition of NVR+ keras model."""
import tensorflow.compat.v1 as tf
import tensorflow_graphics.projects.neural_voxel_renderer.layers as layer_utils

initializer = tf.keras.initializers.glorot_normal()
layers = tf.keras.layers


def unet_3x_with_res_in_mid(feat_in, out_filters, norm2d):
  """Helper function of a Unet with res blocks in the middle."""
  e1 = layer_utils.residual_block_2d(feat_in,
                                     nfilters=128,
                                     strides=(2, 2),
                                     normalization=norm2d)  # 16x128
  e2 = layer_utils.residual_block_2d(e1,
                                     nfilters=256,
                                     strides=(2, 2),
                                     normalization=norm2d)  # 8x256
  e3 = layer_utils.residual_block_2d(e2,
                                     nfilters=512,
                                     strides=(2, 2),
                                     normalization=norm2d)  # 4x512

  mid1 = layer_utils.residual_block_2d(e3,
                                       nfilters=512,
                                       strides=(1, 1),
                                       normalization=norm2d)  # 32x32xnf_2d
  mid2 = layer_utils.residual_block_2d(mid1,
                                       nfilters=512,
                                       strides=(1, 1),
                                       normalization=norm2d)  # 32x32xnf_2d
  mid3 = layer_utils.residual_block_2d(mid2,
                                       nfilters=512,
                                       strides=(1, 1),
                                       normalization=norm2d)  # 32x32xnf_2d
  d0 = layer_utils.upconv(mid3,
                          nfilters=256,
                          size=4,
                          strides=1)  # 8x256
  d1 = layers.concatenate([d0, e2])  # 8x512
  d2 = layers.Conv2D(256,
                     kernel_size=4,
                     strides=(1, 1),
                     padding='same',
                     kernel_initializer=initializer)(d1)  # 8x256

  d3 = layer_utils.upconv(d2,
                          nfilters=128,
                          size=4,
                          strides=1)  # 16x128
  d4 = layers.concatenate([d3, e1])  # 16x256
  d5 = layers.Conv2D(128,
                     kernel_size=4,
                     strides=(1, 1),
                     padding='same',
                     kernel_initializer=initializer)(d4)  # 8x256

  d6 = layer_utils.upconv(d5,
                          nfilters=64,
                          size=4,
                          strides=1)  # 32x64
  d7 = layers.concatenate([d6, feat_in])  # 32xN
  d8 = layers.Conv2D(out_filters,
                     kernel_size=4,
                     strides=(1, 1),
                     padding='same',
                     kernel_initializer=initializer)(d7)  # 32xout

  return d8


def neural_voxel_renderer_plus(voxels,
                               rerendering,
                               light_pos,
                               size=4,
                               norm2d='batchnorm',
                               norm3d='batchnorm'):
  """Neural Voxel Renderer + keras model."""
  with tf.name_scope('Network/'):

    voxels = layers.Input(tensor=voxels)
    rerendering = layers.Input(tensor=rerendering)
    light_pos = layers.Input(tensor=light_pos)

    nf_2d = 512

    with tf.name_scope('VoxelProcessing'):
      vol0_a = layer_utils.conv_block_3d(voxels,
                                         nfilters=16,
                                         size=size,
                                         strides=2,
                                         normalization=norm3d)  # 64x64x64x16
      vol0_b = layer_utils.conv_block_3d(vol0_a,
                                         nfilters=16,
                                         size=size,
                                         strides=1,
                                         normalization=norm3d)  # 64x64x64x16
      vol1_a = layer_utils.conv_block_3d(vol0_b,
                                         nfilters=16,
                                         size=size,
                                         strides=2,
                                         normalization=norm3d)  # 32x32x32x16
      vol1_b = layer_utils.conv_block_3d(vol1_a,
                                         nfilters=32,
                                         size=size,
                                         strides=1,
                                         normalization=norm3d)  # 32x32x32x32
      vol1_c = layer_utils.conv_block_3d(vol1_b,
                                         nfilters=32,
                                         size=size,
                                         strides=1,
                                         normalization=norm3d)  # 32x32x32x32
      shortcut = vol1_c
      vol_a1 = layer_utils.residual_block_3d(vol1_c,
                                             32,
                                             strides=(1, 1, 1),
                                             normalization=norm3d)  # 32x
      vol_a2 = layer_utils.residual_block_3d(vol_a1,
                                             32,
                                             strides=(1, 1, 1),
                                             normalization=norm3d)  # 32x
      vol_a3 = layer_utils.residual_block_3d(vol_a2,
                                             32,
                                             strides=(1, 1, 1),
                                             normalization=norm3d)  # 32x
      vol_a4 = layer_utils.residual_block_3d(vol_a3,
                                             32,
                                             strides=(1, 1, 1),
                                             normalization=norm3d)  # 32x
      vol_a5 = layer_utils.residual_block_3d(vol_a4,
                                             32,
                                             strides=(1, 1, 1),
                                             normalization=norm3d)  # 32x
      encoded_vol = layers.add([shortcut, vol_a5])
      encoded_vol = layers.Reshape([32, 32, 32*32])(encoded_vol)
      encoded_vol = layers.Conv2D(nf_2d,
                                  kernel_size=1,
                                  strides=(1, 1),
                                  padding='same',
                                  kernel_initializer=initializer)(encoded_vol)
      latent_projection = layers.LeakyReLU()(encoded_vol)  # 32x32x512

    with tf.name_scope('ProjectionProcessing'):
      shortcut = latent_projection  # 32x32xnf_2d
      e1 = layer_utils.residual_block_2d(latent_projection,
                                         nfilters=nf_2d,
                                         strides=(1, 1),
                                         normalization=norm2d)  # 32x32xnf_2d
      e2 = layer_utils.residual_block_2d(e1,
                                         nfilters=nf_2d,
                                         strides=(1, 1),
                                         normalization=norm2d)  # 32x32xnf_2d
      e3 = layer_utils.residual_block_2d(e2,
                                         nfilters=nf_2d,
                                         strides=(1, 1),
                                         normalization=norm2d)  # 32x32xnf_2d
      e4 = layer_utils.residual_block_2d(e3,
                                         nfilters=nf_2d,
                                         strides=(1, 1),
                                         normalization=norm2d)  # 32x32xnf_2d
      e5 = layer_utils.residual_block_2d(e4,
                                         nfilters=nf_2d,
                                         strides=(1, 1),
                                         normalization=norm2d)  # 32x32xnf_2d
      encoded_proj = layers.add([shortcut, e5])  # 32x32xnf_2d

    with tf.name_scope('LightProcessing'):
      fc_light = layers.Dense(64, kernel_initializer=initializer)(light_pos)
      light_code = layers.Dense(64, kernel_initializer=initializer)(fc_light)
      light_code = \
        layers.Lambda(lambda v: tf.tile(v[0], [1, 32*32]))([light_code])
      light_code = layers.Reshape((32, 32, 64))(light_code)  # 32x32x64

    with tf.name_scope('Merger'):
      latent_code_final = layers.concatenate([encoded_proj, light_code])
      latent_code_final = layer_utils.conv_block_2d(latent_code_final,
                                                    nfilters=nf_2d,
                                                    size=size,
                                                    strides=1,
                                                    normalization=norm3d)
      shortcut = latent_code_final
      m1 = layer_utils.residual_block_2d(latent_code_final,
                                         nfilters=nf_2d,
                                         strides=(1, 1),
                                         normalization=norm2d)  # 32x32xnf_2d
      m2 = layer_utils.residual_block_2d(m1,
                                         nfilters=nf_2d,
                                         strides=(1, 1),
                                         normalization=norm2d)  # 32x32xnf_2d
      m3 = layer_utils.residual_block_2d(m2,
                                         nfilters=nf_2d,
                                         strides=(1, 1),
                                         normalization=norm2d)  # 32x32xnf_2d
      m4 = layer_utils.residual_block_2d(m3,
                                         nfilters=nf_2d,
                                         strides=(1, 1),
                                         normalization=norm2d)  # 32x32xnf_2d
      m5 = layer_utils.residual_block_2d(m4,
                                         nfilters=nf_2d,
                                         strides=(1, 1),
                                         normalization=norm2d)  # 32x32xnf_2d

      latent_code_final2 = layers.add([shortcut, m5])  # 32x32xnf_2d

    with tf.name_scope('Decoder'):
      d7 = layer_utils.conv_t_block_2d(latent_code_final2,
                                       nfilters=128,
                                       size=size,
                                       strides=2,
                                       normalization=norm2d)  # 64x64x128
      d7 = layer_utils.conv_block_2d(d7,
                                     nfilters=128,
                                     size=size,
                                     strides=1,
                                     normalization=norm2d)  # 64x64x128
      d8 = layer_utils.conv_t_block_2d(d7,
                                       nfilters=64,
                                       size=size,
                                       strides=2,
                                       normalization=norm2d)  # 128x128x64
      d8 = layer_utils.conv_block_2d(d8,
                                     nfilters=64,
                                     size=size,
                                     strides=1,
                                     normalization=norm2d)  # 128x128x64
      d9 = layer_utils.conv_t_block_2d(d8,
                                       nfilters=32,
                                       size=size,
                                       strides=2,
                                       normalization=norm2d)  # 256x256x32
      d9 = layer_utils.conv_block_2d(d9,
                                     nfilters=32,
                                     size=size,
                                     strides=1,
                                     normalization=norm2d)  # 256x256x32
      rendered_image = layers.Conv2D(32,
                                     size,
                                     strides=1,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     use_bias=False)(d9)  # 256x256x3

    with tf.name_scope('ImageProcessingNetwork'):
      ec1 = layer_utils.conv_block_2d(rerendering,
                                      nfilters=32,
                                      size=size,
                                      strides=1,
                                      normalization=norm2d)  # 256x
      ec2 = layer_utils.conv_block_2d(ec1,
                                      nfilters=32,
                                      size=size,
                                      strides=1,
                                      normalization=norm2d)  # 256x

    with tf.name_scope('NeuralRerenderingNetwork'):
      latent_img = layers.add([rendered_image, ec2])
      target_code = unet_3x_with_res_in_mid(latent_img, 32, norm2d=norm2d)
      out0 = layer_utils.conv_block_2d(target_code,
                                       nfilters=32,
                                       size=size,
                                       strides=1,
                                       normalization=norm2d)  # 256x
      predicted_image = layers.Conv2D(3,
                                      size,
                                      strides=1,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False)(out0)  # 256x256x3

    return tf.keras.Model(inputs=[voxels, rerendering, light_pos],
                          outputs=[predicted_image])


def neural_voxel_renderer_plus_tf2(size=4,
                                   norm2d='batchnorm',
                                   norm3d='batchnorm'):
  """Neural Voxel Renderer + keras model for tf2."""
  with tf.name_scope('Network/'):

    voxels = layers.Input(shape=[128, 128, 128, 4])
    rerendering = layers.Input(shape=[256, 256, 3])
    light_pos = layers.Input(shape=[3])

    nf_2d = 512

    with tf.name_scope('VoxelProcessing'):
      vol0_a = layer_utils.conv_block_3d(voxels,
                                         nfilters=16,
                                         size=size,
                                         strides=2,
                                         normalization=norm3d)  # 64x64x64x16
      vol0_b = layer_utils.conv_block_3d(vol0_a,
                                         nfilters=16,
                                         size=size,
                                         strides=1,
                                         normalization=norm3d)  # 64x64x64x16
      vol1_a = layer_utils.conv_block_3d(vol0_b,
                                         nfilters=16,
                                         size=size,
                                         strides=2,
                                         normalization=norm3d)  # 32x32x32x16
      vol1_b = layer_utils.conv_block_3d(vol1_a,
                                         nfilters=32,
                                         size=size,
                                         strides=1,
                                         normalization=norm3d)  # 32x32x32x32
      vol1_c = layer_utils.conv_block_3d(vol1_b,
                                         nfilters=32,
                                         size=size,
                                         strides=1,
                                         normalization=norm3d)  # 32x32x32x32
      shortcut = vol1_c
      vol_a1 = layer_utils.residual_block_3d(vol1_c,
                                             32,
                                             strides=(1, 1, 1),
                                             normalization=norm3d)  # 32x
      vol_a2 = layer_utils.residual_block_3d(vol_a1,
                                             32,
                                             strides=(1, 1, 1),
                                             normalization=norm3d)  # 32x
      vol_a3 = layer_utils.residual_block_3d(vol_a2,
                                             32,
                                             strides=(1, 1, 1),
                                             normalization=norm3d)  # 32x
      vol_a4 = layer_utils.residual_block_3d(vol_a3,
                                             32,
                                             strides=(1, 1, 1),
                                             normalization=norm3d)  # 32x
      vol_a5 = layer_utils.residual_block_3d(vol_a4,
                                             32,
                                             strides=(1, 1, 1),
                                             normalization=norm3d)  # 32x
      encoded_vol = layers.add([shortcut, vol_a5])
      encoded_vol = layers.Reshape([32, 32, 32*32])(encoded_vol)
      encoded_vol = layers.Conv2D(nf_2d,
                                  kernel_size=1,
                                  strides=(1, 1),
                                  padding='same',
                                  kernel_initializer=initializer)(encoded_vol)
      latent_projection = layers.LeakyReLU()(encoded_vol)  # 32x32x512

    with tf.name_scope('ProjectionProcessing'):
      shortcut = latent_projection  # 32x32xnf_2d
      e1 = layer_utils.residual_block_2d(latent_projection,
                                         nfilters=nf_2d,
                                         strides=(1, 1),
                                         normalization=norm2d)  # 32x32xnf_2d
      e2 = layer_utils.residual_block_2d(e1,
                                         nfilters=nf_2d,
                                         strides=(1, 1),
                                         normalization=norm2d)  # 32x32xnf_2d
      e3 = layer_utils.residual_block_2d(e2,
                                         nfilters=nf_2d,
                                         strides=(1, 1),
                                         normalization=norm2d)  # 32x32xnf_2d
      e4 = layer_utils.residual_block_2d(e3,
                                         nfilters=nf_2d,
                                         strides=(1, 1),
                                         normalization=norm2d)  # 32x32xnf_2d
      e5 = layer_utils.residual_block_2d(e4,
                                         nfilters=nf_2d,
                                         strides=(1, 1),
                                         normalization=norm2d)  # 32x32xnf_2d
      encoded_proj = layers.add([shortcut, e5])  # 32x32xnf_2d

    with tf.name_scope('LightProcessing'):
      fc_light = layers.Dense(64, kernel_initializer=initializer)(light_pos)
      light_code = layers.Dense(64, kernel_initializer=initializer)(fc_light)
      light_code = \
        layers.Lambda(lambda v: tf.tile(v[0], [1, 32*32]))([light_code])
      light_code = layers.Reshape((32, 32, 64))(light_code)  # 32x32x64

    with tf.name_scope('Merger'):
      latent_code_final = layers.concatenate([encoded_proj, light_code])
      latent_code_final = layer_utils.conv_block_2d(latent_code_final,
                                                    nfilters=nf_2d,
                                                    size=size,
                                                    strides=1,
                                                    normalization=norm3d)
      shortcut = latent_code_final
      m1 = layer_utils.residual_block_2d(latent_code_final,
                                         nfilters=nf_2d,
                                         strides=(1, 1),
                                         normalization=norm2d)  # 32x32xnf_2d
      m2 = layer_utils.residual_block_2d(m1,
                                         nfilters=nf_2d,
                                         strides=(1, 1),
                                         normalization=norm2d)  # 32x32xnf_2d
      m3 = layer_utils.residual_block_2d(m2,
                                         nfilters=nf_2d,
                                         strides=(1, 1),
                                         normalization=norm2d)  # 32x32xnf_2d
      m4 = layer_utils.residual_block_2d(m3,
                                         nfilters=nf_2d,
                                         strides=(1, 1),
                                         normalization=norm2d)  # 32x32xnf_2d
      m5 = layer_utils.residual_block_2d(m4,
                                         nfilters=nf_2d,
                                         strides=(1, 1),
                                         normalization=norm2d)  # 32x32xnf_2d

      latent_code_final2 = layers.add([shortcut, m5])  # 32x32xnf_2d

    with tf.name_scope('Decoder'):
      d7 = layer_utils.conv_t_block_2d(latent_code_final2,
                                       nfilters=128,
                                       size=size,
                                       strides=2,
                                       normalization=norm2d)  # 64x64x128
      d7 = layer_utils.conv_block_2d(d7,
                                     nfilters=128,
                                     size=size,
                                     strides=1,
                                     normalization=norm2d)  # 64x64x128
      d8 = layer_utils.conv_t_block_2d(d7,
                                       nfilters=64,
                                       size=size,
                                       strides=2,
                                       normalization=norm2d)  # 128x128x64
      d8 = layer_utils.conv_block_2d(d8,
                                     nfilters=64,
                                     size=size,
                                     strides=1,
                                     normalization=norm2d)  # 128x128x64
      d9 = layer_utils.conv_t_block_2d(d8,
                                       nfilters=32,
                                       size=size,
                                       strides=2,
                                       normalization=norm2d)  # 256x256x32
      d9 = layer_utils.conv_block_2d(d9,
                                     nfilters=32,
                                     size=size,
                                     strides=1,
                                     normalization=norm2d)  # 256x256x32
      rendered_image = layers.Conv2D(32,
                                     size,
                                     strides=1,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     use_bias=False)(d9)  # 256x256x3

    with tf.name_scope('ImageProcessingNetwork'):
      ec1 = layer_utils.conv_block_2d(rerendering,
                                      nfilters=32,
                                      size=size,
                                      strides=1,
                                      normalization=norm2d)  # 256x
      ec2 = layer_utils.conv_block_2d(ec1,
                                      nfilters=32,
                                      size=size,
                                      strides=1,
                                      normalization=norm2d)  # 256x

    with tf.name_scope('NeuralRerenderingNetwork'):
      latent_img = layers.add([rendered_image, ec2])
      target_code = unet_3x_with_res_in_mid(latent_img, 32, norm2d=norm2d)
      out0 = layer_utils.conv_block_2d(target_code,
                                       nfilters=32,
                                       size=size,
                                       strides=1,
                                       normalization=norm2d)  # 256x
      predicted_image = layers.Conv2D(3,
                                      size,
                                      strides=1,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False)(out0)  # 256x256x3

    return tf.keras.Model(inputs=[voxels, rerendering, light_pos],
                          outputs=[predicted_image])
