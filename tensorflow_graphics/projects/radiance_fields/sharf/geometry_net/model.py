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
"""Implementation of the geometry network."""
from absl import logging
import tensorflow as tf
import tensorflow_graphics.projects.radiance_fields.sharf.geometry_net.layers as geometry_layers
import tensorflow_graphics.projects.radiance_fields.sharf.voxel_functions as voxel_functions


layers = tf.keras.layers
initializer = tf.keras.initializers.glorot_normal()


class GeometryNetwork:
  """Learn to generate voxels from a latent code using GLO."""

  def __init__(self,
               n_latent_codes=4371,
               latent_code_dim=256,
               fc_channels=512,
               fc_activation='relu',
               conv_size=4,
               norm3d='batchnorm',
               bce_gamma=0.8,
               proj_weight=0.01,
               mirror_weight=1.0):

    self.n_latent_codes = n_latent_codes
    self.latent_code_dim = latent_code_dim
    self.fc_channels = fc_channels
    self.fc_activation = fc_activation
    self.conv_size = conv_size
    self.norm3d = norm3d
    self.model = None
    self.model_backup = None
    self.latent_code_vars = None
    self.network_vars = None
    self.global_step = None
    self.latest_epoch = None

    self.optimizer_network = None
    self.optimizer_latent = None

    self.checkpoint = None
    self.manager = None
    self.summary_writer = None

    self.bce_gamma = bce_gamma
    self.proj_weight = proj_weight
    self.mirror_weight = mirror_weight

    self.mask_voxels = voxel_functions.get_mask_voxels()

  def get_model(self):
    """Voxel GLO network."""
    fc_channels = self.fc_channels
    norm3d = self.norm3d

    if self.fc_activation == 'relu':
      activation = layers.ReLU()
    else:
      activation = None

    with tf.name_scope('Network/'):
      latent_code = layers.Input(shape=[self.latent_code_dim])

      with tf.name_scope('FC_layers'):
        fc0 = layers.Dense(fc_channels, activation=activation)(latent_code)
        fc1 = layers.Dense(fc_channels, activation=activation)(fc0)
        fc2 = layers.Dense(fc_channels, activation=activation)(fc1)
        fc2_as_volume = layers.Reshape([1, 1, 1, fc_channels])(fc2)

      with tf.name_scope('GLO_VoxelDecoder'):
        decoder_1 = geometry_layers.conv_t_block_3d(fc2_as_volume,
                                                    nfilters=32,
                                                    size=self.conv_size,
                                                    strides=2,
                                                    normalization=norm3d)  # 2
        decoder_2 = geometry_layers.conv_t_block_3d(decoder_1,
                                                    nfilters=32,
                                                    size=self.conv_size,
                                                    strides=2,
                                                    normalization=norm3d)  # 4
        decoder_3 = geometry_layers.conv_t_block_3d(decoder_2,
                                                    nfilters=32,
                                                    size=self.conv_size,
                                                    strides=2,
                                                    normalization=norm3d)  # 8
        decoder_4 = geometry_layers.conv_t_block_3d(decoder_3,
                                                    nfilters=16,
                                                    size=self.conv_size,
                                                    strides=2,
                                                    normalization=norm3d)  # 16
        decoder_5 = geometry_layers.conv_t_block_3d(decoder_4,
                                                    nfilters=8,
                                                    size=self.conv_size,
                                                    strides=2,
                                                    normalization=norm3d)  # 32
        decoder_6 = geometry_layers.conv_t_block_3d(decoder_5,
                                                    nfilters=4,
                                                    size=self.conv_size,
                                                    strides=2,
                                                    normalization=norm3d)  # 64
        volume_out = layers.Conv3DTranspose(1,
                                            self.conv_size,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)(decoder_6)  # 128
    return tf.keras.Model(inputs=[latent_code], outputs=[volume_out])

  def init_model(self):
    """Initialize models and variables."""
    self.model = self.get_model()
    self.model_backup = self.get_model()
    self.latest_epoch = tf.Variable(0, trainable=False, dtype=tf.int64)
    self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
    init_latent_code = tf.random.normal((self.n_latent_codes,
                                         self.latent_code_dim))
    self.latent_code_vars = tf.Variable(init_latent_code, trainable=True)
    self.network_vars = self.model.trainable_variables

  def init_optimizer(self, learning_rate=0.0001, decay_steps=100000,
                     decay_rate=0.96, staircase=True, latent_lr_factor=10):
    """Initialize the optimizers with a scheduler."""
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase)
    self.optimizer_network = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate * latent_lr_factor,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase)
    self.optimizer_latent = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

  def init_checkpoint(self, logdir, checkpoint=None):
    """Initialize the checkpoints."""
    self.summary_writer = tf.summary.create_file_writer(logdir)
    self.checkpoint = tf.train.Checkpoint(
        model=self.model,
        latent_code_var=self.latent_code_vars,
        optimizer_network=self.optimizer_network,
        optimizer_latent=self.optimizer_latent,
        epoch=self.latest_epoch,
        global_step=self.global_step)
    self.manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                              directory=logdir,
                                              max_to_keep=2)
    self.load_checkpoint(checkpoint=checkpoint)

  def load_checkpoint(self, checkpoint=None):
    """Load checkpoints."""
    if checkpoint is None:
      latest_checkpoint = self.manager.latest_checkpoint
    else:
      latest_checkpoint = checkpoint
    if latest_checkpoint is not None:
      logging.info('Checkpoint %s restored', latest_checkpoint)
      _ = self.checkpoint.restore(latest_checkpoint).expect_partial()
      for a, b in zip(self.model_backup.variables,
                      self.model.variables):
        a.assign(b)

  def reset_models(self):
    for a, b in zip(self.model.variables,
                    self.model_backup.variables):
      a.assign(b)

  def train_step(self):
    # TODO(krematas)
    pass
