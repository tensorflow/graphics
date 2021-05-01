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
"""Conditional NeRF model."""
from absl import logging
import tensorflow as tf

import tensorflow_graphics.geometry.representation.ray as ray
import tensorflow_graphics.math.feature_representation as feature_rep
import tensorflow_graphics.projects.radiance_fields.utils as utils
import tensorflow_graphics.rendering.volumetric.ray_radiance as ray_radiance

layers = tf.keras.layers


class NeRF:
  """Original NeRF network."""

  def __init__(self,
               ray_samples_coarse=128,
               ray_samples_fine=128,
               near=1.0,
               far=3.0,
               posenc_loc=8,
               posenc_dir=4,
               posenc_loc_scale=1.0,
               n_filters=256,
               white_background=True):

    # Ray parameters
    self.ray_samples_coarse = ray_samples_coarse
    self.ray_samples_fine = ray_samples_fine
    self.near = near
    self.far = far
    self.white_background = white_background
    # Network parameters
    self.posenc_loc = posenc_loc
    self.posenc_dir = posenc_dir
    self.posenc_loc_scale = posenc_loc_scale
    self.n_filters = n_filters

    self.input_dim = (posenc_loc * 2 * 3 + 3) + (posenc_dir * 2 * 3 + 3)
    self.coarse_model = None
    self.fine_model = None
    self.optimizer_network = None
    self.network_vars = None

    self.coarse_model_backup = None
    self.fine_model_backup = None

    self.latest_epoch = None
    self.global_step = None
    self.summary_writer = None
    self.checkpoint = None
    self.manager = None

  def init_coarse_and_fine_models(self):
    """Initialize models and variables."""
    self.coarse_model = self.get_model()
    self.fine_model = self.get_model()
    self.latest_epoch = tf.Variable(0, trainable=False, dtype=tf.int64)
    self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
    self.network_vars = (self.coarse_model.trainable_variables +
                         self.fine_model.trainable_variables)

  def init_optimizer(self, learning_rate=0.0001, learning_rate_decay=500):
    """Initialize the optimizers with a scheduler."""
    if learning_rate_decay > 0:
      learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
          learning_rate, decay_steps=learning_rate_decay*1000, decay_rate=0.1)
    self.optimizer_network = tf.keras.optimizers.Adam(
        learning_rate=learning_rate)

  def init_checkpoint(self, checkpoint_dir, checkpoint=None):
    """Initialize the checkpoints."""
    self.summary_writer = tf.summary.create_file_writer(checkpoint_dir)
    self.checkpoint = tf.train.Checkpoint(
        coarse_nerf=self.coarse_model,
        fine_nerf=self.fine_model,
        optimizer_network=self.optimizer_network,
        epoch=self.latest_epoch,
        global_step=self.global_step)
    self.manager = tf.train.CheckpointManager(
        checkpoint=self.checkpoint, directory=checkpoint_dir, max_to_keep=5)
    self.load_checkpoint(checkpoint=checkpoint)

  def load_checkpoint(self, checkpoint=None):
    """Load checkpoints."""
    if checkpoint is None:
      latest_checkpoint = self.manager.latest_checkpoint
    else:
      latest_checkpoint = checkpoint
    if latest_checkpoint is not None:
      logging.info("Checkpoint %s restored", latest_checkpoint)
      _ = self.checkpoint.restore(latest_checkpoint).expect_partial()

  def get_model(self):
    """Original NeRF network."""
    n_filters = self.n_filters
    with tf.name_scope("Network/"):
      input_features = layers.Input(shape=[self.input_dim])
      fc0 = layers.Dense(n_filters, activation=layers.ReLU())(input_features)
      fc1 = layers.Dense(n_filters, activation=layers.ReLU())(fc0)
      fc2 = layers.Dense(n_filters, activation=layers.ReLU())(fc1)
      fc3 = layers.Dense(n_filters, activation=layers.ReLU())(fc2)
      fc4 = layers.Dense(n_filters, activation=layers.ReLU())(fc3)
      fc4 = layers.concatenate([fc4, input_features], -1)
      fc5 = layers.Dense(n_filters, activation=layers.ReLU())(fc4)
      fc6 = layers.Dense(n_filters, activation=layers.ReLU())(fc5)
      fc7 = layers.Dense(n_filters, activation=layers.ReLU())(fc6)
      rgba = layers.Dense(4)(fc7)

      return tf.keras.Model(inputs=[input_features], outputs=[rgba])

  @tf.function
  def get_network_input(self, ray_points, ray_dir):
    """Estimate the features for input to the network (pos, dir).

    Args:
      ray_points: A tensor of shape `[A, B, C, 3]` where A is the batch size, B
        is the number of rays, C is the number of samples per ray.
      ray_dir: A tensor of shape `[A, B, 3]` where A is the batch size, B is the
        number of rays.

    Returns:
      A list of tensors of shape `[A*B*C, M]`.
    """
    features_xyz = feature_rep.positional_encoding(
        self.posenc_loc_scale * ray_points, self.posenc_loc)
    ray_dir = tf.tile(
        tf.expand_dims(ray_dir, -2), [1, 1, tf.shape(ray_points)[-2], 1])
    features_dir = feature_rep.positional_encoding(ray_dir, self.posenc_dir)
    features_in = tf.concat([features_xyz, features_dir], axis=-1)
    features_in = tf.reshape(features_in, [-1, tf.shape(features_in)[-1]])
    return [features_in]

  @tf.function
  def render_network_output(self, rgba, ray_points):
    """Renders the network output into rgb and density values.

    Args:
      rgba: A tensor of shape `[A*B*C, 4]` where A is the batch size, B
        is the number of rays, C is the number of samples per ray.
      ray_points: A tensor of shape `[A, B, C, 3]`.

    Returns:
      A tensor of shape `[A, B, 3]` and a tensor of shape `[A, B, C]`.

    """
    target_shape = tf.concat([tf.shape(ray_points)[:-1], [4]], axis=-1)
    rgba = tf.reshape(rgba, target_shape)
    rgb, alpha = tf.split(rgba, [3, 1], axis=-1)
    rgb = tf.sigmoid(rgb)
    alpha = tf.nn.relu(alpha)
    rgba = tf.concat([rgb, alpha], axis=-1)
    dists = utils.get_distances_between_points(ray_points)
    rgb_render, a_render, weights = ray_radiance.compute_radiance(rgba, dists)
    if self.white_background:
      rgb_render = rgb_render +  1 - a_render
    return rgb_render, weights

  @tf.function
  def inference(self, r_org, r_dir):
    """Run both coarse and fine networks for given rays.

    Args:
      r_org: A tensor of shape `[A, B, 3]` where A is the batch size, B is the
        number of rays.
      r_dir: A tensor of shape `[A, B, 3]` where A is the batch size, B is the
        number of rays.

    Returns:
      Two tensors of size `[A, B, 3]`.
    """
    ray_points_coarse, z_vals_coarse = ray.sample_stratified_1d(
        r_org,
        r_dir,
        near=self.near,
        far=self.far,
        n_samples=self.ray_samples_coarse)
    network_inputs = self.get_network_input(ray_points_coarse, r_dir)
    network_outputs = self.coarse_model(network_inputs)
    rgb_coarse, weights_coarse = self.render_network_output(network_outputs,
                                                            ray_points_coarse)

    ray_points_fine, z_vals_fine = ray.sample_inverse_transform_stratified_1d(
        r_org,
        r_dir,
        z_vals_coarse,
        weights_coarse,
        n_samples=self.ray_samples_fine,
        combine_z_values=True)
    network_inputs = self.get_network_input(ray_points_fine, r_dir)
    network_outputs = self.fine_model(network_inputs)
    rgb_fine, _ = self.render_network_output(network_outputs, ray_points_fine)

    with self.summary_writer.as_default():
      z_vals_fine = tf.reshape(z_vals_fine, [-1, self.ray_samples_fine])
      z_vals_coarse = tf.reshape(z_vals_coarse, [-1, self.ray_samples_coarse])
      tf.summary.histogram("z_vals_f", z_vals_fine, step=self.global_step)
      tf.summary.histogram("z_vals_c", z_vals_coarse, step=self.global_step)
    return rgb_fine, rgb_coarse

  @tf.function
  def train_step(self, r_org, r_dir, gt_rgb):
    """Training function for coarse and fine networks.

    Args:
      r_org: A tensor of shape `[A, B, 3]` where A is the batch size, B is the
        number of rays.
      r_dir: A tensor of shape `[A, B, 3]` where A is the batch size, B is the
        number of rays.
      gt_rgb: A tensor of shape `[A, B, 3]` where A is the batch size, B is the
        number of rays.

    Returns:
      A scalar.
    """
    with tf.GradientTape() as tape:
      rgb_fine, rgb_coarse = self.inference(r_org, r_dir)
      rgb_coarse_loss = utils.l2_loss(rgb_coarse, gt_rgb)
      rgb_fine_loss = utils.l2_loss(rgb_fine, gt_rgb)
      total_loss = rgb_coarse_loss + rgb_fine_loss
    gradients = tape.gradient(total_loss, self.network_vars)
    self.optimizer_network.apply_gradients(zip(gradients, self.network_vars))

    with self.summary_writer.as_default():
      step = self.global_step
      tf.summary.scalar("total_loss", total_loss, step=step)
      tf.summary.scalar("rgb_loss_f", rgb_fine_loss, step=step)
      tf.summary.scalar("rgb_loss_c", rgb_coarse_loss, step=step)
    return total_loss
