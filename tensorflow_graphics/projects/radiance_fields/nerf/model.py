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
"""NeRF models."""
from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.representation.ray as ray
import tensorflow_graphics.math.feature_representation as feature_rep
import tensorflow_graphics.projects.radiance_fields.nerf.layers as nerf_layers
import tensorflow_graphics.projects.radiance_fields.utils as utils
import tensorflow_graphics.rendering.volumetric.ray_radiance as ray_radiance


class NeRF:
  """Original NeRF network."""

  def __init__(self,
               ray_samples_coarse=128,
               ray_samples_fine=128,
               near=1.0,
               far=3.0,
               n_freq_posenc_xyz=8,
               n_freq_posenc_dir=4,
               scene_bbox=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
               n_filters=256,
               white_background=True,
               coarse_sampling_strategy="stratified"):

    # Ray parameters
    self.ray_samples_coarse = ray_samples_coarse
    self.ray_samples_fine = ray_samples_fine
    self.near = near
    self.far = far
    self.white_background = white_background
    # Network parameters
    self.n_freq_posenc_xyz = n_freq_posenc_xyz
    self.n_freq_posenc_dir = n_freq_posenc_dir

    scene_bbox = np.array(scene_bbox).reshape([2, 3])
    area_dims = scene_bbox[1, :] - scene_bbox[0, :]
    scene_scale = 1./(max(area_dims)/2.)
    scene_translation = -np.mean(scene_bbox, axis=0)
    self.scene_scale = scene_scale
    self.scene_transl = scene_translation

    self.n_filters = n_filters

    self.coarse_sampling_strategy = coarse_sampling_strategy

    self.xyz_dim = n_freq_posenc_xyz * 2 * 3 + 3
    self.dir_dim = n_freq_posenc_dir * 2 * 3 + 3
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

  def init_optimizer(self, learning_rate=0.0001, decay_steps=1000,
                     decay_rate=0.98, staircase=True):
    """Initialize the optimizers with a scheduler."""
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase)
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
    latest_checkpoint = self.manager.latest_checkpoint if checkpoint is None else checkpoint

    if latest_checkpoint is not None:
      logging.info("Checkpoint %s restored", latest_checkpoint)
      _ = self.checkpoint.restore(latest_checkpoint).expect_partial()
    else:
      logging.warning("No checkpoint was restored.")

  def get_model(self):
    """Constructs the original NeRF network as a keras model."""
    with tf.name_scope("Network/"):
      xyz_features = tf.keras.layers.Input(shape=[None, None, self.xyz_dim])
      dir_features = tf.keras.layers.Input(shape=[None, None, self.dir_dim])

      feat0 = nerf_layers.concat_block(xyz_features,
                                       n_filters=self.n_filters,
                                       n_layers=4)
      feat1 = nerf_layers.dense_block(feat0,
                                      n_filters=self.n_filters,
                                      n_layers=4)
      feat2 = tf.keras.layers.Dense(self.n_filters)(feat1)
      density = tf.keras.layers.Dense(1)(feat2)
      feat2_dir = tf.keras.layers.concatenate([feat2, dir_features], -1)
      feat3 = tf.keras.layers.Dense(self.n_filters//2)(feat2_dir)
      rgb = tf.keras.layers.Dense(3)(feat3)
      rgb_density = tf.keras.layers.concatenate([rgb, density], -1)
      return tf.keras.Model(inputs=[xyz_features, dir_features],
                            outputs=[rgb_density])

  @tf.function
  def prepare_positional_encoding(self, ray_points, ray_dirs):
    """Estimate the positional encoding of the 3D position and direction of the samples along a ray.

    Args:
      ray_points: A tensor of shape `[A, B, C, 3]` where A is the batch size, B
        is the number of rays, C is the number of samples per ray.
      ray_dirs: A tensor of shape `[A, B, 3]` where A is the batch size, B
      is the number of rays.

    Returns:
      A list containing a tensor of shape `[A, B, C, M]` and a tensor of shape
      `[A, B, C, N]`, where M is the dimensionality of the location positional
      encoding and N is dimensionality of the direction positional encoding.
    """
    n_ray_samples = tf.shape(ray_points)[-2]
    scaled_ray_points = self.scene_scale * (ray_points + self.scene_transl)
    features_xyz = feature_rep.positional_encoding(scaled_ray_points,
                                                   self.n_freq_posenc_xyz)
    ray_dirs = tf.tile(tf.expand_dims(ray_dirs, -2), [1, 1, n_ray_samples, 1])
    features_dir = feature_rep.positional_encoding(ray_dirs,
                                                   self.n_freq_posenc_dir)
    return features_xyz, features_dir

  @tf.function
  def render_network_output(self, rgb_density, ray_points):
    """Renders the network output into rgb and density values.

    Args:
      rgb_density: A tensor of shape `[A, B, C, 4]` where A is the batch size, B
        is the number of rays, C is the number of samples per ray.
      ray_points: A tensor of shape `[A, B, C, 3]`.

    Returns:
      A tensor of shape `[A, B, 3]` and a tensor of shape `[A, B, C]`.

    """
    rgb, density = tf.split(rgb_density, [3, 1], axis=-1)
    rgb = tf.sigmoid(rgb)
    density = tf.nn.relu(density)
    rgb_density = tf.concat([rgb, density], axis=-1)

    dists = utils.get_distances_between_points(ray_points)
    rgb_render, a_render, weights = ray_radiance.compute_radiance(rgb_density,
                                                                  dists)
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
    ray_points_coarse, z_vals_coarse = ray.sample_1d(
        r_org,
        r_dir,
        near=self.near,
        far=self.far,
        n_samples=self.ray_samples_coarse,
        strategy=self.coarse_sampling_strategy)
    posenc_features = self.prepare_positional_encoding(ray_points_coarse, r_dir)
    rgb_density = self.coarse_model(posenc_features)
    rgb_coarse, weights_coarse = self.render_network_output(rgb_density,
                                                            ray_points_coarse)
    depth_map_coarse = tf.reduce_sum(weights_coarse * z_vals_coarse, axis=-1)

    ray_points_fine, z_vals_fine = ray.sample_inverse_transform_stratified_1d(
        r_org,
        r_dir,
        z_vals_coarse,
        weights_coarse,
        n_samples=self.ray_samples_fine,
        combine_z_values=True)
    posenc_features = self.prepare_positional_encoding(ray_points_fine, r_dir)
    rgb_density = self.fine_model(posenc_features)
    rgb_fine, weights_fine = self.render_network_output(rgb_density,
                                                        ray_points_fine)
    depth_map_fine = tf.reduce_sum(weights_fine * z_vals_fine, axis=-1)
    return rgb_fine, rgb_coarse, depth_map_fine, depth_map_coarse

  @tf.function
  def train_step(self, r_org, r_dir, gt_rgb):
    """Training function for coarse and fine networks.

    Args:
      r_org: A tensor of shape `[B, N, 3]` where B is the batch size, N is the
        number of rays.
      r_dir: A tensor of shape `[B, N, 3]` where B is the batch size, N is the
        number of rays.
      gt_rgb: A tensor of shape `[B, N, 3]` where B is the batch size, N is the
        number of rays.

    Returns:
      A scalar.
    """
    with tf.GradientTape() as tape:
      rgb_fine, rgb_coarse, _, _ = self.inference(r_org, r_dir)
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
