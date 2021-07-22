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
"""Functions to train a nerf network."""
from absl import logging
import numpy as np

import tensorflow as tf
import tensorflow_graphics.geometry.representation.ray as ray
import tensorflow_graphics.math.feature_representation as feature_rep
import tensorflow_graphics.projects.radiance_fields.nerf.layers as nerf_layers
import tensorflow_graphics.projects.radiance_fields.sharf.voxel_functions as voxel_functions
import tensorflow_graphics.projects.radiance_fields.utils as utils
import tensorflow_graphics.rendering.volumetric.ray_radiance as ray_radiance


class AppearanceNetwork:
  """Shape and appearance conditioned network for radiance fields."""

  def __init__(self,
               ray_samples_coarse=128,
               ray_samples_fine=128,
               near=1.0,
               far=3.0,
               n_freq_posenc_xyz=8,
               n_freq_posenc_dir=0,
               scene_bbox=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
               n_filters=256,
               num_latent_codes=4371,
               latent_code_dim=512,
               white_background=True,
               coarse_sampling_strategy="stratified"):

    # Ray parameters
    self.ray_samples_coarse = ray_samples_coarse
    self.ray_samples_fine = ray_samples_fine
    self.near = near
    self.far = far
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

    self.white_background = white_background

    # Latent Code parameters
    self.latent_code_dim = latent_code_dim
    self.num_latent_codes = num_latent_codes

    self.model = None
    self.model_backup = None
    self.latent_code_vars = None
    self.optimizer_network = None
    self.optimizer_latent = None
    self.network_vars = None

    self.latest_epoch = None
    self.global_step = None
    self.summary_writer = None
    self.checkpoint = None
    self.manager = None

    self.xyz_dim = n_freq_posenc_xyz * 2 * 3 + 3
    self.dir_dim = n_freq_posenc_dir * 2 * 3 + 3
    self.input_dim = self.xyz_dim + self.dir_dim + self.latent_code_dim + 1

  def init_model_and_codes(self):
    """Initialize models and variables."""
    self.model = self.get_model()
    self.model_backup = self.get_model()
    self.latest_epoch = tf.Variable(0, trainable=False, dtype=tf.int64)
    self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
    init_latent_codes = tf.random.normal((self.num_latent_codes,
                                          self.latent_code_dim))
    self.latent_code_vars = tf.Variable(init_latent_codes, trainable=True)
    self.network_vars = self.model.trainable_variables

  def init_optimizer(self,
                     learning_rate_network=0.0001,
                     learning_rate_latent=0.01,
                     decay_steps=10000,
                     decay_rate=0.96,
                     staircase=True):
    """Initialize the optimizers with a scheduler."""
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate_network,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase)
    self.optimizer_network = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate_latent,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase)
    self.optimizer_latent = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

  def init_checkpoint(self, checkpoint_dir, checkpoint=None):
    """Initialize the checkpoints."""
    self.summary_writer = tf.summary.create_file_writer(checkpoint_dir)
    self.checkpoint = tf.train.Checkpoint(
        model_nerf=self.model,
        latent_code_var=self.latent_code_vars,
        optimizer_network=self.optimizer_network,
        optimizer_latent=self.optimizer_latent,
        epoch=self.latest_epoch,
        global_step=self.global_step)
    self.manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                              directory=checkpoint_dir,
                                              max_to_keep=2)
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
      for a, b in zip(self.model_backup.variables,
                      self.model.variables):
        a.assign(b)
    else:
      logging.warning("No checkpoint was restored.")

  def reset_models(self):
    for a, b in zip(self.model.variables,
                    self.model_backup.variables):
      a.assign(b)

  def get_model(self):
    """NeRF-based network."""
    with tf.name_scope("Network/"):
      input_features = tf.keras.layers.Input(shape=[self.input_dim])
      feat0 = nerf_layers.concat_block(input_features,
                                       n_filters=self.n_filters,
                                       n_layers=5)
      feat1 = nerf_layers.dense_block(feat0,
                                      n_filters=self.n_filters,
                                      n_layers=3)
      rgb_density = tf.keras.layers.Dense(4)(feat1)
      return tf.keras.Model(inputs=[input_features], outputs=[rgb_density])

  @tf.function
  def prepare_positional_encoding(self, ray_points, ray_dirs):
    """Estimate the positional encoding of the 3D position and direction of the samples along a ray.

    Args:
      ray_points: A tensor of shape `[B, R, S, 3]` where B is the batch size, R
        is the number of rays, S is the number of samples per ray.
      ray_dirs: A tensor of shape `[B, R, 3]` where B is the batch size, R
      is the number of rays.

    Returns:
      A list containing a tensor of shape `[B, R, S, M]` and a tensor of shape
      `[B, R, S, N]`, where M is the dimensionality of the location positional
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
      rgb_density: A tensor of shape `[B*R*S, 4]` where B is the batch size,
      R is the number of rays, S is the number of samples per ray.
      ray_points: A tensor of shape `[B, R, S, 3]`.

    Returns:
      A tensor of shape `[B, R, 3]` and a tensor of shape `[B, R, C]`.

    """
    target_shape = tf.concat([tf.shape(ray_points)[:-1], [4]], axis=-1)
    rgb_density = tf.reshape(rgb_density, target_shape)
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
  def inference(self, r_org, r_dir, latent_code,
                voxels, w2v_alpha, w2v_beta, near, far):
    """Run both coarse and fine networks for given rays."""
    # Sample points along the rays ---------------------------------------------
    ray_points_coarse, z_vals_coarse = ray.sample_1d(
        r_org,
        r_dir,
        near=near,
        far=far,
        n_samples=self.ray_samples_coarse,
        strategy=self.coarse_sampling_strategy)
    # Extract the alpha for every point by looking the corresponding voxel -----
    voxel_values = voxel_functions.ray_sample_voxel_grid(ray_points_coarse,
                                                         voxels,
                                                         w2v_alpha,
                                                         w2v_beta)
    # Sample Additional points close to the surface ----------------------------
    ray_points_fine, _ = ray.sample_inverse_transform_stratified_1d(
        r_org,
        r_dir,
        z_vals_coarse,
        tf.squeeze(voxel_values, -1),
        n_samples=self.ray_samples_fine,
        combine_z_values=True)
    # Get all the features for the network (xyz, dir, latent, alpha) -----------
    posenc_xyz, posenc_dir = self.prepare_positional_encoding(ray_points_fine,
                                                              r_dir)
    latent_code = utils.match_intermediate_batch_dimensions(latent_code,
                                                            ray_points_fine)
    alpha = voxel_functions.ray_sample_voxel_grid(ray_points_fine,
                                                  voxels,
                                                  w2v_alpha,
                                                  w2v_beta)
    net_inputs = tf.concat([posenc_xyz,
                            posenc_dir,
                            latent_code,
                            alpha], axis=-1)
    net_inputs = tf.reshape(net_inputs, [-1, tf.shape(net_inputs)[-1]])
    # Run the network and render -----------------------------------------------
    rgb_density = self.model([net_inputs])
    rgb_fine, _ = self.render_network_output(rgb_density, ray_points_fine)
    return rgb_fine

  @tf.function
  def train_step(self, r_org, r_dir, shape_index,
                 voxels, w2v_alpha, w2v_beta, gt_rgb):
    """Main training function for coarse and fine networks.

    Args:
      r_org: A tensor of shape `[B, R, 3]` where B is the batch size,
        R is the number of rays and the last dimensios store the ray origin.
      r_dir: A tensor of shape `[B, R, 3]` where B is the batch size,
        R is the number of rays and the last dimensios store the ray direction.
      shape_index: A tensor of shape `[B]` where B is the batch size
      voxels: A tensor of shape `[B, 128, 128, 128, 1]` where B is the batch
        size and the other dimensions contain the voxel grid
      w2v_alpha: A tensor of shape `[B, 3]`
      w2v_beta: A tensor of shape `[B, 3]`
      gt_rgb: A tensor of shape `[B, R, 3]`

    Returns:
      A scalar.
    """
    with tf.GradientTape() as tape:
      latent_code = tf.gather(self.latent_code_vars, shape_index)
      rgb_fine = self.inference(r_org,
                                r_dir,
                                latent_code,
                                voxels,
                                w2v_alpha,
                                w2v_beta,
                                self.near,
                                self.far)
      rgb_fine_loss = utils.l2_loss(rgb_fine, gt_rgb)
      total_loss = rgb_fine_loss
    gradients = tape.gradient(total_loss,
                              self.network_vars + [self.latent_code_vars])
    self.optimizer_network.apply_gradients(
        zip(gradients[:len(self.network_vars)], self.network_vars))
    self.optimizer_latent.apply_gradients(
        zip(gradients[len(self.network_vars):], [self.latent_code_vars]))
    return total_loss
