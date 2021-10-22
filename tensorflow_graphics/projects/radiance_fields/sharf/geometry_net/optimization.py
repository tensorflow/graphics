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
"""Optimization functions for geometry network."""
import math
from absl import logging
import tensorflow as tf

import tensorflow_graphics.geometry.representation.ray as ray
import tensorflow_graphics.projects.radiance_fields.sharf.voxel_functions as voxel_functions
import tensorflow_graphics.projects.radiance_fields.utils as utils
import tensorflow_graphics.rendering.camera.perspective as perspective
import tensorflow_graphics.rendering.volumetric.ray_density as ray_density


def optimize_for_mask(geometry_network,
                      mask,
                      focal,
                      principal_point,
                      rotation_matrix,
                      translation_vector,
                      w2v_alpha,
                      w2v_beta,
                      density=20,
                      mirror_weight=1.0,
                      near=1.25,
                      far=3.5,
                      n_samples=128,
                      learning_rate_network=0.0001,
                      learning_rate_code=0.1,
                      n_rays=1024,
                      n_iter=100,
                      opt_mode='code',
                      nearest_train_shape=None,
                      ray_sampling_strategy='stratified',
                      verbose_steps=10):
  """Optimize geometry network and/or its latent codes to fit a mask.

  Args:
    geometry_network: A keras model for the geometry network.
    mask: A tensor with shape `[1, H, W, 1]`
      where H and W are the height and width of the target mask.
    focal: A tensor with shape `[1, 2]`
    principal_point:  A tensor with shape `[1, 2]`.
    rotation_matrix:  A tensor with shape `[1, 3, 3]`.
    translation_vector: A tensor with shape `[1, 3, 1]`.
    w2v_alpha: A tensor with shape `[1, 3, 1]` containing the
      scaling transformation between world and voxel coordinates.
    w2v_beta: A tensor with shape `[1, 3, 1]` containing the
      translation transformation between world and voxel coordinates.
    density: A scalar weight for rendering the voxel grid.
    mirror_weight: A scalar for the weight for the mirror (symmetry) loss.
    near: A scalar for the lower bound for sampling along a ray.
    far: A scalar for the uper bound for sampling along a ray.
    n_samples: The number of samples along a ray.
    learning_rate_network: The fine-tuning learning rate of the nework.
    learning_rate_code: The fine-tuning learning rate of the latent codes.
    n_rays: The number of random rays to sample per iteration.
    n_iter: The number of iterations for the optimization.
    opt_mode: A string indicating the optimization strategy ('network', 'code').
    nearest_train_shape: Index of the nearest training shape if available.
    ray_sampling_strategy: A string indicating the ray sampling strategy.
    verbose_steps: After how many steps to print the losses.

  Returns:
    The optimized latent code for the input mask ().
  """
  height, width = mask.shape[-3], mask.shape[-2]
  # Retrieve the latent code of the nearest shape if available.
  if nearest_train_shape:
    voxel_code_var = tf.gather(geometry_network.latent_code_vars,
                               [nearest_train_shape])
  else:
    voxel_code_var = tf.reduce_mean(geometry_network.latent_code_vars,
                                    axis=0,
                                    keepdims=True)
  voxel_code_var = tf.Variable(voxel_code_var)
  # Set up the optimizers and variables to be optimized
  optimizer_network = tf.keras.optimizers.Adam(
      learning_rate=learning_rate_network)
  optimizer_latent = tf.keras.optimizers.Adam(
      learning_rate=learning_rate_code)
  if opt_mode == 'all':
    network_vars = geometry_network.model.trainable_variables
  else:
    network_vars = []

  # The main optimization function
  @tf.function
  def opt_step(r_org, r_dir, gt_a):
    with tf.GradientTape() as tape:
      # Run the network
      pred_logits_voxels = geometry_network.model(voxel_code_var)
      voxels = tf.sigmoid(pred_logits_voxels) * geometry_network.mask_voxels
      # Sample points along the rays and estimate the voxel values
      ray_points_coarse, _ = ray.sample_1d(r_org,
                                           r_dir,
                                           near=near,
                                           far=far,
                                           n_samples=n_samples,
                                           strategy=ray_sampling_strategy)
      voxel_values = voxel_functions.ray_sample_voxel_grid(ray_points_coarse,
                                                           voxels,
                                                           w2v_alpha,
                                                           w2v_beta)
      # Render the silhouette
      silhouettes, *_ = ray_density.compute_density(
          voxel_values, density*tf.ones_like(voxel_values[..., 0]))
      silhouette_loss = utils.l2_loss(silhouettes, gt_a)
      mirror_voxel_loss = utils.l2_loss(voxels, tf.reverse(voxels, [1]))
      total_loss = silhouette_loss + mirror_weight*mirror_voxel_loss
    gradients = tape.gradient(total_loss, network_vars + [voxel_code_var])
    optimizer_network.apply_gradients(zip(gradients[:len(network_vars)],
                                          network_vars))
    optimizer_latent.apply_gradients(zip(gradients[len(network_vars):],
                                         [voxel_code_var]))
    return total_loss, silhouettes

  for it in range(n_iter):
    # Sample random rays.
    random_rays, random_pixels_xy = perspective.random_rays(
        focal,
        principal_point,
        height,
        width,
        n_rays)
    # Change the coordinate system from TFG to blender SRN
    random_rays = utils.change_coordinate_system(random_rays,
                                                 (0., math.pi, math.pi),
                                                 (-1., 1., 1.))
    rays_org, rays_dir = utils.camera_rays_from_extrinsics(
        random_rays,
        rotation_matrix,
        translation_vector)
    random_pixels_yx = tf.reverse(random_pixels_xy, axis=[-1])
    random_pixels_yx = tf.cast(random_pixels_yx, tf.int32)
    pixels = tf.gather_nd(mask, random_pixels_yx, batch_dims=1)

    loss, _ = opt_step(rays_org, rays_dir, pixels)
    if it % verbose_steps == 0:
      logging.info('Iter %d. loss: %.5f', it, loss)
  return voxel_code_var
