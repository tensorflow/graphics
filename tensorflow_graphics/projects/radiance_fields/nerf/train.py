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
"""Train script for NeRF."""
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_graphics.projects.radiance_fields.data_loaders as data_loaders
import tensorflow_graphics.projects.radiance_fields.nerf.model as model_lib
import tensorflow_graphics.projects.radiance_fields.utils as utils
import tensorflow_graphics.rendering.camera.perspective as perspective

flags.DEFINE_string('checkpoint_dir', '/tmp/lego/',
                    'Path to the directory of the checkpoint.')
flags.DEFINE_string('split', 'train', 'Train/val/test split.')
flags.DEFINE_string('dataset_dir', '/path/to/dataset/',
                    'Path to the directory of the dataset images.')
flags.DEFINE_string('dataset_name', 'lego', 'Dataset name.')
flags.DEFINE_float('dataset_scale', 0.5,
                   'Resolution of the dataset (1.0=800 pixels).')
flags.DEFINE_integer('num_epochs', 10000, 'How many epochs to train')
flags.DEFINE_integer('batch_size', 5, 'Number of images for each batch.')
flags.DEFINE_float('learning_rate', 0.0004, 'The optimizer learning rate.')
flags.DEFINE_integer('decay_steps', 10000, 'Number of images for each batch.')
flags.DEFINE_integer('n_filters', 256, 'Number of filters of the MLP.')
flags.DEFINE_integer('n_freq_posenc_xyz', 10,
                     'Frequencies for the 3D location positional encoding.')
flags.DEFINE_string('scene_bbox', '-1.0,-1.0,-1.0,1.0,1.0,1.0',
                    'Bounding box of the scene.')

flags.DEFINE_integer('n_freq_posenc_dir', 4,
                     'Frequencies for the direction positional encoding.')
flags.DEFINE_float('near', 2.0, 'Closest ray location to get samples.')
flags.DEFINE_float('far', 6.0, 'Furthest ray location to get samples.')
flags.DEFINE_integer('ray_samples_coarse', 64,
                     'Samples on a ray for the coarse network.')
flags.DEFINE_integer('ray_samples_fine', 128,
                     'Samples on a ray for the fine network.')
flags.DEFINE_integer('n_rays', 512, 'Number of rays per image for training.')
flags.DEFINE_boolean('white_background', True, 'Use white background.')
flags.DEFINE_string('master', 'local', 'Location of the session.')

FLAGS = flags.FLAGS


def main(_):

  dataset, height, width = data_loaders.load_synthetic_nerf_dataset(
      dataset_dir=FLAGS.dataset_dir,
      dataset_name=FLAGS.dataset_name,
      split=FLAGS.split,
      scale=FLAGS.dataset_scale,
      batch_size=FLAGS.batch_size)

  model = model_lib.NeRF(
      ray_samples_coarse=FLAGS.ray_samples_coarse,
      ray_samples_fine=FLAGS.ray_samples_fine,
      near=FLAGS.near,
      far=FLAGS.far,
      n_freq_posenc_xyz=FLAGS.n_freq_posenc_xyz,
      scene_bbox=tuple([float(x) for x in FLAGS.scene_bbox.split(',')]),
      n_freq_posenc_dir=FLAGS.n_freq_posenc_dir,
      n_filters=FLAGS.n_filters,
      white_background=FLAGS.white_background)
  model.init_coarse_and_fine_models()
  model.init_optimizer(learning_rate=FLAGS.learning_rate,
                       decay_steps=FLAGS.decay_steps)
  model.init_checkpoint(checkpoint_dir=FLAGS.checkpoint_dir)

  for epoch in range(int(model.latest_epoch.numpy()), FLAGS.num_epochs):
    epoch_loss = 0.0
    for image, focal, principal_point, transform_matrix in dataset:
      random_rays, random_pixels_xy = perspective.random_rays(
          focal,
          principal_point,
          height,
          width,
          FLAGS.n_rays)
      random_rays = utils.change_coordinate_system(random_rays,
                                                   (0., 0., 0.),
                                                   (1., -1., -1.))
      rays_org, rays_dir = utils.camera_rays_from_transformation_matrix(
          random_rays,
          transform_matrix)
      random_pixels_yx = tf.reverse(random_pixels_xy, axis=[-1])
      pixels = tf.gather_nd(image, random_pixels_yx, batch_dims=1)
      pixels_rgb, pixels_a = tf.split(pixels, [3, 1], axis=-1)
      pixels_rgb = pixels_rgb * pixels_a + 1 - pixels_a

      dist_loss = model.train_step(rays_org, rays_dir, pixels_rgb)
      epoch_loss += dist_loss.numpy()
      model.global_step.assign_add(1)

    with model.summary_writer.as_default():
      tf.summary.scalar('epoch_loss', epoch_loss, step=epoch)
    if epoch % 20 == 0:
      model.manager.save()
    logging.info('Epoch %d: %.3f.', epoch, epoch_loss)
    model.latest_epoch.assign(epoch + 1)
  model.manager.save()

if __name__ == '__main__':
  app.run(main)
