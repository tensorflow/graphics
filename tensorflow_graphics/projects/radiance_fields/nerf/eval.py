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
"""Train."""
import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image
from skimage import metrics
import tensorflow as tf

import tensorflow_graphics.projects.radiance_fields.data_loaders as data_loaders
import tensorflow_graphics.projects.radiance_fields.nerf.model as model_lib
import tensorflow_graphics.projects.radiance_fields.utils as utils
import tensorflow_graphics.rendering.camera.perspective as perspective


flags.DEFINE_string('checkpoint_dir', '/tmp/lego/',
                    'Path to the directory of the checkpoint.')
flags.DEFINE_string('split', 'val', 'Train/val/test split.')
flags.DEFINE_boolean('single_eval', False, 'How many times to perform eval.')
flags.DEFINE_string('output_dir', '/tmp/lego/eval',
                    'Path to the directory of the output results.')
flags.DEFINE_string('dataset_dir', '/path/to/dataset/',
                    'Path to the directory of the dataset images.')
flags.DEFINE_string('dataset_name', 'lego', 'Dataset name.')
flags.DEFINE_float('dataset_scale', 0.5,
                   'Resolution of the dataset (1.0=800 pixels).')
flags.DEFINE_integer('num_epochs', 10000, 'How many epochs to train')
flags.DEFINE_integer('batch_size', 5, 'Number of images for each batch.')
flags.DEFINE_float('learning_rate', 0.0004, 'The optimizer learning rate.')
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
      batch_size=1,
      shuffle=False)

  model = model_lib.NeRF(
      ray_samples_coarse=FLAGS.ray_samples_coarse,
      ray_samples_fine=FLAGS.ray_samples_fine,
      near=FLAGS.near,
      far=FLAGS.far,
      n_freq_posenc_xyz=FLAGS.n_freq_posenc_xyz,
      scene_bbox=tuple([float(x) for x in FLAGS.scene_bbox.split(',')]),
      n_freq_posenc_dir=FLAGS.n_freq_posenc_dir,
      n_filters=FLAGS.n_filters,
      white_background=True)
  model.init_coarse_and_fine_models()
  model.init_optimizer(learning_rate=FLAGS.learning_rate)
  model.init_checkpoint(checkpoint_dir=FLAGS.checkpoint_dir)

  if not tf.io.gfile.exists(FLAGS.output_dir):
    tf.io.gfile.makedirs(FLAGS.output_dir)
  summary_writer = tf.summary.create_file_writer(FLAGS.output_dir)

  # ----------------------------------------------------------------------------
  current_evaluation = 0
  current_checkpoint = ''
  while True:
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    if latest_checkpoint is None:
      continue

    if current_checkpoint == latest_checkpoint:
      continue

    current_checkpoint = latest_checkpoint
    model.load_checkpoint(current_checkpoint)

    total_psnr = []
    total_ssim = []
    image_counter = 0
    for image, focal, principal_point, transform_matrix in dataset:

      img_rays, _ = perspective.random_patches(
          focal,
          principal_point,
          height,
          width,
          patch_height=height,
          patch_width=width,
          scale=1.0)

      # Batchify the image to fit into memory
      batch_rays = tf.split(img_rays, height, axis=1)
      output = []
      for random_rays in batch_rays:
        random_rays = utils.change_coordinate_system(random_rays,
                                                     (0., 0., 0.),
                                                     (1., -1., -1.))
        rays_org, rays_dir = utils.camera_rays_from_transformation_matrix(
            random_rays,
            transform_matrix)

        rgb_fine, *_ = model.inference(rays_org, rays_dir)
        output.append(rgb_fine)
      final_image = tf.concat(output, axis=0)
      final_image_np = final_image.numpy()

      image_rgb_no_alpha, image_a = tf.split(image, [3, 1], axis=-1)
      if FLAGS.white_background:
        image = image_rgb_no_alpha * image_a + 1 - image_a

      image_np = image.numpy()[0]
      ssim = metrics.structural_similarity(image_np,
                                           final_image_np,
                                           multichannel=True,
                                           data_range=1)
      psnr = metrics.peak_signal_noise_ratio(image_np,
                                             final_image_np,
                                             data_range=1)
      total_psnr.append(psnr)
      total_ssim.append(ssim)

      filename = os.path.join(FLAGS.output_dir,
                              '{0:05d}.png'.format(image_counter))
      img_to_save = Image.fromarray((final_image_np*255).astype(np.uint8))
      with tf.io.gfile.GFile(filename, 'wb') as f:
        img_to_save.save(f)

      logging.info('Image %d: ssim %.3f / psnr: %.3f',
                   image_counter, ssim, psnr)
      image_counter += 1

      # Show some images
      if image_counter < 5:
        with summary_writer.as_default():
          tf.summary.image('rgb_fine/{0}'.format(image_counter),
                           tf.expand_dims(final_image, 0),
                           step=current_evaluation,
                           max_outputs=4)
    with summary_writer.as_default():
      tf.summary.scalar('eval_ssim', np.mean(total_ssim),
                        step=current_evaluation)
      tf.summary.scalar('eval_psnr', np.mean(total_psnr),
                        step=current_evaluation)
    logging.info('ssim %.3f', np.mean(total_ssim))
    logging.info('psnr %.3f', np.mean(total_psnr))
    current_evaluation += 1

    if FLAGS.single_eval:
      break


if __name__ == '__main__':
  app.run(main)
