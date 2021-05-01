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
from skimage import measure
import tensorflow as tf

import tensorflow_graphics.projects.radiance_fields.data_loaders as data_loaders
import tensorflow_graphics.projects.radiance_fields.nerf.model as model_lib
import tensorflow_graphics.projects.radiance_fields.utils as utils
import tensorflow_graphics.rendering.camera.perspective as perspective


CHECKPOINT_DIR = '/tmp/lego/'
OUTPUT_DIR = '/tmp/lego/results/'

flags.DEFINE_string('checkpoint_dir', CHECKPOINT_DIR,
                    'Path to the directory of the checkpoint.')
flags.DEFINE_string('output_dir', OUTPUT_DIR,
                    'Path to the directory of the output results.')
flags.DEFINE_string('split', 'test', 'Train/val/test split.')
flags.DEFINE_string('dataset_dir', '/path/to/dataset/',
                    'Path to the directory of the dataset images.')
flags.DEFINE_string('dataset_name', 'lego', 'Dataset name.')
flags.DEFINE_float('dataset_scale', 0.5,
                   'Resolution of the dataset (1.0=800 pixels).')
flags.DEFINE_integer('n_filters', 256, 'Number of filters of the MLP.')
flags.DEFINE_integer('posenc_loc', 10,
                     'Frequencies for the 3D location positional encoding.')
flags.DEFINE_float('posenc_loc_scale', 1.0,
                   'Scaling factor for the 3D location positional encoding.')
flags.DEFINE_integer('posenc_dir', 4,
                     'Frequencies for the direction positional encoding.')
flags.DEFINE_float('near', 2.0, 'Closest ray location to get samples.')
flags.DEFINE_float('far', 6.0, 'Furthest ray location to get samples.')
flags.DEFINE_integer('ray_samples_coarse', 64,
                     'Samples on a ray for the coarse network.')
flags.DEFINE_integer('ray_samples_fine', 128,
                     'Samples on a ray for the fine network.')
FLAGS = flags.FLAGS


def main(_):
  dataset, height, width = data_loaders.load_nerf_dataset(
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
      posenc_loc=FLAGS.posenc_loc,
      posenc_loc_scale=FLAGS.posenc_loc_scale,
      posenc_dir=FLAGS.posenc_dir,
      n_filters=FLAGS.n_filters,
      white_background=True)
  model.init_coarse_and_fine_models()
  model.init_optimizer(learning_rate=0.0001)
  model.init_checkpoint(checkpoint_dir=FLAGS.checkpoint_dir)

  if not tf.io.gfile.exists(FLAGS.output_dir):
    tf.io.gfile.makedirs(FLAGS.output_dir)

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
        scale=1.0,
        indexing='xy')

    batch_rays = tf.split(img_rays, width, axis=1)

    output = []
    for random_rays in batch_rays:
      random_rays = utils.change_coordinate_system(random_rays,
                                                   (0., 0., 0.),
                                                   (1., -1., -1.))
      rays_org, rays_dir = utils.camera_rays_from_transformation_matrix(
          random_rays,
          transform_matrix)

      rgb_fine, _ = model.inference(rays_org, rays_dir)
      output.append(rgb_fine)
    final_image = tf.transpose(tf.concat(output, axis=0), [0, 1, 2])
    final_image = tf.cast(final_image*255, tf.uint8)
    final_image = final_image.numpy()

    image_np_ = image.numpy()
    image_np = image_np_[0, :, :, :3]
    alpha_np = image_np_[0, :, :, 3][:, :, None]
    image_np = image_np * alpha_np + 1 - alpha_np

    image_name = '{0:05d}.png'.format(image_counter)
    filename = os.path.join(FLAGS.output_dir, image_name)
    img_to_save = Image.fromarray(final_image)
    img_to_save.save(filename)

    ssim = measure.compare_ssim(image_np,
                                final_image/255.,
                                multichannel=True,
                                data_range=1)
    psnr = measure.compare_psnr(image_np,
                                final_image/255.,
                                data_range=1)
    total_psnr.append(psnr)
    total_ssim.append(ssim)

    logging.info('Image %d: ssim %.3f / psnr: %.3f', image_counter, ssim, psnr)
    image_counter += 1

  logging.info('ssim %.3f', np.mean(total_ssim))
  logging.info('psnr %.3f', np.mean(total_psnr))

if __name__ == '__main__':
  app.run(main)
