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
r"""Tests for the NeRF model."""

from absl import flags
import tensorflow as tf
import tensorflow_graphics.projects.radiance_fields.nerf.model as model_lib

from tensorflow_graphics.util import test_case

FLAGS = flags.FLAGS


class ModelTest(test_case.TestCase):

  def test_model_training(self):
    """Tests whether the NeRF model is initialized properly and can be trained."""

    model = model_lib.NeRF(
        ray_samples_coarse=128,
        ray_samples_fine=128,
        near=1.0,
        far=6.0,
        scene_bbox=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        n_freq_posenc_xyz=10,
        n_freq_posenc_dir=4,
        n_filters=128,
        white_background=True)
    model.init_coarse_and_fine_models()
    model.init_optimizer(learning_rate=0.0001)
    model.init_checkpoint(checkpoint_dir="/tmp/")

    batch_size = 10
    n_rays = 256

    rays_org = tf.zeros((batch_size, n_rays, 3), dtype=tf.float32)
    rays_dir = tf.zeros((batch_size, n_rays, 3), dtype=tf.float32)
    pixels_rgb = tf.zeros((batch_size, n_rays, 3), dtype=tf.float32)

    rgb_loss = model.train_step(rays_org, rays_dir, pixels_rgb)
    self.assertAllInRange(rgb_loss, 0.0, 1000.0)

if __name__ == "__main__":
  test_case.main()
