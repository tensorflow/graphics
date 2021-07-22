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

import tensorflow as tf
import tensorflow_graphics.projects.radiance_fields.sharf.appearance_net.model as model_lib

from tensorflow_graphics.util import test_case


class ModelTest(test_case.TestCase):

  def test_model_training(self):
    """Tests whether the NeRF model is initialized properly and can be trained."""

    app_network = model_lib.AppearanceNetwork(
        ray_samples_coarse=64,
        ray_samples_fine=64,
        near=1.0,
        far=3.0,
        n_freq_posenc_xyz=8,
        n_freq_posenc_dir=0,
        scene_bbox=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        n_filters=128,
        num_latent_codes=4371,
        latent_code_dim=16,
        white_background=True,
        coarse_sampling_strategy="stratified")
    app_network.init_model_and_codes()
    app_network.init_optimizer()
    app_network.init_checkpoint(checkpoint_dir="/tmp/")

    batch_size = 5
    n_rays = 128

    r_org = tf.zeros((batch_size, n_rays, 3), dtype=tf.float32)
    r_dir = tf.zeros((batch_size, n_rays, 3), dtype=tf.float32)
    shape_index = tf.zeros((batch_size), dtype=tf.int32)
    voxels = tf.zeros((batch_size, 128, 128, 128, 1), dtype=tf.float32)
    w2v_alpha = tf.zeros((batch_size, 3), dtype=tf.float32)
    w2v_beta = tf.zeros((batch_size, 3), dtype=tf.float32)
    gt_rgb = tf.zeros((batch_size, n_rays, 3), dtype=tf.float32)

    loss = app_network.train_step(r_org,
                                  r_dir,
                                  shape_index,
                                  voxels,
                                  w2v_alpha,
                                  w2v_beta,
                                  gt_rgb)

    self.assertAllInRange(loss, 0.0, 1000.0)


if __name__ == "__main__":
  test_case.main()
