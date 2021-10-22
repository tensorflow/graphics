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
import tensorflow_graphics.projects.radiance_fields.sharf.geometry_net.model as model_lib
import tensorflow_graphics.projects.radiance_fields.sharf.geometry_net.optimization as optimization

from tensorflow_graphics.util import test_case

FLAGS = flags.FLAGS


class OptimizationTest(test_case.TestCase):

  def test_optimization(self):
    """Tests whether the NeRF model is initialized properly and can be trained."""

    latent_code_dim = 256

    geom_network = model_lib.GeometryNetwork(
        num_latent_codes=4371,
        latent_code_dim=latent_code_dim,
        fc_channels=512,
        fc_activation="relu",
        conv_size=4,
        norm3d="batchnorm",
        bce_gamma=0.8,
        proj_weight=0.01,
        mirror_weight=1.0)
    geom_network.init_model()
    geom_network.init_optimizer()
    geom_network.init_checkpoint(checkpoint_dir="/tmp/")

    mask = tf.ones((1, 128, 128, 1))
    latent_code = optimization.optimize_for_mask(
        geom_network,
        mask,
        focal=tf.ones((1, 2)),
        principal_point=tf.ones((1, 2)),
        rotation_matrix=tf.expand_dims(tf.eye(3, 3), 0),
        translation_vector=tf.ones((1, 3, 1)),
        w2v_alpha=tf.ones((1, 3)),
        w2v_beta=tf.ones((1, 3)),
        density=1,
        mirror_weight=50.0,
        near=1.25,
        far=3.5,
        n_samples=128,
        learning_rate_network=0.0001,
        learning_rate_code=0.1,
        n_rays=2024,
        n_iter=10,
        opt_mode="all",
        nearest_train_shape=None,
        verbose_steps=10)
    pred_logits_voxels = geom_network.model(latent_code)
    pred_voxels = tf.sigmoid(pred_logits_voxels) * geom_network.mask_voxels

    self.assertAllInRange(pred_voxels, 0.0, 1.0)


if __name__ == "__main__":
  test_case.main()
