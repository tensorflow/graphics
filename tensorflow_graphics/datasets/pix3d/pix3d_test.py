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
# Lint as: python3
"""Tests for the Pix3D dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from tensorflow_graphics.datasets import pix3d
from tensorflow_graphics.rendering import rasterization_backend


class Pix3dTest(tfds.testing.DatasetBuilderTestCase):
  """Test Cases for Pix3D Dataset implementation."""
  DATASET_CLASS = pix3d.Pix3d
  SPLITS = {
      'train': 2,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  DL_EXTRACT_RESULT = ''
  EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), 'fakes')
  MOCK_OUT_FORBIDDEN_OS_FUNCTIONS = False

  # SKIP_CHECKSUMS = True

  def test_dataset_items(self):
    """Compares Object mask with rendered mask of transformed 3D Mesh."""
    builder = pix3d.Pix3d(data_dir=self.EXAMPLE_DIR)
    self._download_and_prepare_as_dataset(builder)
    for split_name in self.SPLITS:
      items = tfds.as_numpy(builder.as_dataset(split=split_name))
      for item in items:
        expected = item['mask']
        vertices = item['model']['vertices']
        faces = item['model']['faces']
        object_rotation = item['pose']['R']
        object_t = item['pose']['t']
        camera_rotation = item['camera']['parameters']['pose']['R']
        camera_t = item['camera']['parameters']['pose']['t']
        intrinsics = item['camera']['parameters']['intrinsics']
        perspective_matrix = self._build_4x4_projection(intrinsics)
        model_to_world = self._build_4x4_transform(object_rotation, object_t)
        world_to_eye = self._build_4x4_transform(camera_rotation, camera_t)
        model_to_eye = tf.matmul(world_to_eye, model_to_world)
        view_projection_matrix = tf.matmul(perspective_matrix, model_to_eye)
        _, _, rendered = rasterization_backend.rasterize(vertices,
                                                   faces,
                                                   view_projection_matrix,
                                                   item['image'].shape[:3])

        self.assertClose(expected, rendered)

  def setUp(self):  # pylint: disable=invalid-name
    """See base class for details."""
    super(Pix3dTest, self).setUp()
    self.builder.TRAIN_SPLIT_IDX = os.path.join(self.EXAMPLE_DIR,
                                                'pix3d_train.npy')
    self.builder.TEST_SPLIT_IDX = os.path.join(self.EXAMPLE_DIR,
                                               'pix3d_test.npy')

  def _build_4x4_transform(self, rotation, translation):
    """Builds a 4x4 transform matrix."""
    rotation4x3 = tf.concat([rotation, [[0., 0., 0.]]], 0)
    translation4x1 = tf.concat([tf.transpose(translation), [[1.]]], 0)
    return tf.concat([rotation4x3, translation4x1], 1)

  def _build_4x4_projection(self, intrinsics):
    """Builds a 4x4 perspective projection transform"""
    intrinsics4x3 = tf.concat([intrinsics, [[0., 0., 0.]]], 0)
    intrinsics4x4 = tf.concat([intrinsics4x3, [[0.], [0.], [0.], [1.]]], 1)
    return intrinsics4x4


if __name__ == '__main__':
  tfds.testing.test_main()
