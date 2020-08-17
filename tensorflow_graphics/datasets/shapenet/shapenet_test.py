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
"""Tests for the Shapenet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow_datasets as tfds
from tensorflow_graphics.datasets.shapenet import shapenet

_EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), 'fakes')


class ShapenetTest(tfds.testing.DatasetBuilderTestCase):
  DATASET_CLASS = shapenet.Shapenet
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
      'validation': 1  # Number of fake validation examples
  }

  DL_EXTRACT_RESULT = 'all.csv'
  EXAMPLE_DIR = _EXAMPLE_DIR

  EXPECTED_ITEMS = {
      'train': {
          b'a98038807a61926abce962d6c4b37336': {
              'synset': '02691156',
              'num_vertices': 8,
              'num_faces': 12
          },
          b'a800bd725fe116447a84e76181a9e08f': {
              'synset': '03001627',
              'num_vertices': 6,
              'num_faces': 9
          },
          b'9550774ad1c19b24a5a118bd15e6e34f': {
              'synset': '02691156',
              'num_vertices': 12,
              'num_faces': 20
          },
      },
      'test': {
          b'3d5354863690ac7eca27bba175814d1': {
              'synset': '02691156',
              'num_vertices': 4,
              'num_faces': 4
          }
      },
      'validation': {
          b'7eff60e0d72800b8ca8607f540cc62ba': {
              'synset': '02691156',
              'num_vertices': 6,
              'num_faces': 8
          },
      }
  }

  def test_dataset_items(self):
    builder = shapenet.Shapenet(data_dir=self.tmp_dir)
    self._download_and_prepare_as_dataset(builder)
    for split_name in self.SPLITS:
      items = tfds.as_numpy(builder.as_dataset(split=split_name))
      for item in items:
        expected = self.EXPECTED_ITEMS[split_name][item['model_id']]
        self.assertEqual(item['label'],
                         self._encode_synset(builder, expected['synset']))
        self.assertLen(item['trimesh']['vertices'], expected['num_vertices'])
        self.assertLen(item['trimesh']['faces'], expected['num_faces'])

  def _encode_synset(self, builder, synset):
    return builder.info.features['label'].encode_example(synset)


if __name__ == '__main__':
  tfds.testing.test_main()
