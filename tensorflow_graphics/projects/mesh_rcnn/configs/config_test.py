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
"""Test cases for configuration for Mesh R-CNN shape branches."""

import os

import jsonschema.exceptions as json_exceptions

from tensorflow_graphics.projects.mesh_rcnn.configs.config import MeshRCNNConfig
from tensorflow_graphics.util import test_case


class ConfigTest(test_case.TestCase):
  """Tests the config loading and validation."""

  def test_wrong_path_config_load(self):
    """Tests for expection if invalid config path is provided."""
    path_to_json = "foo.json"
    with self.assertRaisesWithPredicateMatch(ValueError,
                                             f'Cannot find {path_to_json}.'):
      _ = MeshRCNNConfig(path_to_json)

  def test_missing_keys(self):
    """Tests for expection if config with missing keys is provided."""

    path_to_json = os.path.join(os.path.dirname(__file__),
                                'test_data',
                                'missing_keys_config.json')

    with self.assertRaisesWithPredicateMatch(
        json_exceptions.ValidationError,
        '\'voxel_prediction\' is a required property'):
      _ = MeshRCNNConfig(path_to_json)

  def test_invalid_value(self):
    """Tests for expection if a field in provided config has a wrong dtype."""

    path_to_json = os.path.join(os.path.dirname(__file__),
                                'test_data',
                                'invalid_dtype.json')
    with self.assertRaisesWithPredicateMatch(
        json_exceptions.ValidationError,
        '5000.5 is not of type \'integer\''):
      _ = MeshRCNNConfig(path_to_json)


if __name__ == '__main__':
  test_case.main()
