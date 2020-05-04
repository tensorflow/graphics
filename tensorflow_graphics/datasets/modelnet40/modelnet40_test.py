#Copyright 2019 Google LLC
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
"""Tests the ModelNet40 dataset with fake data."""

import os
import tensorflow_datasets as tfds
from tensorflow_graphics.datasets import modelnet40

_EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "fakes")


class ModelNet40Test(tfds.testing.DatasetBuilderTestCase):
  """Tests the ModelNet40 dataset with fake data."""
  DATASET_CLASS = modelnet40.ModelNet40
  SPLITS = {
      "train": 24,  # Number of fake train example
      "test": 16,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  DL_EXTRACT_RESULT = ""
  EXAMPLE_DIR = _EXAMPLE_DIR
  # SKIP_CHECKSUMS = True


if __name__ == "__main__":
  tfds.testing.test_main()
