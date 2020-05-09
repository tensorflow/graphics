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
"""Tests of pointnet module."""

import sys
import tempfile

import tensorflow_datasets as tfds
from tensorflow_graphics.util import test_case
from tensorflow_graphics.datasets import testing  # will fail until #310 merge
from tensorflow_graphics.projects.pointnet import fit


class TrainTest(test_case.TestCase):

  def test_train(self):
    batch_size = 8
    with tfds.testing.mock_data(batch_size * 2, data_dir=testing.DATA_DIR):
      with tempfile.TemporaryDirectory() as logdir:
        sys.argv = [
            "train.py",
            "--num_epochs",
            "2",
            "--assert_gpu",
            "False",
            "--ev_every",
            "1",
            "--tb_every",
            "1",
            "--logdir",
            logdir,
            "--batch_size",
            str(batch_size),
        ]
        fit.main()


if __name__ == "__main__":
  test_case.main()
