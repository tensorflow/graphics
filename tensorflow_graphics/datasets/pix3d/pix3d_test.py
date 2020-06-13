"""pix3d dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow_datasets.public_api as tfds
from tensorflow_graphics.datasets import pix3d


class Pix3dTest(tfds.testing.DatasetBuilderTestCase):
  # TODO(pix3d):
  DATASET_CLASS = pix3d.Pix3d
  SPLITS = {
      "train": 2,  # Number of fake train example
      "test": 1,  # Number of fake test example
  }
  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}

  DL_EXTRACT_RESULT = ""
  EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "fakes")
  MOCK_OUT_FORBIDDEN_OS_FUNCTIONS = False

  def setUp(self):
    super(Pix3dTest, self).setUp()
    self.builder._TRAIN_SPLIT_IDX = "fakes/pix3d_train.npy"
    self.builder._TEST_SPLIT_IDX = "fakes/pix3d_test.npy"


if __name__ == "__main__":
  tfds.testing.test_main()
