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
"""Download, computes and stores the checksums."""

from absl import app
import tensorflow_datasets as tfds
from tensorflow_graphics.datasets.modelnet40 import ModelNet40


def main(_):
  config = tfds.download.DownloadConfig(register_checksums=True)
  modelnet40_builder = ModelNet40(data_dir="~/tensorflow_datasets")
  modelnet40_builder.download_and_prepare(download_config=config)


if __name__ == "__main__":
  app.run(main)
