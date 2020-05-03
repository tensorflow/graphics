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
"""Simple demo of modelnet40 dataset.

See: https://www.tensorflow.org/datasets/api_docs/python/tfds/load
"""
from absl import app
import trimesh
import tensorflow_datasets as tfds
from tensorflow_graphics.datasets.modelnet40 import ModelNet40, TRIMESH

def main(_):
  builder = ModelNet40(config=TRIMESH)
  builder.download_and_prepare(
      download_config=tfds.core.download.DownloadConfig(
          register_checksums=True))
  ds_train = builder.as_dataset(split='test')

  for example in ds_train.take(1):
    mesh = example["trimesh"]
    label = example["label"]

  # --- example accessing data
  print("mesh['vertices'].shape=", mesh['vertices'].shape)
  print("mesh['faces'].shape=", mesh['faces'].shape)
  print("label.shape", label.shape)

  trimesh.Trimesh(
      vertices=mesh['vertices'].numpy(), faces=mesh['faces'].numpy()).show()


if __name__ == "__main__":
  app.run(main)
