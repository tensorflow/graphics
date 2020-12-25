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
"""Simple demo of modelnet40 dataset.

See: https://www.tensorflow.org/datasets/api_docs/python/tfds/load
"""
from absl import app

from tensorflow_graphics.datasets.modelnet40 import ModelNet40


def main(_):
  ds_train, info = ModelNet40.load(
      split="train", data_dir="~/tensorflow_dataset", with_info=True)

  for example in ds_train.take(1):
    points = example["points"]
    label = example["label"]

  # --- example accessing data
  print("points.shape=", points.shape)
  print("label.shape", label.shape)

  # --- example accessing info
  print("Example of string='{}' to ID#={}".format(
      "airplane", info.features["label"].str2int("airplane")))
  print("Example of ID#={} to string='{}'".format(
      12, info.features["label"].int2str(12)))


if __name__ == "__main__":
  app.run(main)
