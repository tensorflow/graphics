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
"""Visualization in 3D of modelnet40 dataset.

See: https://www.tensorflow.org/datasets/api_docs/python/tfds/load
"""
from absl import app
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pylint:disable=unused-import
from tensorflow_graphics.datasets.modelnet40 import ModelNet40


def main(_):
  ds_train, _ = ModelNet40.load(
      split="train", data_dir="~/tensorflow_dataset", with_info=True)

  for example in ds_train.take(1):
    points = example["points"]
    label = example["label"]

  fig = plt.figure()
  ax3 = fig.add_subplot(111, projection="3d")
  ax3.set_title("Example with label {}".format(label))
  scatter3 = lambda p, c="r", *args: ax3.scatter(p[:, 0], p[:, 1], p[:, 2], c)
  scatter3(points)


if __name__ == "__main__":
  app.run(main)
