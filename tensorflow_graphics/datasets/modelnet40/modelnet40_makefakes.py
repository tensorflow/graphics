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
"""Generates fake data for testing."""

import os
from absl import app
from absl import flags
import h5py
import numpy as np

flags.DEFINE_string("fakes_path", ".", "path where files will be generated")
FLAGS = flags.FLAGS


def main(argv):
  """Generates files with the internal structure.

  Args:
    argv: the path where to generate the fake files
  Reference: f = h5py.File("modelnet40_ply_hdf5_2048/ply_data_train0.h5", "r")
    print(f['data'])  # <HDF5 dataset "data": shape(2048, 2048, 3), type "<f4">
    print(f['label']) # <HDF5 dataset "label": shape(2048, 1), type "|u1">
  """
  if len(argv) != 1:
    raise app.UsageError("One argument required.")

  for i in range(3):
    fake_points = np.random.randn(8, 2048, 3).astype(np.float32)
    fake_label = np.random.uniform(low=0, high=40, size=(8, 1)).astype(np.uint8)
    path = os.path.join(FLAGS.fakes_path, "ply_data_train{}.h5".format(i))
    with h5py.File(path, "w") as h5f:
      h5f.create_dataset("data", data=fake_points)
      h5f.create_dataset("label", data=fake_label)
  for i in range(2):
    fake_points = np.random.randn(8, 2048, 3).astype(np.float32)
    fake_label = np.random.uniform(low=0, high=40, size=(8, 1)).astype(np.uint8)
    path = os.path.join(FLAGS.fakes_path, "ply_data_test{}.h5".format(i))
    with h5py.File(path, "w") as h5f:
      h5f.create_dataset("data", data=fake_points)
      h5f.create_dataset("label", data=fake_label)


if __name__ == "__main__":
  app.run(main)
