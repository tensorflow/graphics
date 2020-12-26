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

import shutil
import os
from tempfile import TemporaryDirectory
import zipfile

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from tensorflow_graphics.datasets.modelnet40 import LABELS
from tensorflow_graphics.datasets.modelnet40.mesh import off

# flags.DEFINE_string("fakes_path",
#                     default="",
#                     help="path where files will be generated")
# FLAGS = flags.FLAGS


def main(argv):
  """Generates files with the internal structure.

  Reference: f = h5py.File("modelnet40_ply_hdf5_2048/ply_data_train0.h5", "r")
    print(f['data'])  # <HDF5 dataset "data": shape(2048, 2048, 3), type "<f4">
    print(f['label']) # <HDF5 dataset "label": shape(2048, 1), type "|u1">
  """
  # if argv:
  #   raise app.UsageError("Invalid argument received.")

  # fakes_path = FLAGS.fakes_path
  fakes_path = ""
  if fakes_path == "":
    fakes_path = os.path.join(os.path.dirname(__file__), 'fakes')
  with TemporaryDirectory() as tmp_dir:
    seed = 0
    for label in LABELS:
      for split, num_examples in (('train', 3), ('test', 2)):
        folder = os.path.join(tmp_dir, 'foo', label, split)
        tf.io.gfile.makedirs(folder)
        for i in range(num_examples):
          filename = os.path.join(folder, '{}_{:04d}.off'.format(label, i + 1))
          off_obj = off.random_off(seed,
                                   num_vertices=(10, 20),
                                   num_faces=(2, 5),
                                   max_vertices_per_face=3)
          seed += 1
          with tf.io.gfile.GFile(filename, 'wb') as fp:
            off_obj.to_file(fp)
    zip_path = os.path.join(fakes_path, 'modelnet40')
    shutil.make_archive(zip_path, 'zip', tmp_dir)
    print('Data written to {}'.format(zip_path))


if __name__ == "__main__":
  app.run(main)
