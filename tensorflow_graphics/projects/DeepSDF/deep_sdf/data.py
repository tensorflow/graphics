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
""" NO COMMENT NOW"""


import os
import random
import glob
import logging
import numpy as np
import tensorflow as tf

import deep_sdf.workspace as ws


def get_instance_filenames(data_source, split):
  npzfiles = []
  for dataset in split:
    for class_name in split[dataset]:
      for instance_name in split[dataset][class_name]:
        instance_filename = os.path.join(
            dataset, class_name, instance_name + ".npz"
        )
        if not os.path.isfile(
            os.path.join(
                data_source, ws.sdf_samples_subdir, instance_filename)
        ):
          # raise RuntimeError(
          #     'Requested non-existent file "' + instance_filename + "'"
          # )
          logging.warning(
              "Requested non-existent file '%s'",
              instance_filename
          )
        npzfiles += [instance_filename]
  return npzfiles


class NoMeshFileError(RuntimeError):
  """Raised when a mesh file is not found in a shape directory"""


class MultipleMeshFileError(RuntimeError):
  """"Raised when a there a multiple mesh files in a shape directory"""


def find_mesh_in_directory(shape_dir):
  mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
      glob.iglob(shape_dir + "/*.obj")
  )
  if len(mesh_filenames) == 0:
    raise NoMeshFileError()
  elif len(mesh_filenames) > 1:
    raise MultipleMeshFileError()
  return mesh_filenames[0]


def remove_nans(tensor):
  tensor_nan = tf.math.is_nan(tensor[:, 3])
  return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
  npz = np.load(filename)
  pos_tensor = tf.convert_to_tensor(npz["pos"])
  neg_tensor = tf.convert_to_tensor(npz["neg"])

  return [pos_tensor, neg_tensor]


# https://stackoverflow.com/questions/58464790/is-there-an-equivalent-function-of-pytorch-named-index-select-in-tensorflow
def tf_index_select(input_, dim, indices):
  """
  input_(tensor): input tensor
  dim(int): dimension
  indices(list): selected indices list
  """
  shape = input_.get_shape().as_list()
  if dim == -1:
    dim = len(shape) - 1
  shape[dim] = 1

  tmp = []
  for idx in indices:
    begin = [0] * len(shape)
    begin[dim] = idx
    tmp.append(tf.slice(input_, begin, shape))
  res = tf.concat(tmp, axis=dim)

  return res


def unpack_sdf_samples(filename, subsample=None):
  npz = np.load(filename)
  if subsample is None:
    return npz
  pos_tensor = remove_nans(tf.convert_to_tensor(npz["pos"]))
  neg_tensor = remove_nans(tf.convert_to_tensor(npz["neg"]))

  # split the sample into half
  half = int(subsample / 2)

  random_pos = (tf.random.uniform(half) * pos_tensor.shape[0])
  random_neg = (tf.random.uniform(half) * neg_tensor.shape[0])

  sample_pos = tf_index_select(pos_tensor, 0, random_pos)
  sample_neg = tf_index_select(neg_tensor, 0, random_neg)

  samples = tf.concat([sample_pos, sample_neg], axis=0)

  return samples


def unpack_sdf_samples_from_ram(data, subsample=None):
  if subsample is None:
    return data
  pos_tensor = data[0]
  neg_tensor = data[1]

  # split the sample into half
  half = int(subsample / 2)

  pos_size = pos_tensor.shape[0]
  neg_size = neg_tensor.shape[0]

  pos_start_ind = random.randint(0, pos_size - half)
  sample_pos = pos_tensor[pos_start_ind: (pos_start_ind + half)]

  if neg_size <= half:
    random_neg = tf.cast(tf.random.uniform(
        half) * neg_tensor.shape[0], dtype=tf.int64)
    sample_neg = tf_index_select(neg_tensor, 0, random_neg)
  else:
    neg_start_ind = random.randint(0, neg_size - half)
    sample_neg = neg_tensor[neg_start_ind: (neg_start_ind + half)]

  samples = tf.concat([sample_pos, sample_neg], axis=0)

  return samples


class SDFSamples(object):
  def __init__(
      self,
      data_source,
      split,
      subsample,
      batch_size=64,
      shuffle=True,
      epoch=2001,
      load_ram=False,
  ):
    self.subsample = subsample
    self.batch_sizd = batch_size
    self.shuffle = shuffle
    self.epoch = epoch

    self.data_source = data_source
    self.npyfiles = get_instance_filenames(data_source, split)

    logging.debug(
        "using %s shapes from data source %s ", str(
            len(self.npyfiles)), data_source
    )

    self.load_ram = load_ram

    self.data_filename = []

    if load_ram:
      self.data_ram = []
      for _, f in enumerate(self.npyfiles):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, f)
        npz = np.load(filename)
        pos_tensor = remove_nans(tf.convert_to_tensor(npz["pos"]))
        neg_tensor = remove_nans(tf.convert_to_tensor(npz["neg"]))
        self.data_ram.append(
            [
                pos_tensor[tf.random.shuffle(
                    tf.range(start=0, limit=pos_tensor.shape[0]))],
                neg_tensor[tf.random.shuffle(
                    tf.range(start=0, limit=neg_tensor.shape[0]))]
            ]
        )

    for _, f in enumerate(self.npyfiles):
      filename = os.path.join(
          self.data_source, ws.sdf_samples_subdir, f
      )
      self.data_filename.append(filename)

    random.shuffle(self.data_ram)
    random.shuffle(self.data_filename)

  def __len__(self):
    return len(self.npyfiles)

  def generator(self):
    for idx, _ in enumerate(self.npyfiles):
      if self.load_ram:
        yield (
            unpack_sdf_samples_from_ram(
                self.data_ram[idx], self.subsample),
            idx,
        )
      else:
        yield unpack_sdf_samples(self.data_filename[idx], self.subsample), idx

  def dataset(self):
    dataset = tf.data.Dataset.from_generator(
        self.generator, output_types=tf.float64)
    if self.shuffle:
      dataset.apply(tf.data.experimental.shuffle_and_repeat(
          self.batch_size * 3, self.epoch))
    dataset = dataset.batch(batch_size=self.batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
