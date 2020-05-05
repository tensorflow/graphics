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
"""ModelNet40 classification dataset fom https://modelnet.cs.princeton.edu."""

import abc
import os
import numpy as np
import h5py
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_graphics.datasets.modelnet40 import off

# Constants
_POINTNET_URL = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'

_POINTNET_CITATION = """
@inproceedings{qi2017pointnet,
  title={Pointnet: Deep learning on point sets for 3d classification and segmentation},
  author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={652--660},
  year={2017}
}
"""

_POINTNET_DESCRIPTION = """
The dataset contains point clouds sampling CAD models from 40 different categories.
The files have been retrieved from https://modelnet.cs.princeton.edu

To generate each example, the authors first uniformly sampled a modelnet40 mesh
with 10000 uniform samples, and then employed furthest point sampling to downsample
to 2048 points. The procedure is explained [here](https://github.com/charlesq34/pointnet2/blob/master/tf_ops/sampling/tf_sampling.py)
"""

_LABELS = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
    'wardrobe', 'xbox'
]

# --- registers the checksum
_CHECKSUM_DIR = os.path.join(os.path.dirname(__file__), 'checksums/')
_CHECKSUM_DIR = os.path.normpath(_CHECKSUM_DIR)
tfds.download.add_checksums_dir(_CHECKSUM_DIR)


class ModelNet40Config(tfds.core.BuilderConfig):
  """
  Base class for ModelNet40 BuilderConfigs.

  The builder delegates required implementations to the builder config. This
  allows multiple versions of the same dataset - including those that source
  data from different locations e.g. pointnet - to exist under the same
  tfds name.
  """
  @abc.abstractmethod
  def info(self, builder):
    """Delegated GeneratorBaseBuilder._info"""
    raise NotImplementedError('Abstract method')

  @abc.abstractmethod
  def split_generators(self, download_manager):
    """Delegated GeneratorBaseBuilder._split_generators"""
    raise NotImplementedError('Abstract method')

  @abc.abstractmethod
  def generate_examples(self, **kwargs):
    """Delegated GeneratorBaseBuilder._generate_examples"""
    raise NotImplementedError('Abstract method')


class PointnetConfig(ModelNet40Config):
  """Config for point cloud data used in Pointnet."""
  def __init__(self):
    super().__init__(
        name='pointnet',
        description=_POINTNET_DESCRIPTION,
        version=tfds.core.Version('1.0.0')
    )

  def info(self, builder):
    return tfds.core.DatasetInfo(
        builder=builder,
        features=tfds.features.FeaturesDict({
            'points': tfds.features.Tensor(shape=(2048, 3), dtype=tf.float32),
            'label': tfds.features.ClassLabel(names=_LABELS)
        }),
        supervised_keys=('points', 'label'),
        homepage='https://modelnet.cs.princeton.edu',
        citation=_POINTNET_CITATION,
    )

  def split_generators(self, download_manager):
    extracted_path = download_manager.download_and_extract(_POINTNET_URL)

    # Note: VALIDATION split was not provided by the authors
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs=dict(
                filename_list_path=os.path.join(
                    extracted_path,
                    'modelnet40_ply_hdf5_2048/train_files.txt'),)),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs=dict(
                filename_list_path=os.path.join(
                    extracted_path,
                    'modelnet40_ply_hdf5_2048/test_files.txt'),)),
    ]

  def generate_examples(self, filename_list_path):  # pylint:disable=arguments-differ
    """Delegated GeneratorBaseBuilder._generate_examples"""
    ancestor_path = os.path.dirname(os.path.dirname(filename_list_path))
    with tf.io.gfile.GFile(filename_list_path, 'r') as fid:
      filename_list = fid.readlines()
    filename_list = [line.rstrip()[5:] for line in filename_list]

    example_key = -1  # as yield exists, need to pre-increment
    for filename in filename_list:
      h5path = os.path.join(ancestor_path, filename)

      with h5py.File(h5path, 'r') as h5file:
        points = h5file['data'][:]  # shape=(2048, 2048, 3)
        label = h5file['label'][:]  # shape=(2048, )

        models_per_file = points.shape[0]
        for imodel in range(models_per_file):
          example_key += 1
          yield example_key, {
              'points': points[imodel, :, :],
              'label': int(label[imodel])
          }


_POLYMESH_DESCRIPTION = """\
The dataset contains polygonal mesh data from 40 different categories.

The files have been retrieved from https://modelnet.cs.princeton.edu
"""

_TRIMESH_DESCRIPTION = """\
The dataset contains triangular mesh data from 40 different categories.

The files have been retrieved from https://modelnet.cs.princeton.edu
"""

_MODELNET_CITATION = """\
@inproceedings{wu20153d,
  title={3d shapenets: A deep representation for volumetric shapes},
  author={Wu, Zhirong and Song, Shuran and Khosla, Aditya and Yu, Fisher and Zhang, Linguang and Tang, Xiaoou and Xiao, Jianxiong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1912--1920},
  year={2015}
}
"""


class MeshConfig(ModelNet40Config):
  """Base class for derivatives of the original .off mesh data."""
  def __init__(self, input_key, input_feature, **kwargs):
    self._input_key = input_key
    self._input_feature = input_feature
    super().__init__(**kwargs)

  @abc.abstractmethod
  def load(self, fp):
    raise NotImplementedError('Abstract method')

  def info(self, builder):
    return tfds.core.DatasetInfo(
        builder=builder,
        features=tfds.features.FeaturesDict({
            self._input_key: self._input_feature,
            'example_index': tfds.features.Tensor(shape=(), dtype=tf.int64),
            'label': tfds.features.ClassLabel(names=_LABELS)
        }),
        supervised_keys=(self._input_key, 'label'),
        homepage='https://modelnet.cs.princeton.edu',
        citation=_MODELNET_CITATION,
    )

  def split_generators(self, download_manager):
    archive_res = download_manager.download(
        'http://modelnet.cs.princeton.edu/ModelNet40.zip')
    archive_iter = download_manager.iter_archive(archive_res)
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs=dict(archive_iter=archive_iter, split_name='train'),
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs=dict(archive_iter=archive_iter, split_name='test'),
        )
    ]

  def generate_examples(self, archive_iter, split_name):  # pylint:disable=arguments-differ
    input_key = self._input_key

    for path, fp in archive_iter:
      if not path.endswith('.off'):
        continue
      class_name, split, fn = path.split('/')[-3:]
      if split != split_name:
        continue

      inputs = self.load(fp)
      if inputs is not None:
        yield path, {
            input_key: inputs,
            'example_index': int(fn.split('_')[-1][:-4]) - 1,
            'label': class_name,
        }


class PolymeshConfig(MeshConfig):
  """MeshConfig with original polygonal meshes."""
  def __init__(self):
    super().__init__(
        input_key='polymesh',
        input_feature=dict(
            vertices=tfds.features.Tensor(shape=(None, 3), dtype=tf.float32),
            face_values=tfds.features.Tensor(shape=(None,), dtype=tf.int64),
            face_lengths=tfds.features.Tensor(shape=(None,), dtype=tf.int64),
        ),
        name='polymesh',
        version=tfds.core.Version('1.0.0'),
        description=_POLYMESH_DESCRIPTION
    )

  def load(self, fp):  # pylint: disable=unused-argument
    data = off.OffObject.from_file(fp)
    return dict(
        vertices=data.vertices.astype(np.float32),
        face_values=data.face_values,
        face_lengths=data.face_lengths,
    )


class TrimeshConfig(MeshConfig):
  """MeshConfig with triangular mesh features."""
  def __init__(self):
    super().__init__(
        input_key='trimesh',
        input_feature=dict(
            vertices=tfds.features.Tensor(shape=(None, 3), dtype=tf.float32),
            faces=tfds.features.Tensor(shape=(None, 3), dtype=tf.int64),
        ),
        name='trimesh',
        version=tfds.core.Version('1.0.0'),
        description=_TRIMESH_DESCRIPTION,
    )

  def load(self, fp):
    data = off.OffObject.from_file(fp)
    return dict(
        vertices=data.vertices.astype(np.float32),
        faces=off.triangulated_faces(data.face_values, data.face_lengths)
    )


POINTNET = PointnetConfig()

POLYMESH = PolymeshConfig()

TRIMESH = TrimeshConfig()


class ModelNet40(tfds.core.GeneratorBasedBuilder):
  """ModelNet40."""
  BUILDER_CONFIGS = [POINTNET, POLYMESH, TRIMESH]

  @staticmethod
  def load(*args, **kwargs):
    return tfds.load('model_net40', *args, **kwargs)

  def _info(self):
    return self.builder_config.info(self)

  def _split_generators(self, download_manager):
    """Returns SplitGenerators."""
    return self.builder_config.split_generators(download_manager)

  def _generate_examples(self, **kwargs):
    """Yields examples."""
    return self.builder_config.generate_examples(**kwargs)
