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
"""ModelNet40 classification dataset from https://modelnet.cs.princeton.edu."""

import os
import h5py
import tensorflow as tf
import tensorflow_datasets as tfds

# Constants
_URL = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'

_CITATION = """
@inproceedings{qi2017pointnet,
  title={Pointnet: Deep learning on point sets for 3d classification and segmentation},
  author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={652--660},
  year={2017}
}
"""

_DESCRIPTION = """
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


class ModelNet40(tfds.core.GeneratorBasedBuilder):
  """ModelNet40."""

  VERSION = tfds.core.Version('1.0.0')

  @staticmethod
  def load(*args, **kwargs):
    return tfds.load('model_net40', *args, **kwargs)  # pytype: disable=wrong-arg-count

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'points': tfds.features.Tensor(shape=(2048, 3), dtype=tf.float32),
            'label': tfds.features.ClassLabel(names=_LABELS)
        }),
        supervised_keys=('points', 'label'),
        homepage='https://modelnet.cs.princeton.edu',
        citation=_CITATION,
    )

  def _split_generators(self, download_manager):
    """Returns SplitGenerators."""

    extracted_path = download_manager.download_and_extract(_URL)

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

  def _generate_examples(self, filename_list_path):
    """Yields examples."""

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
