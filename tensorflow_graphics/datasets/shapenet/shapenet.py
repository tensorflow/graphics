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
"""Shapenet Core dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import json
import os
import textwrap

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow_datasets import features as tfds_features
from tensorflow_graphics.datasets import features as tfg_features

_CITATION = """
@techreport{shapenet2015,
  title       = {{ShapeNet: An Information-Rich 3D Model Repository}},
  author      = {Chang, Angel X. and Funkhouser, Thomas and Guibas, Leonidas and Hanrahan, Pat and Huang, Qixing and Li, Zimo and Savarese, Silvio and Savva, Manolis and Song, Shuran and Su, Hao and Xiao, Jianxiong and Yi, Li and Yu, Fisher},
  number      = {arXiv:1512.03012 [cs.GR]},
  institution = {Stanford University --- Princeton University --- Toyota Technological Institute at Chicago},
  year        = {2015}
}
"""

_DESCRIPTION = """
ShapeNetCore is a densely annotated subset of ShapeNet covering 55 common object
categories with ~51,300 unique 3D models. Each model in ShapeNetCore is linked
to an appropriate synset in WordNet (version 3.0).

The synsets will be extracted from the taxonomy.json file in the ShapeNetCore.v2.zip
archive and the splits from http://shapenet.cs.stanford.edu/shapenet/obj-zip/SHREC16/all.csv
"""

_TAXONOMY_FILE_NAME = 'taxonomy.json'

_SPLIT_FILE_URL = \
    'http://shapenet.cs.stanford.edu/shapenet/obj-zip/SHREC16/all.csv'


class ShapenetConfig(tfds.core.BuilderConfig):
  """Base class for Shapenet BuilderConfigs.

  The Shapenet database builder delegates the implementation of info,
  split_generators and generate_examples to the specified ShapenetConfig. This
  is done to allow multiple versions of the dataset.
  """

  def info(self, dataset_builder):
    """Delegated Shapenet._info."""
    raise NotImplementedError('Abstract method')

  def split_generators(self, dl_manager, dataset_builder):
    """Delegated Shapenet._split_generators."""
    raise NotImplementedError('Abstract method')

  def generate_examples(self, **kwargs):
    """Delegated Shapenet._generate_examples."""
    raise NotImplementedError('Abstract method')


class MeshConfig(ShapenetConfig):
  """A Shapenet config for loading the original .obj files."""

  _MODEL_SUBPATH = os.path.join('models', 'model_normalized.obj')

  def __init__(self, model_subpath=_MODEL_SUBPATH):
    super(MeshConfig, self).__init__(
        name='shapenet_trimesh',
        description=_DESCRIPTION,
        version=tfds.core.Version('1.0.0'))
    self.model_subpath = model_subpath

  def info(self, dataset_builder):
    return tfds.core.DatasetInfo(
        builder=dataset_builder,
        description=_DESCRIPTION,
        features=tfds_features.FeaturesDict({
            'trimesh': tfg_features.TriangleMesh(),
            'label': tfds_features.ClassLabel(num_classes=353),
            'model_id': tfds_features.Text(),
        }),
        supervised_keys=('trimesh', 'label'),
        # Homepage of the dataset for documentation
        homepage='https://shapenet.org/',
        citation=_CITATION,
    )

  def split_generators(self, dl_manager, dataset_builder):
    # Extract the synset ids from the taxonomy file and update the ClassLabel
    # feature.
    with tf.io.gfile.GFile(
        os.path.join(dl_manager.manual_dir,
                     _TAXONOMY_FILE_NAME)) as taxonomy_file:
      labels = [x['synsetId'] for x in json.loads(taxonomy_file.read())]
      # Remove duplicate labels (the json file contains two identical entries
      # for synset '04591713').
      labels = list(collections.OrderedDict.fromkeys(labels))
    dataset_builder.info.features['label'].names = labels

    split_file = dl_manager.download(_SPLIT_FILE_URL)
    fieldnames = ['id', 'synset', 'sub_synset', 'model_id', 'split']
    model_items = collections.defaultdict(list)
    with tf.io.gfile.GFile(split_file) as csvfile:
      for row in csv.DictReader(csvfile, fieldnames):
        model_items[row['split']].append(row)

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                'base_dir': dl_manager.manual_dir,
                'models': model_items['train']
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                'base_dir': dl_manager.manual_dir,
                'models': model_items['test']
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                'base_dir': dl_manager.manual_dir,
                'models': model_items['val']
            },
        ),
    ]

  def generate_examples(self, base_dir, models):
    """Yields examples.

    The structure of the examples:
    {
      'trimesh': tensorflow_graphics.datasets.features.TriangleMesh
      'label': tensorflow_datasets.features.ClassLabel
      'model_id': tensorflow_datasets.features.Text
    }

    Args:
      base_dir: The base directory of shapenet.
      models: The list of models in the split.
    """
    for model in models:
      synset = model['synset']
      model_id = model['model_id']
      model_filepath = os.path.join(base_dir, synset, model_id,
                                    self.model_subpath)
      # If the model doesn't exist, skip it.
      if not tf.io.gfile.exists(model_filepath):
        continue
      yield model_id, {
          'trimesh': model_filepath,
          'label': synset,
          'model_id': model_id,
      }


class Shapenet(tfds.core.GeneratorBasedBuilder):
  """ShapeNetCore V2.

  Example usage of the dataset:

  import tensorflow_datasets as tfds
  from tensorflow_graphics.datasets.shapenet import Shapenet

  data_set = Shapenet.load(
      split='train',
      download_and_prepare_kwargs={
          'download_config':
              tfds.download.DownloadConfig(manual_dir='~/shapenet_base')
      })

  for example in data_set.take(1):
    trimesh, label, model_id = example['trimesh'], example['label'],
                               example['model_id']

  """

  BUILDER_CONFIGS = [MeshConfig()]

  VERSION = tfds.core.Version('1.0.0')

  @staticmethod
  def load(*args, **kwargs):
    return tfds.load('shapenet', *args, **kwargs)  # pytype: disable=wrong-arg-count

  MANUAL_DOWNLOAD_INSTRUCTIONS = textwrap.dedent("""\
  manual_dir should contain the extracted ShapeNetCore.v2.zip archive.
  You need to register on https://shapenet.org/download/shapenetcore in order
  to get the link to download the dataset.
  """)

  def _info(self):
    return self.builder_config.info(self)

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    return self.builder_config.split_generators(dl_manager, self)

  def _generate_examples(self, **kwargs):
    """Yields examples."""
    return self.builder_config.generate_examples(**kwargs)
