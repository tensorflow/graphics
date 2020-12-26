"""Mesh-based `ModelNet40Config`s."""
import abc
import functools
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_graphics.datasets.modelnet40.core import ModelNet40Config, LABELS, _MODELNET_CITATION
from tensorflow_graphics.datasets.modelnet40.mesh import off
from tensorflow_graphics.datasets.features.trimesh_feature import TriangleMesh

_MESH_DESCRIPTION = """\
The dataset contains triangle mesh data from 40 different categories.

The files have been retrieved from https://modelnet.cs.princeton.edu
"""


class MeshConfig(ModelNet40Config):
  """Config for the original triangle mesh data."""

  def __init__(self):
    super().__init__(name='mesh',
                     version=tfds.core.Version('1.0.0'),
                     description=_MESH_DESCRIPTION)

  def info(self, builder):
    return tfds.core.DatasetInfo(
        builder=builder,
        features=tfds.features.FeaturesDict({
            'mesh': TriangleMesh(),
            'example_index': tfds.features.Tensor(shape=(), dtype=tf.int64),
            'label': tfds.features.ClassLabel(names=LABELS)
        }),
        supervised_keys=('mesh', 'label'),
        homepage='https://modelnet.cs.princeton.edu',
        citation=_MODELNET_CITATION,
    )

  def split_generators(self, download_manager):
    archive_res = download_manager.download(
        'http://modelnet.cs.princeton.edu/ModelNet40.zip')
    archive_iter_fn = functools.partial(download_manager.iter_archive,
                                        archive_res)
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs=dict(archive_iter_fn=archive_iter_fn,
                            split_name='train'),
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs=dict(archive_iter_fn=archive_iter_fn, split_name='test'),
        )
    ]

  def generate_examples(self, archive_iter_fn, split_name):
    for path, fp in archive_iter_fn():
      if not path.endswith('.off'):
        continue
      class_name, split, fn = path.split('/')[-3:]
      if split != split_name:
        continue

      off_obj = off.OffObject.from_file(fp)
      assert np.all(off_obj.face_lengths == 3)
      mesh = dict(vertices=off_obj.vertices.astype(np.float32),
                  faces=np.reshape(off_obj.face_values,
                                   (-1, 3)).astype(np.uint64))

      if mesh is not None:
        yield path, {
            'mesh': mesh,
            'example_index': int(fn.split('_')[-1][:-4]) - 1,
            'label': class_name,
        }


MESH = MeshConfig()
