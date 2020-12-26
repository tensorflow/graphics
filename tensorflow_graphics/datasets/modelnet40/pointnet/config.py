"""`ModelNet40Config`s for dataset used by Pointnet."""
import os
import h5py
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_graphics.datasets.modelnet40.core import ModelNet40Config, LABELS

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


class PointnetConfig(ModelNet40Config):
  """Config for point cloud data used in Pointnet."""

  def __init__(self):
    super().__init__(name='pointnet',
                     description=_POINTNET_DESCRIPTION,
                     version=tfds.core.Version('1.0.0'))

  def info(self, builder):
    return tfds.core.DatasetInfo(
        builder=builder,
        features=tfds.features.FeaturesDict({
            'points': tfds.features.Tensor(shape=(2048, 3), dtype=tf.float32),
            'label': tfds.features.ClassLabel(names=LABELS)
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
            gen_kwargs=dict(filename_list_path=os.path.join(
                extracted_path, 'modelnet40_ply_hdf5_2048/train_files.txt'),)),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs=dict(filename_list_path=os.path.join(
                extracted_path, 'modelnet40_ply_hdf5_2048/test_files.txt'),)),
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


POINTNET = PointnetConfig()
