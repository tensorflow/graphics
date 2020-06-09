"""pix3d dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets import features as tfds_features
from tensorflow_graphics.datasets import features as tfg_features

_CITATION = """@inproceedings{pix3d,
  title={Pix3D: Dataset and Methods for Single-Image 3D Shape Modeling},
  author={Sun, Xingyuan and Wu, Jiajun and Zhang, Xiuming and Zhang, Zhoutong and Zhang, Chengkai and Xue, Tianfan and Tenenbaum, Joshua B and Freeman, William T},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
"""

_DESCRIPTION = """Pix3D is a large-scale dataset of diverse image-shape pairs with pixel-level 2D-3D alignment. 
It has wide applications in shape-related tasks including reconstruction, retrieval, viewpoint estimation, etc.

Pix3D contains 10,069 2D-3D pairs of 395 distinct 3D shapes, categorised into nine object categories. 
Each sample comprises of an image, 3D shape represented as (non-watertight) triangle mesh and voxel grid, 
bounding-box, segmentation mask, intrinsic and extrinsic camera parameters and 2D and 3D key points. 
"""


class Pix3d(tfds.core.GeneratorBasedBuilder):
  """Pix3D is a large-scale dataset of diverse image-shape pairs with pixel-level 2D-3D alignment."""

  # TODO(pix3d): Set up version.
  VERSION = tfds.core.Version('0.1.0')

  def _info(self):
    # TODO(pix3d): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
          'image': tfds_features.Image(shape=(None, None, 3), dtype=tf.uint8),
          'image/filename': tfds_features.Text(),
          'image/source': tfds_features.Text(),
          '2d_keypoints': tfds_features.Tensor(shape=(None, None, 2), dtype=tf.float32),
          'mask': tfds_features.Image(shape=(None, None, 1), dtype=tf.float32),
          'model': tfg_features.TriangleMesh(),
          'model/source': tfds_features.Text(),
          '3d_keypoints': tfds_features.Tensor(shape=(None, 3), dtype=tf.float32),
          'voxel': tfg_features.VoxelGrid(shape=(128, 128, 128)),
          'pose': tfg_features.Pose(),
          'camera': tfg_features.Camera(),
          'category': tfds_features.ClassLabel(shape=(), dtype=tf.int64, num_classes=9),
          'bbox': tfds_features.BBoxFeature(),
          'truncated': tf.bool,
          'occluded': tf.bool,
          'slightly_occluded': tf.bool
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=(),
        # Homepage of the dataset for documentation
        homepage='http://pix3d.csail.mit.edu/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    # TODO(pix3d): Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={},
        ),
    ]

  def _generate_examples(self):
    """Yields examples."""
    # TODO(pix3d): Yields (key, example) tuples from the dataset
    yield 'key', {}

