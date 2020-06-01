"""pix3d dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets.public_api as tfds

# TODO(pix3d): BibTeX citation
_CITATION = """@inproceedings{pix3d,
  title={Pix3D: Dataset and Methods for Single-Image 3D Shape Modeling},
  author={Sun, Xingyuan and Wu, Jiajun and Zhang, Xiuming and Zhang, Zhoutong and Zhang, Chengkai and Xue, Tianfan and Tenenbaum, Joshua B and Freeman, William T},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
"""

# TODO(pix3d):
_DESCRIPTION = """Pix3D is a large-scale dataset of diverse image-shape pairs with pixel-level 2D-3D alignment. 
It has wide applications in shape-related tasks including reconstruction, retrieval, viewpoint estimation, etc.

Pix3D contains 10,069 2D-3D pairs of 395 distinct 3D shapes, categorised into nine object categories. 
Each sample comprises of an image, 3D shape represented as (non-watertight) triangle mesh and voxel grid, 
bounding-box, segmentation mask, intrinsic and extrinsic camera parameters and 2D and 3D key points. 
"""


class Pix3d(tfds.core.GeneratorBasedBuilder):
  """TODO(pix3d): Short description of my dataset."""

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
            # These are the features of your dataset like images, labels ...
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=(),
        # Homepage of the dataset for documentation
        homepage='https://dataset-homepage/',
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

