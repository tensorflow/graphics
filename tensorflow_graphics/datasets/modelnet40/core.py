import abc
import tensorflow_datasets as tfds


LABELS = (
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
    'wardrobe', 'xbox'
)

_MODELNET_CITATION = """\
@inproceedings{wu20153d,
  title={3d shapenets: A deep representation for volumetric shapes},
  author={Wu, Zhirong and Song, Shuran and Khosla, Aditya and Yu, Fisher and Zhang, Linguang and Tang, Xiaoou and Xiao, Jianxiong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1912--1920},
  year={2015}
}
"""


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

  def load(self, *args, **kwargs):
    return tfds.load(f'model_net40/{self.name}', *args, **kwargs)  # pytype: disable=wrong-arg-count

  @abc.abstractmethod
  def generate_examples(self, **kwargs):
    """Delegated GeneratorBaseBuilder._generate_examples"""
    raise NotImplementedError('Abstract method')
