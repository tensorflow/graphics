import os
import tensorflow_datasets as tfds
from tensorflow_graphics.datasets.modelnet40.mesh import MESH
from tensorflow_graphics.datasets.modelnet40.pointnet import POINTNET

# --- registers the checksum
_CHECKSUM_DIR = os.path.join(os.path.dirname(__file__), 'checksums/')
_CHECKSUM_DIR = os.path.normpath(_CHECKSUM_DIR)
tfds.download.add_checksums_dir(_CHECKSUM_DIR)


class ModelNet40(tfds.core.GeneratorBasedBuilder):
  """ModelNet40."""
  BUILDER_CONFIGS = [POINTNET, MESH]

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
