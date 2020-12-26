import os
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import tensorflow as tf
from tensorflow_graphics.datasets.modelnet40.mesh import off


class OffTest(unittest.TestCase):

  def test_save_load(self):
    mesh = off.random_off(0)
    with TemporaryDirectory() as tmp_dir:
      path = os.path.join(tmp_dir, 'random.off')
      with tf.io.gfile.GFile(path, 'wb') as fp:
        mesh.to_file(fp)
      with tf.io.gfile.GFile(path, 'rb') as fp:
        loaded = off.OffObject.from_file(fp)
    np.testing.assert_equal(mesh.vertices, loaded.vertices)
    np.testing.assert_equal(mesh.face_values, loaded.face_values)
    np.testing.assert_equal(mesh.face_lengths, loaded.face_lengths)


if __name__ == '__main__':
  unittest.main()
