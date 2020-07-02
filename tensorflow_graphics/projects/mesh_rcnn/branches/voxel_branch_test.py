"""
Created by Robin Baumann <https://github.com/RobinBaumann> at June 26, 2020.
"""

import tensorflow.compat.v1 as tf
from tensorflow_graphics.util import test_case
from tensorflow_graphics.projects.mesh_rcnn.branches.voxel_branch import cubify

class CufifyTest(test_case.TestCase):

  def test_all_below_threshold(self):
    N, V = 32, 16
    voxels = tf.random.uniform((N, V, V, V), minval=0, maxval=0.5, dtype=tf.float32)
    vertices, faces = cubify(voxels, threshold=0.7)
    self.assertEmpty(vertices)
    self.assertEmpty(faces)
