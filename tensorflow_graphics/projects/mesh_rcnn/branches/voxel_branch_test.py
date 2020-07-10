"""
Created by Robin Baumann <https://github.com/RobinBaumann> at June 26, 2020.
"""
import numpy as np
import tensorflow as tf

from tensorflow_graphics.projects.mesh_rcnn.branches.voxel_branch import cubify
from tensorflow_graphics.util import test_case


class CufifyTest(test_case.TestCase):

  def test_all_below_threshold(self):
    N, V = 32, 16
    voxels = tf.random.uniform((N, V, V, V), minval=0, maxval=0.5,
                               dtype=tf.float32)
    vertices, faces = cubify(voxels, threshold=0.7)
    self.assertEmpty(vertices[0])
    self.assertEmpty(faces[0])

  def test_cubify_on_cube(self):
    N, V = 2, 2

    # top left corner in front plane is 1, everything else empty
    one_voxel = tf.constant([[[1., 0.],
                              [0., 0.]],
                             [[0., 0.],
                              [0., 0.]]],
                            dtype=tf.float32)
    full_cube = tf.ones((V, V, V), dtype=tf.float32)

    test_data = tf.stack([one_voxel, full_cube])

    self.assertShapeEqual(np.array([N, V, V, V]), test_data)

    vertices, faces = cubify(test_data, 0.5)

    # ~~~~~~~~~~ Test first batch element ~~~~~~~~~~ #
    expected_vertices_topleftnear = tf.constant(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ], dtype=tf.float32
    )

    expected_faces_topleftnear = tf.constant(
        [
            [0, 1, 4],
            [1, 5, 4],
            [4, 5, 6],
            [5, 7, 6],
            [0, 4, 6],
            [0, 6, 2],
            [0, 3, 1],
            [0, 2, 3],
            [6, 7, 3],
            [6, 3, 2],
            [1, 7, 5],
            [1, 3, 7],
        ], dtype=tf.float32
    )

    self.assertAllClose(vertices[0], expected_vertices_topleftnear)
    self.assertAllClose(faces[0], expected_faces_topleftnear)

    # ~~~~~~~~~~ Test second batch element ~~~~~~~~~~ #
    expected_vertices_full = tf.constant(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ], dtype=tf.float32
    )

    expected_faces_full = tf.constant(
        [
            [0, 1, 4],
            [1, 5, 4],
            [4, 5, 6],
            [5, 7, 6],
            [0, 4, 6],
            [0, 6, 2],
            [0, 3, 1],
            [0, 2, 3],
            [6, 7, 3],
            [6, 3, 2],
            [1, 7, 5],
            [1, 3, 7],
        ], dtype=tf.float32
    )

    self.assertAllClose(vertices[1], expected_vertices_full)
    self.assertAllClose(faces[1], expected_faces_full)
