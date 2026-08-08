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
""" NO COMMENT NOW"""

import numpy as np
import trimesh
from scipy.spatial import cKDTree as KDTree


def compute_trimesh_chamfer(gt_points, gen_mesh, offset, scale,
                            num_mesh_samples=30000):
  """
  Note:
    This function computes a symmetric chamfer distance, \
      i.e. the sum of both chamfers.

  Args:
    gt_points: trimesh.points.PointCloud of just poins, \
      sampled from the surface (see
              compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from \
      whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

  """

  gen_points_sampled = trimesh.sample.sample_surface(
      gen_mesh, num_mesh_samples)[0]

  gen_points_sampled = gen_points_sampled / scale - offset

  # only need numpy array of points
  # gt_points_np = gt_points.vertices
  gt_points_np = gt_points.vertices

  # one direction
  gen_points_kd_tree = KDTree(gen_points_sampled)
  one_distances, _ = gen_points_kd_tree.query(gt_points_np)
  gt_to_gen_chamfer = np.mean(np.square(one_distances))

  # other direction
  gt_points_kd_tree = KDTree(gt_points_np)
  two_distances, _ = gt_points_kd_tree.query(gen_points_sampled)
  gen_to_gt_chamfer = np.mean(np.square(two_distances))

  return gt_to_gen_chamfer + gen_to_gt_chamfer
