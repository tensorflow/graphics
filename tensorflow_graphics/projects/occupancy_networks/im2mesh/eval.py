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

# from im2mesh import icp
import logging
import numpy as np
import trimesh

# from scipy.spatial import cKDTree
from im2mesh.utils.libkdtree import KDTree
from im2mesh.utils.libmesh import check_mesh_contains
from im2mesh.common import compute_iou


# Maximum values for bounding box [-0.5, 0.5]^3
EMPTY_PCL_DICT = {
    "completeness": np.sqrt(3),
    "accuracy": np.sqrt(3),
    "completeness2": 3,
    "accuracy2": 3,
    "chamfer": 6,
}

EMPTY_PCL_DICT_NORMALS = {
    "normals completeness": -1.0,
    "normals accuracy": -1.0,
    "normals": -1.0,
}

logger = logging.getLogger(__name__)


class MeshEvaluator(object):
  """ Mesh evaluation class.

  It handles the mesh evaluation process.

  Args:
      n_points (int): number of points to be used for evaluation
  """

  def __init__(self, n_points=100000):
    self.n_points = n_points

  def eval_mesh(self, mesh, pointcloud_tgt, normals_tgt, points_iou, occ_tgt):
    """ Evaluates a mesh.

    Args:
        mesh (trimesh): mesh which should be evaluated
        pointcloud_tgt (numpy array): target point cloud
        normals_tgt (numpy array): target normals
        points_iou (numpy_array): points tensor for IoU evaluation
        occ_tgt (numpy_array): GT occupancy values for IoU points
    """
    if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
      pointcloud, idx = mesh.sample(self.n_points, return_index=True)
      pointcloud = pointcloud.astype(np.float32)
      normals = mesh.face_normals[idx]
    else:
      pointcloud = np.empty((0, 3))
      normals = np.empty((0, 3))

    out_dict = self.eval_pointcloud(
        pointcloud, pointcloud_tgt, normals, normals_tgt
    )

    if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
      occ = check_mesh_contains(mesh, points_iou)
      out_dict["iou"] = compute_iou(occ, occ_tgt)
    else:
      out_dict["iou"] = 0.0

    return out_dict

  def eval_pointcloud(
      self, pointcloud, pointcloud_tgt, normals=None, normals_tgt=None
  ):
    """ Evaluates a point cloud.

    Args:
        pointcloud (numpy array): predicted point cloud
        pointcloud_tgt (numpy array): target point cloud
        normals (numpy array): predicted normals
        normals_tgt (numpy array): target normals
    """
    # Return maximum losses if pointcloud is empty
    if pointcloud.shape[0] == 0:
      logger.warning("Empty pointcloud / mesh detected!")
      out_dict = EMPTY_PCL_DICT.copy()
      if normals is not None and normals_tgt is not None:
        out_dict.update(EMPTY_PCL_DICT_NORMALS)
      return out_dict

    pointcloud = np.asarray(pointcloud)
    pointcloud_tgt = np.asarray(pointcloud_tgt)

    # Completeness: how far are the points of the target point cloud
    # from thre predicted point cloud
    completeness, completeness_normals = distance_p2p(
        pointcloud_tgt, normals_tgt, pointcloud, normals
    )
    completeness2 = completeness ** 2

    completeness = completeness.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals.mean()

    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(
        pointcloud, normals, pointcloud_tgt, normals_tgt
    )
    accuracy2 = accuracy ** 2

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals.mean()

    # Chamfer distance
    chamfer_l2 = 0.5 * (completeness2 + accuracy2)
    normals_correctness = 0.5 * completeness_normals + 0.5 * accuracy_normals
    chamfer_l1 = 0.5 * (completeness + accuracy)

    out_dict = {
        "completeness": completeness,
        "accuracy": accuracy,
        "normals completeness": completeness_normals,
        "normals accuracy": accuracy_normals,
        "normals": normals_correctness,
        "completeness2": completeness2,
        "accuracy2": accuracy2,
        "chamfer-L2": chamfer_l2,
        "chamfer-L1": chamfer_l1,
    }

    return out_dict


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
  """ Computes minimal distances of each point in points_src to points_tgt.

  Args:
      points_src (numpy array): source points
      normals_src (numpy array): source normals
      points_tgt (numpy array): target points
      normals_tgt (numpy array): target normals
  """
  kdtree = KDTree(points_tgt)
  dist, idx = kdtree.query(points_src)

  if normals_src is not None and normals_tgt is not None:
    normals_src = normals_src / \
        np.linalg.norm(normals_src, axis=-1, keepdims=True)
    normals_tgt = normals_tgt / \
        np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

    normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
    # Handle normals that point into wrong direction gracefully
    # (mostly due to mehtod not caring about this in generation)
    normals_dot_product = np.abs(normals_dot_product)
  else:
    normals_dot_product = np.array(
        [np.nan] * points_src.shape[0], dtype=np.float32)
  return dist, normals_dot_product


def distance_p2m(points, mesh):
  """ Compute minimal distances of each point in points to mesh.

  Args:
      points (numpy array): points array
      mesh (trimesh): mesh

  """
  _, dist, _ = trimesh.proximity.closest_point(mesh, points)
  return dist
