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
# Lint as: python3
"""Postprocess to remove interior backface from reconstruction artifact.
"""
import numpy as np
from scipy import sparse
from scipy import spatial
import trimesh


def merge_meshes(mesh_list):
  """Merge a list of individual meshes into a single mesh."""
  verts = []
  faces = []
  nv = 0
  for m in mesh_list:
    verts.append(m.vertices)
    faces.append(m.faces + nv)
    nv += m.vertices.shape[0]
  v = np.concatenate(verts, axis=0)
  f = np.concatenate(faces, axis=0)
  merged_mesh = trimesh.Trimesh(v, f)
  return merged_mesh


def average_onto_vertex(mesh, per_face_attrib):
  """Average per-face attribute onto vertices."""
  assert per_face_attrib.shape[0] == mesh.faces.shape[0]
  assert len(per_face_attrib.shape) == 1
  c = np.concatenate([[0], per_face_attrib], axis=0)
  v2f_orig = mesh.vertex_faces.copy()
  v2f = v2f_orig.copy()
  v2f += 1
  per_vert_sum = np.sum(c[v2f], axis=1)
  per_vert_count = np.sum(np.logical_not(v2f == 0), axis=1)
  per_vert_attrib = per_vert_sum / per_vert_count
  return per_vert_attrib


def average_onto_face(mesh, per_vert_attrib):
  """Average per-vert attribute onto faces."""
  assert per_vert_attrib.shape[0] == mesh.vertices.shape[0]
  assert len(per_vert_attrib.shape) == 1
  per_face_attrib = per_vert_attrib[mesh.faces]
  per_face_attrib = np.mean(per_face_attrib, axis=1)
  return per_face_attrib


def remove_backface(mesh, pc, k=3, lap_iter=50, lap_val=0.50,
                    area_threshold=1, verbose=False):
  """Remove the interior backface resulting from reconstruction artifacts.

  Args:
    mesh: trimesh instance. mesh recon. from lig that may contain backface.
    pc: np.array of shape [n, 6], original input point cloud.
    k: int, number of nearest neighbor for pooling sign.
    lap_iter: int, number of laplacian smoothing iterations.
    lap_val: float, lambda value for laplacian smoothing of cosine distance.
    area_threshold: float, minimum area connected components to preserve.
    verbose: bool, verbose print.
  Returns:
    mesh_new: trimesh instance. new mesh with backface removed.
  """
  mesh.remove_degenerate_faces()

  v, n = pc[:, :3], pc[:, 3:]

  # build cKDTree to accelerate nearest point search
  if verbose: print("Building KDTree...")
  tree_pc = spatial.cKDTree(data=v)

  # for each vertex, find nearest point in input point cloud
  if verbose: print("{}-nearest neighbor search...".format(k))
  _, idx = tree_pc.query(mesh.vertices, k=k, n_jobs=-1)

  # slice out the nn points
  n_nn = n[idx]  # shape: [#v_query, k, dim]

  # find dot products.
  if verbose: print("Computing norm alignment...")
  n_v = mesh.vertex_normals[:, None, :]  # shape: [#v_query, 1, dim]
  per_vert_norm_alignment = np.sum(n_nn * n_v, axis=-1)  # [#v_query, k]
  per_vert_norm_alignment = np.mean(per_vert_norm_alignment, axis=-1)

  # laplacian smoothing of per vertex normal alignment
  if verbose: print("Computing laplacian smoothing...")
  lap = trimesh.smoothing.laplacian_calculation(mesh)
  dlap = lap.shape[0]
  op = sparse.eye(dlap) + lap_val * (lap - sparse.eye(dlap))
  for _ in range(lap_iter):
    per_vert_norm_alignment = op.dot(per_vert_norm_alignment)

  # average onto face
  per_face_norm_alignment = average_onto_face(mesh, per_vert_norm_alignment)

  # remove faces with per_face_norm_alignment < 0
  if verbose: print("Removing backfaces...")
  ff = mesh.faces[per_face_norm_alignment > -0.75]
  mesh_new = trimesh.Trimesh(mesh.vertices, ff)
  mesh_new.remove_unreferenced_vertices()

  if verbose: print("Cleaning up...")
  mesh_list = mesh_new.split(only_watertight=False)
  # filter out small floating junk from backface
  areas = [m.area for m in mesh_list]
  threshold = min(np.max(areas)/5, area_threshold)
  mesh_list = [m for m in mesh_list if m.area > threshold]
  mesh_new = merge_meshes(mesh_list)

  # fill small holes
  mesh_new.fill_holes()

  return mesh_new
