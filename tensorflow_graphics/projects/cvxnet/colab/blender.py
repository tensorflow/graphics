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
"""Example of conversion between convex hyperplanes and mesh."""

# --- being forgiving as this is a colab
# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring
# pylint: disable=using-constant-test

import os
import shutil

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import HalfspaceIntersection
import trimesh


def load_npz(path):
  """Load a halfspace definition from a numpy file."""
  data = np.load(path)
  trans = data['trans']
  planes = data['planes']
  T, C, H = planes.shape[0:3]
  return T, C, H, trans, planes


def load_cube():
  """A dummy halfspace definition of a cube."""
  T = 1  #< temporal
  C = 3  #< num convexes
  H = 6  #< num planes
  trans = np.zeros([T, C, 3])
  planes = np.zeros([T, C, H, 4])
  cube = np.array([[-1., 0., 0., -.1], [1., 0., 0., -.1], [0., -1., 0., -.1],
                   [0., 1., 0., -.1], [0., 0., -1., -.1], [0., 0., 1., -.1]])
  planes[0, 0, ...] = cube
  planes[0, 1, ...] = cube
  planes[0, 2, ...] = cube
  trans[0, 0, ...] = np.array([0, 0, 0])
  trans[0, 1, ...] = np.array([+.25, 0, 0])
  trans[0, 2, ...] = np.array([-.25, 0, 0])
  return T, C, H, trans, planes


T, C, H, trans, planes = load_npz('example.npz')
T, C, H, trans, planes = load_cube()


def halfspaces_to_vertices(halfspaces):
  # pre-condition: euclidean origin is within the halfspaces
  # Input: Hx(D+1) numpy array of halfplane constraints
  # Output: convex hull vertices
  n_dims = halfspaces.shape[1]-1
  feasible_point = np.zeros([n_dims,])
  hs = HalfspaceIntersection(halfspaces, feasible_point)
  return hs.intersections


def vertices_to_convex(vertices):
  mesh = trimesh.Trimesh(vertices=vertices, faces=None)
  return mesh.convex_hull.vertices, mesh.convex_hull.faces


def plot_wireframe(ax, vertices, triangles):
  xs, ys, zs = zip(*vertices)
  ax.plot_trisurf(
      xs,
      ys,
      zs,
      triangles=triangles,
      shade=True,
      linewidth=0.1,
      edgecolors=(0, 0, 0),
      antialiased=True)
  ax.set_xlim(-.5, .5)
  ax.set_ylim(-.5, .5)
  ax.set_zlim(-.5, .5)


def makedirs(path):
  os.makedirs(path)


def deletedirs(path):
  if os.path.isdir(path):
    shutil.rmtree(path)


def export_mesh(path, vertices, faces):
  mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
  mesh.export(path)


# --- Reset environment
deletedirs('data/')
fig = plt.figure()
ax = Axes3D(fig)


for iT in range(T):
  iT_folder = 'data/frame_{0:02d}/'.format(iT)
  makedirs(iT_folder)
  obj_shape = planes[iT, ...]
  obj_trans = trans[iT, ...]

  combo = list()
  for iC in range(C):
    cvx_halfspaces = obj_shape[iC, ...]
    cvx_translation = obj_trans[iC, ...]
    vertices = halfspaces_to_vertices(cvx_halfspaces)
    vertices, faces = vertices_to_convex(vertices)
    vertices += cvx_translation
    plot_wireframe(ax, vertices, faces)

    if True:
      iC_folder = iT_folder+'cvx_{0:02d}.obj'.format(iC)
      export_mesh(iC_folder, vertices, faces)
      plot_wireframe(ax, vertices, faces)

    combo.append(trimesh.Trimesh(vertices=vertices, faces=faces))

  if False:
    combo = np.sum(combo)
    combo.export(iT_folder+'/mesh.obj')
    plot_wireframe(ax, combo.vertices, combo.faces)

  plt.show()
