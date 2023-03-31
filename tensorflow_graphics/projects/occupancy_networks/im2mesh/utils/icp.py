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
from sklearn.neighbors import NearestNeighbors


def best_fit_transform(a, b):
  '''
  Calculates the least-squares best-fit transform that maps corresponding
  points a to b in m spatial dimensions
  Input:
    a: Nxm numpy array of corresponding points
    b: Nxm numpy array of corresponding points
  Returns:
    tm: (m+1)x(m+1) homogeneous transformation matrix that maps a on to b
    r: mxm rotation matrix
    tv: mx1 translation vector
  '''

  assert a.shape == b.shape

  # get number of dimensions
  m = a.shape[1]

  # translate points to their centroids
  centroid_a = np.mean(a, axis=0)
  centroid_b = np.mean(b, axis=0)
  aa = a - centroid_a
  bb = b - centroid_b

  # rotation matrix
  h = np.dot(aa.T, bb)
  u, _, vt = np.linalg.svd(h)
  r = np.dot(vt.T, u.T)

  # special reflection case
  if np.linalg.det(r) < 0:
    vt[m-1, :] *= -1
    r = np.dot(vt.T, u.T)

  # translation
  tv = centroid_b.T - np.dot(r, centroid_a.T)

  # homogeneous transformation
  tm = np.identity(m+1)
  tm[:m, :m] = r
  tm[:m, m] = tv

  return tm, r, tv


def nearest_neighbor(src, dst):
  '''
  Find the nearest (Euclidean) neighbor in dst for each point in src
  Input:
      src: Nxm array of points
      dst: Nxm array of points
  Output:
      distances: Euclidean distances of the nearest neighbor
      indices: dst indices of the nearest neighbor
  '''

  assert src.shape == dst.shape

  neigh = NearestNeighbors(n_neighbors=1)
  neigh.fit(dst)
  distances, indices = neigh.kneighbors(src, return_distance=True)
  return distances.ravel(), indices.ravel()


def icp(a, b, init_pose=None, max_iterations=20, tolerance=0.001):
  '''
  The Iterative Closest Point method: finds best-fit transform that maps
  points a on to points b
  Input:
      a: Nxm numpy array of source mD points
      b: Nxm numpy array of destination mD point
      init_pose: (m+1)x(m+1) homogeneous transformation
      max_iterations: exit algorithm after max_iterations
      tolerance: convergence criteria
  Output:
      t: final homogeneous transformation that maps a on to b
      distances: Euclidean distances (errors) of the nearest neighbor
      i: number of iterations to converge
  '''

  assert a.shape == b.shape

  # get number of dimensions
  m = a.shape[1]

  # make points homogeneous, copy them to maintain the originals
  src = np.ones((m+1, a.shape[0]))
  dst = np.ones((m+1, b.shape[0]))
  src[:m, :] = np.copy(a.T)
  dst[:m, :] = np.copy(b.T)

  # apply the initial pose estimation
  if init_pose is not None:
    src = np.dot(init_pose, src)

  prev_error = 0

  for i in range(max_iterations):
    # find the nearest neighbors between the current source and destination points
    distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

    # compute the transformation between the current source and nearest destination points
    t, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

    # update the current source
    src = np.dot(t, src)

    # check error
    mean_error = np.mean(distances)
    if np.abs(prev_error - mean_error) < tolerance:
      break
    prev_error = mean_error

  # calculate final transformation
  t, _, _ = best_fit_transform(a, src[:m, :].T)

  return t, distances, i
