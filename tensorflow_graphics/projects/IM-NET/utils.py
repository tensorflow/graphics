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
"""operations and utilities."""

import math
import numpy as np
import tensorflow as tf


def leaky_relu(x, leak=0.02):
  return tf.math.maximum(x, leak * x)


def write_ply_point(name, vertices):
  fout = open(name, 'w')
  fout.write("ply\n")
  fout.write("format ascii 1.0\n")
  fout.write("element vertex "+str(len(vertices))+"\n")
  fout.write("property float x\n")
  fout.write("property float y\n")
  fout.write("property float z\n")
  fout.write("end_header\n")
  for ii in range(len(vertices)):
    fout.write(str(vertices[ii, 0])+" " +
               str(vertices[ii, 1])+" "+str(vertices[ii, 2])+"\n")
  fout.close()


def write_ply_point_normal(name, vertices, normals=None):
  fout = open(name, 'w')
  fout.write("ply\n")
  fout.write("format ascii 1.0\n")
  fout.write("element vertex "+str(len(vertices))+"\n")
  fout.write("property float x\n")
  fout.write("property float y\n")
  fout.write("property float z\n")
  fout.write("property float nx\n")
  fout.write("property float ny\n")
  fout.write("property float nz\n")
  fout.write("end_header\n")
  if normals is None:
    for ii in range(len(vertices)):
      fout.write(str(vertices[ii, 0]) + " "
                 + str(vertices[ii, 1]) + " "
                 + str(vertices[ii, 2]) + " "
                 + str(vertices[ii, 3]) + " "
                 + str(vertices[ii, 4]) + " "
                 + str(vertices[ii, 5])+"\n")
  else:
    for ii in range(len(vertices)):
      fout.write(str(vertices[ii, 0]) + " "
                 + str(vertices[ii, 1]) + " "
                 + str(vertices[ii, 2]) + " "
                 + str(normals[ii, 0]) + " "
                 + str(normals[ii, 1]) + " "
                 + str(normals[ii, 2])+"\n")
  fout.close()


def write_ply_triangle(name, vertices, triangles):
  fout = open(name, 'w')
  fout.write("ply\n")
  fout.write("format ascii 1.0\n")
  fout.write("element vertex "+str(len(vertices))+"\n")
  fout.write("property float x\n")
  fout.write("property float y\n")
  fout.write("property float z\n")
  fout.write("element face "+str(len(triangles))+"\n")
  fout.write("property list uchar int vertex_index\n")
  fout.write("end_header\n")
  for ii in range(len(vertices)):
    fout.write(str(vertices[ii, 0])+" " +
               str(vertices[ii, 1])+" "+str(vertices[ii, 2])+"\n")
  for ii in range(len(triangles)):
    fout.write("3 "+str(triangles[ii, 0])+" " +
               str(triangles[ii, 1])+" "+str(triangles[ii, 2])+"\n")
  fout.close()


def sample_points_triangle(vertices, triangles, num_of_points):
  epsilon = 1e-6
  triangle_area_list = np.zeros([len(triangles)], np.float32)
  triangle_normal_list = np.zeros([len(triangles), 3], np.float32)
  for i in range(len(triangles)):
    # area = |u x v|/2 = |u||v|sin(uv)/2
    a, b, c = vertices[triangles[i, 1]]-vertices[triangles[i, 0]]
    x, y, z = vertices[triangles[i, 2]]-vertices[triangles[i, 0]]
    ti = b*z-c*y
    tj = c*x-a*z
    tk = a*y-b*x
    area2 = math.sqrt(ti*ti+tj*tj+tk*tk)
    if area2 < epsilon:
      triangle_area_list[i] = 0
      triangle_normal_list[i, 0] = 0
      triangle_normal_list[i, 1] = 0
      triangle_normal_list[i, 2] = 0
    else:
      triangle_area_list[i] = area2
      triangle_normal_list[i, 0] = ti/area2
      triangle_normal_list[i, 1] = tj/area2
      triangle_normal_list[i, 2] = tk/area2

  triangle_area_sum = np.sum(triangle_area_list)
  sample_prob_list = (num_of_points/triangle_area_sum)*triangle_area_list

  triangle_index_list = np.arange(len(triangles))

  point_normal_list = np.zeros([num_of_points, 6], np.float32)
  count = 0
  watchdog = 0

  while count < num_of_points:
    np.random.shuffle(triangle_index_list)
    watchdog += 1
    if watchdog > 100:
      print("infinite loop here!")
      return point_normal_list
    for i, _ in enumerate(triangle_index_list):
      if count >= num_of_points:
        break
      dxb = triangle_index_list[i]
      prob = sample_prob_list[dxb]
      prob_i = int(prob)
      prob_f = prob-prob_i
      if np.random.random() < prob_f:
        prob_i += 1
      normal_direction = triangle_normal_list[dxb]
      u = vertices[triangles[dxb, 1]]-vertices[triangles[dxb, 0]]
      v = vertices[triangles[dxb, 2]]-vertices[triangles[dxb, 0]]
      base = vertices[triangles[dxb, 0]]
      for _ in range(prob_i):
        # sample a point here:
        u_x = np.random.random()
        v_y = np.random.random()
        if u_x+v_y >= 1:
          u_x = 1-u_x
          v_y = 1-v_y
        ppp = u*u_x+v*v_y+base

        point_normal_list[count, :3] = ppp
        point_normal_list[count, 3:] = normal_direction
        count += 1
        if count >= num_of_points:
          break

  return point_normal_list
