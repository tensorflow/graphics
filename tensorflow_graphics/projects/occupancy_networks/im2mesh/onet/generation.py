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


import time
import trimesh
from tqdm import trange
import numpy as np
import tensorflow as tf
from im2mesh.utils import libmcubes
from im2mesh.common import make_3d_grid
from im2mesh.utils.libsimplify import simplify_mesh
from im2mesh.utils.libmise import MISE


class Generator3D(object):
  '''  Generator class for Occupancy Networks.
  It provides functions to generate the final mesh as well refining options.
  Args:
      model (tf.keras.layers.Model): trained Occupancy Network model
      points_batch_size (int): batch size for points evaluation
      threshold (float): threshold value
      refinement_step (int): number of refinement steps
      resolution0 (int): start resolution for MISE
      upsampling steps (int): number of upsampling steps
      with_normals (bool): whether normals should be estimated
      padding (float): how much padding should be used for MISE
      sample (bool): whether z should be sampled
      simplify_nfaces (int): number of faces the mesh should be simplified to
      preprocessor (tf.keras.Models): preprocessor for inputs
  '''

  def __init__(self, model, points_batch_size=100000,
               threshold=0.5, refinement_step=0,
               resolution0=16, upsampling_steps=3,
               with_normals=False, padding=0.1, sample=False,
               simplify_nfaces=None,
               preprocessor=None):
    self.model = model
    self.points_batch_size = points_batch_size
    self.refinement_step = refinement_step
    self.threshold = threshold
    self.resolution0 = resolution0
    self.upsampling_steps = upsampling_steps
    self.with_normals = with_normals
    self.padding = padding
    self.sample = sample
    self.simplify_nfaces = simplify_nfaces
    self.preprocessor = preprocessor

  def generate_mesh(self, data, return_stats=True):
    ''' Generates the output mesh.
    Args:
        data (tensor): data tensor
        return_stats (bool): whether stats should be returned
    '''
    self.model.trainable = False  # TODO: CHECK
    stats_dict = {}

    inputs = data.get('inputs', tf.zeros([1, 0]))
    kwargs = {}

    # Preprocess if requires
    if self.preprocessor is not None:
      t0 = time.time()
      inputs = self.preprocessor(inputs)
      stats_dict['time (preprocess)'] = time.time() - t0

    # Encode inputs
    t0 = time.time()
    c = self.model.encode_inputs(inputs, training=False)
    stats_dict['time (encode inputs)'] = time.time() - t0

    z = self.model.get_z_from_prior((1,), sample=self.sample)
    mesh = self.generate_from_latent(z, c, stats_dict=stats_dict, **kwargs)

    if return_stats:
      return mesh, stats_dict
    else:
      return mesh

  def generate_from_latent(self, z, c=None, stats_dict={}, **kwargs):
    ''' Generates mesh from latent.
    Args:
        z (tensor): latent code z
        c (tensor): latent conditioned code c
        stats_dict (dict): stats dictionary
    '''
    threshold = np.log(self.threshold) - np.log(1. - self.threshold)

    t0 = time.time()
    # Compute bounding box size
    box_size = 1 + self.padding

    # Shortcut
    if self.upsampling_steps == 0:
      nx = self.resolution0
      pointsf = box_size * make_3d_grid(
          (-0.5,) * 3, (0.5,) * 3, (nx,) * 3
      )
      values = self.eval_points(pointsf, z, c, **kwargs).numpy()
      value_grid = values.reshape(nx, nx, nx)
    else:
      mesh_extractor = MISE(
          self.resolution0, self.upsampling_steps, threshold)

      points = mesh_extractor.query()

      while points.shape[0] != 0:
        # Query points
        pointsf = tf.convert_to_tensor(points)
        # Normalize to bounding box
        pointsf = pointsf / mesh_extractor.resolution
        pointsf = box_size * (pointsf - 0.5)
        # Evaluate model and update
        values = self.eval_points(
            pointsf, z, c, **kwargs).numpy()
        values = values.astype(np.float64)
        mesh_extractor.update(points, values)
        points = mesh_extractor.query()

      value_grid = mesh_extractor.to_dense()

    # Extract mesh
    stats_dict['time (eval points)'] = time.time() - t0

    mesh = self.extract_mesh(value_grid, z, c, stats_dict=stats_dict)
    return mesh

  def eval_points(self, p, z, c=None, **kwargs):
    ''' Evaluates the occupancy values for the points.
    Args:
        p (tensor): points
        z (tensor): latent code z
        c (tensor): latent conditioned code c
    '''
    p = tf.expand_dims(p, 0)
    occ_hat = self.model.decode(p, z, c, **kwargs).logits
    occ_hat = tf.squeeze(occ_hat, axis=0)

    # TODO:
    # tf.split can't separate a tensor by numbers that can't divide tensor's dim. that is a problem.
    # p_split = tf.split(p, self.points_batch_size)
    # occ_hats = []

    # for pi in p_split:
    #   pi = tf.expand_dims(pi, 0)
    #   occ_hat = self.model.decode(pi, z, c, **kwargs).logits

    #   occ_hats.append(tf.squeeze(occ_hat, 0))

    # occ_hat = tf.concat(occ_hats, axis=0)

    return occ_hat

  def extract_mesh(self, occ_hat, z, c=None, stats_dict=dict()):
    ''' Extracts the mesh from the predicted occupancy grid.
    Args:
        occ_hat (tensor): value grid of occupancies
        z (tensor): latent code z
        c (tensor): latent conditioned code c
        stats_dict (dict): stats dictionary
    '''
    # Some short hands
    n_x, n_y, n_z = occ_hat.shape
    box_size = 1 + self.padding
    threshold = np.log(self.threshold) - np.log(1. - self.threshold)
    # Make sure that mesh is watertight
    t0 = time.time()
    occ_hat_padded = np.pad(
        occ_hat, 1, 'constant', constant_values=-1e6)
    vertices, triangles = libmcubes.marching_cubes(
        occ_hat_padded, threshold)
    stats_dict['time (marching cubes)'] = time.time() - t0
    # Strange behaviour in libmcubes: vertices are shifted by 0.5
    vertices -= 0.5
    # Undo padding
    vertices -= 1
    # Normalize to bounding box
    vertices /= np.array([n_x-1, n_y-1, n_z-1])
    vertices = box_size * (vertices - 0.5)

    # mesh_pymesh = pymesh.form_mesh(vertices, triangles)
    # mesh_pymesh = fix_pymesh(mesh_pymesh)

    # Estimate normals if needed
    if self.with_normals and not vertices.shape[0] == 0:
      t0 = time.time()
      normals = self.estimate_normals(vertices, z, c)
      stats_dict['time (normals)'] = time.time() - t0

    else:
      normals = None

    # Create mesh
    mesh = trimesh.Trimesh(vertices, triangles,
                           vertex_normals=normals,
                           process=False)

    # Directly return if mesh is empty
    if vertices.shape[0] == 0:
      return mesh

    # TODO: normals are lost here
    if self.simplify_nfaces is not None:
      t0 = time.time()
      mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
      stats_dict['time (simplify)'] = time.time() - t0

    # Refine mesh
    if self.refinement_step > 0:
      t0 = time.time()
      self.refine_mesh(mesh, occ_hat, z, c)
      stats_dict['time (refine)'] = time.time() - t0

    return mesh

  def estimate_normals(self, vertices, z, c=None):
    ''' Estimates the normals by computing the gradient of the objective.
    Args:
        vertices (numpy array): vertices of the mesh
        z (tensor): latent code z
        c (tensor): latent conditioned code c
    '''
    vertices = tf.convert_to_tensor(vertices)
    vertices_split = tf.split(vertices, self.points_batch_size)

    normals = []
    z, c = tf.expand_dims(z, axis=0), tf.expand_dims(c, axis=0)
    for vi in vertices_split:
      vi = tf.expand_dims(vi, axis=0)
      vi.requires_grad_()
      with tf.GradientTape() as tape:
        occ_hat = self.model.decode(vi, z, c).logits
        out = occ_hat.sum()
      ni = -tape.gradient(out, vi)
      ni = ni / tf.norm(ni, axis=-1, keepdims=True)
      ni = tf.squeeze(tf, axis=0).numpy()
      normals.append(ni)

    normals = np.concatenate(normals, axis=0)
    return normals

  def refine_mesh(self, mesh, occ_hat, z, c=None):
    ''' Refines the predicted mesh.
    Args:
        mesh (trimesh object): predicted mesh
        occ_hat (tensor): predicted occupancy grid
        z (tensor): latent code z
        c (tensor): latent conditioned code c
    '''

    self.model.trainable = False  # TODO CHECK

    # Some shorthands
    n_x, n_y, n_z = occ_hat.shape
    assert n_x == n_y == n_z
    # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
    threshold = self.threshold

    # Vertex parameter
    v = tf.convert_to_tensor(mesh.vertices)

    # Faces of mesh
    faces = tf.convert_to_tensor(mesh.faces, dtype=tf.int64)

    # Start optimization
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4)

    for it_r in trange(self.refinement_step):
      with tf.GradientTape() as tape:
        # Loss
        face_vertex = v[faces]
        eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
        eps = tf.convert_to_tensor(eps)

        with tf.GradientTape() as tape2:
          face_point = tf.reduce_sum(
              face_vertex * eps[:, :, None], axis=1)

          face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
          face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
          face_normal = tf.linalg.cross(face_v1, face_v2)
          face_normal = face_normal / \
              (tf.norm(
                  face_normal.norm, axis=1, keepdims=True) + 1e-10)
          face_value = tf.math.sigmoid(self.model.decode(
              tf.expand_dims(face_point, axis=0), z, c).logits)
          face_value_sum = tf.reduce_sum(face_value, axis=0)

        normal_target = -tape2.gradient(face_value_sum, face_point)[0]
        normal_target = normal_target / \
            (tf.norm(normal_target, axis=1, keepdims=True) + 1e-10)

        loss_target = tf.reduce_mean(
            tf.math.pow(face_value - threshold, 2))
        loss_normal = tf.reduce_mean(tf.reduce_sum(
            tf.math.pow(face_normal - normal_target, 2), axis=1))

        loss = loss_target + 0.01 * loss_normal

      # Update
      grad = tape.gradient(loss, v)
      optimizer.apply_gradients(zip(grad, v))

    mesh.vertices = v.data.numpy()

    return mesh
