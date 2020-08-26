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
"""General helper functions."""
from os import path

import numpy as np
from skimage import measure
import tensorflow.compat.v1 as tf

from tensorflow_graphics.projects.cvxnet.lib.libmise import mise
from tensorflow_graphics.projects.nasa.lib import datasets
from tensorflow_graphics.projects.nasa.lib import models

import tensorflow_probability as tfp
from tqdm import trange
import trimesh

tf.disable_eager_execution()
tfd = tfp.distributions


def define_flags():
  """Define command line flags."""
  flags = tf.app.flags

  # Dataset Parameters
  flags.DEFINE_enum("dataset", "amass",
                    list(k for k in datasets.dataset_dict.keys()),
                    "Name of the dataset.")
  flags.DEFINE_string("data_dir", None, "Directory to load data from.")
  flags.mark_flag_as_required("data_dir")
  flags.DEFINE_integer("sample_bbox", 1024, "Number of bbox samples.")
  flags.DEFINE_integer("sample_surf", 1024, "Number of surface samples.")
  flags.DEFINE_integer("batch_size", 12, "Batch size.")
  flags.DEFINE_integer("motion", 0, "Index of the motion for evaluation.")
  flags.DEFINE_integer("subject", 0, "Index of the subject for training.")

  # Model Parameters
  flags.DEFINE_enum("model", "nasa", list(k for k in models.model_dict.keys()),
                    "Name of the model.")
  flags.DEFINE_integer("n_parts", 24, "Number of parts.")
  flags.DEFINE_integer("total_dim", 960,
                       "Dimension of the latent vector (in total).")
  flags.DEFINE_bool("shared_decoder", False, "Whether to use shared decoder.")
  flags.DEFINE_float("soft_blend", 5., "The constant to blend parts.")
  flags.DEFINE_bool("projection", True,
                    "Whether to use projected shape features.")
  flags.DEFINE_float("level_set", 0.5, "The value of the level_set.")
  flags.DEFINE_integer("n_dims", 3, "The dimension of the query points.")

  # Training Parameters
  flags.DEFINE_float("lr", 1e-4, "Learning rate")
  flags.DEFINE_string("train_dir", None, "Training directory.")
  flags.mark_flag_as_required("train_dir")
  flags.DEFINE_integer("max_steps", 200000, "Number of optimization steps.")
  flags.DEFINE_integer("save_every", 5000,
                       "Number of steps to save checkpoint.")
  flags.DEFINE_integer("summary_every", 500,
                       "Number of steps to save checkpoint.")
  flags.DEFINE_float("label_w", 0.5, "Weight of labed vertices loss.")
  flags.DEFINE_float("minimal_w", 0.05, "Weight of minimal loss.")
  flags.DEFINE_bool("use_vert", True,
                    "Whether to use vertices on the mesh for training.")
  flags.DEFINE_bool("use_joint", True,
                    "Whether to use joint-based transformation.")
  flags.DEFINE_integer("sample_vert", 2048, "Number of vertex samples.")

  # Evalulation Parameters
  flags.DEFINE_bool("gen_mesh_only", False, "Whether to generate meshes only.")

  # Tracking Parameters
  flags.DEFINE_float("theta_lr", 5e-4, "Learning rate")
  flags.DEFINE_integer("max_steps_per_frame", 1792,
                       "Number of optimization steps for tracking each frame.")
  flags.DEFINE_enum("gradient_type", "reparam", ["vanilla", "reparam"],
                    "Type of gradient to use in theta optimization.")
  flags.DEFINE_integer("sample_track_vert", 1024,
                       "Number of vertex samples for tracking each frame.")
  flags.DEFINE_integer("n_noisy_samples", 8,
                       "Number of noisy samples per vertex")
  flags.DEFINE_float("bandwidth", 1e-2, "Bandwidth of the gaussian noises.")
  flags.DEFINE_bool(
      "left_trans", False,
      "Whether to use left side transformation (True) or right side (False).")
  flags.DEFINE_string("joint_data", None, "Path to load joint data.")
  flags.DEFINE_float("glue_w", 20., "Weight of length constraint loss.")
  flags.DEFINE_float("trans_range", 1., "The range of allowed translations.")


def gen_mesh(sess,
             feed_dict,
             latent_holder,
             point_holder,
             latent,
             occ,
             batch_val,
             hparams,
             idx=0):
  """Generating meshes given a trained NASA model."""
  scale = 1.1  # Scale of the padded bbox regarding the tight one.
  level_set = hparams.level_set
  latent_val = sess.run(latent, feed_dict)
  mesh_extractor = mise.MISE(32, 3, level_set)
  points = mesh_extractor.query()
  gt_verts = batch_val["vert"].reshape([-1, 3])
  gt_bbox = np.stack([gt_verts.min(axis=0), gt_verts.max(axis=0)], axis=0)
  gt_center = (gt_bbox[0] + gt_bbox[1]) * 0.5
  gt_scale = (gt_bbox[1] - gt_bbox[0]).max()

  while points.shape[0] != 0:
    orig_points = points
    points = points.astype(np.float32)
    points = (np.expand_dims(points, axis=0) / mesh_extractor.resolution -
              0.5) * scale
    points = points * gt_scale + gt_center
    n_points = points.shape[1]
    values = []
    for i in range(0, n_points,
                   100000):  # Add this to prevent OOM due to points overload.
      feed_dict[latent_holder] = latent_val
      feed_dict[point_holder] = np.expand_dims(points[:, i:i + 100000], axis=1)
      value = sess.run(occ[:, idx], feed_dict)
      values.append(value)
    values = np.concatenate(values, axis=1)
    values = values[0, :, 0].astype(np.float64)
    mesh_extractor.update(orig_points, values)
    points = mesh_extractor.query()
  value_grid = mesh_extractor.to_dense()

  try:
    value_grid = np.pad(value_grid, 1, "constant", constant_values=-1e6)
    verts, faces, normals, unused_var = measure.marching_cubes_lewiner(
        value_grid, min(level_set, value_grid.max()))
    del normals
    verts -= 1
    verts /= np.array([
        value_grid.shape[0] - 3, value_grid.shape[1] - 3,
        value_grid.shape[2] - 3
    ],
                      dtype=np.float32)
    verts = scale * (verts - 0.5)
    verts = verts * gt_scale + gt_center
    faces = np.stack([faces[..., 1], faces[..., 0], faces[..., 2]], axis=-1)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return mesh
  except:  # pylint: disable=bare-except
    return None


def save_mesh(sess,
              feed_dict,
              latent_holder,
              point_holder,
              latent,
              occ,
              batch_val,
              hparams,
              pth="meshes"):
  """Generate and save meshes to disk given a trained NASA model."""
  name = batch_val["name"][0].decode("utf-8")
  subject, motion, frame = amass_name_helper(name)
  pth = path.join(hparams.train_dir, pth, frame)
  if not tf.io.gfile.isdir(pth):
    tf.io.gfile.makedirs(pth)

  start = hparams.n_parts
  for i in range(start, hparams.n_parts + 1):
    mesh_model = gen_mesh(
        sess,
        feed_dict,
        latent_holder,
        point_holder,
        latent,
        occ,
        batch_val,
        hparams,
        idx=i)
    mesh_name = "full_pred.obj"
    if mesh_model is not None:
      with tf.io.gfile.GFile(path.join(pth, mesh_name), "w") as fout:
        mesh_model.export(fout, file_type="obj")

  return subject, motion, frame, mesh_model


def save_pointcloud(data, hparams, pth="pointcloud"):
  """Save pointcloud to disk."""
  name = data["name"][0].decode("utf-8")
  unused_subject, unused_motion, frame = amass_name_helper(name)
  pth = path.join(hparams.train_dir, pth, frame)
  if not tf.io.gfile.isdir(pth):
    tf.io.gfile.makedirs(pth)

  mesh_name = "pointcloud.obj"
  with tf.io.gfile.GFile(path.join(pth, mesh_name), "w") as fout:
    pointcloud = data["vert"].reshape([-1, 3])
    for v in pointcloud:
      fout.write("v {0} {1} {2}\n".format(*v.tolist()))


def amass_name_helper(name):
  name, frame = name.split("-")
  subject = name[:5]
  motion = name[6:]
  return subject, motion, frame


def make_summary_feed_dict(
    iou_hook,
    iou,
    best_hook,
    best_iou,
):
  feed_dict = {}
  feed_dict[iou_hook] = iou
  feed_dict[best_hook] = best_iou
  return feed_dict


def parse_global_step(ckpt):
  basename = path.basename(ckpt)
  return int(basename.split("-")[-1])


def compute_iou(sess, feed_dict, latent_holder, point_holder, latent, occ,
                point, label, hparams):
  """Compute IoU."""
  iou = 0.
  eps = 1e-9
  latent_val = sess.run(latent, feed_dict)
  n_points = point.shape[2]
  preds = []
  for start in range(0, n_points, 100000):
    feed_dict[point_holder] = point[:, :, start:start + 100000]
    feed_dict[latent_holder] = latent_val
    pred = sess.run(occ, feed_dict)
    preds.append(pred)
  pred = np.concatenate(preds, axis=2)
  pred = (pred >= hparams.level_set).astype(np.float32)
  label = (label[:, :1] >= 0.5).astype(np.float32).squeeze(axis=1)
  iou += np.sum(pred * label) / np.maximum(np.sum(np.maximum(pred, label)), eps)
  return iou


def compute_glue_loss(connect, end_pts, inv_transforms, inv_first_frame_trans,
                      joints, hparams):
  """Compute the prior term as a glue loss."""
  n_dims = hparams.n_dims

  # Invert the transformation
  r_inv = inv_transforms[..., :n_dims, :n_dims]
  t_inv = inv_transforms[..., :n_dims, -1:]
  r = tf.transpose(r_inv, [0, 2, 1])
  t = -tf.matmul(r, t_inv)
  transforms = tf.concat(
      [tf.concat([r, t], axis=-1), inv_transforms[..., -1:, :]], axis=-2)
  transforms = tf.matmul(transforms, inv_first_frame_trans)

  # Compute transformations of father joints and apply it to vectors from frame0
  father_transforms = tf.reduce_sum(
      tf.expand_dims(transforms, axis=1) *
      connect.reshape([hparams.n_parts, hparams.n_parts, 1, 1]),
      axis=0)
  end_pts_homo = tf.expand_dims(
      tf.concat([end_pts, tf.ones_like(end_pts[..., :1])], axis=-1), axis=-1)
  end_pts_transformed = tf.matmul(father_transforms, end_pts_homo)
  end_pts_transformed = tf.squeeze(end_pts_transformed, axis=-1)[..., :n_dims]

  # Compute vectors in current configuration
  pred_links = tf.reshape(joints, [hparams.n_parts, n_dims])

  # Compute distance between links and transformed vectors
  return tf.reduce_sum(tf.square(pred_links - end_pts_transformed))


def vanilla_theta_gradient(model_fn, batch_holder, hparams):
  """A vanilla gradient estimator for the pose, theta."""
  latent_holder, latent, occ_eval = model_fn(batch_holder, None, None,
                                             "gen_mesh")
  if hparams.sample_vert > 0:
    points = batch_holder["point"]
    weights = batch_holder["weight"]
    n_vert = tf.shape(points)[2]
    sample_indices = tf.random.uniform([1, 1, hparams.sample_vert],
                                       minval=0,
                                       maxval=n_vert,
                                       dtype=tf.int32)
    points = tf.gather(points, sample_indices, axis=2, batch_dims=2)
    weights = tf.gather(weights, sample_indices, axis=2, batch_dims=2)
    batch_holder["point"] = points
    batch_holder["weight"] = weights
  unused_var0, unused_var1, occ = model_fn(batch_holder, None, None, "gen_mesh")
  return latent_holder, latent, occ_eval, tf.reduce_mean(
      tf.square(occ - hparams.level_set))


def reparam_theta_gradient(model_fn, batch_holder, hparams):
  """A gradient estimaor for the pose, theta, using the reparam trick."""
  sigma = hparams.bandwidth
  n_samples = hparams.n_noisy_samples
  latent_holder, latent, occ_eval = model_fn(batch_holder, None, None,
                                             "gen_mesh")

  if hparams.sample_vert > 0:
    points = batch_holder["point"]
    weights = batch_holder["weight"]
    n_vert = tf.shape(points)[2]
    sample_indices = tf.random.uniform([1, 1, hparams.sample_vert],
                                       minval=0,
                                       maxval=n_vert,
                                       dtype=tf.int32)
    points = tf.gather(points, sample_indices, axis=2, batch_dims=2)
    weights = tf.gather(weights, sample_indices, axis=2, batch_dims=2)
    batch_holder["point"] = points
    batch_holder["weight"] = weights
  dist = tfd.Normal(loc=0., scale=sigma)
  n_pts = hparams.sample_vert if hparams.sample_vert > 0 else hparams.n_vert
  noises = dist.sample((1, hparams.n_parts, n_pts, n_samples, hparams.n_dims))
  unused_var0, unused_var1, occ = model_fn(batch_holder, noises, None,
                                           "gen_mesh")
  occ = tf.reshape(occ, [1, hparams.n_parts + 1, -1, n_samples, 1])

  occ = tf.reduce_mean(occ[:, hparams.n_parts:], axis=3)
  return latent_holder, latent, occ_eval, tf.reduce_mean(
      tf.square(occ - hparams.level_set))


def optimize_theta(feed_dict, loss, reset_op, train_op, rec_loss, glue_loss,
                   sess, k, hparams):
  """Optimize the pose, theta, during tracking."""
  sess.run(reset_op)
  loss_val = 0
  glue_val = 0
  with trange(hparams.max_steps_per_frame) as t:
    for unused_i in t:
      loss_val, unused_var, rec_val, glue_val = sess.run(
          [loss, train_op, rec_loss, glue_loss], feed_dict)
      t.set_description("Frame_{0} {1:.4f}|{2:.4f}".format(
          k, rec_val, glue_val))
  return loss_val, glue_val
