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
"""Utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from os import path
import numpy as np
import scipy as sp
from skimage import measure
import tensorflow.compat.v1 as tf

from tensorflow_graphics.projects.cvxnet.lib import datasets
from tensorflow_graphics.projects.cvxnet.lib import models
from tensorflow_graphics.projects.cvxnet.lib.libmise import mise

import trimesh

Stats = collections.namedtuple("Stats", ["iou", "chamfer", "fscore"])
SYSNET_CLASSES = {
    "02691156": "airplane",
    "02933112": "cabinet",
    "03001627": "chair",
    "03636649": "lamp",
    "04090263": "rifle",
    "04379243": "table",
    "04530566": "watercraft",
    "02828884": "bench",
    "02958343": "car",
    "03211117": "display",
    "03691459": "speaker",
    "04256520": "sofa",
    "04401088": "telephone",
    "all": "all",
}


def define_flags():
  """Define command line flags."""
  flags = tf.app.flags
  # Model flags
  flags.DEFINE_enum("model", "multiconvex",
                    list(k for k in models.model_dict.keys()),
                    "Name of the model.")
  flags.DEFINE_float("sharpness", 75., "Sharpness term.")
  flags.DEFINE_integer("n_parts", 50, "Number of convexes uesd.")
  flags.DEFINE_integer("n_half_planes", 25, "Number of half spaces used.")
  flags.DEFINE_integer("latent_size", 256, "The size of latent code.")
  flags.DEFINE_integer("dims", 3, "The dimension of query points.")
  flags.DEFINE_bool("image_input", False, "Use color images as input if True.")
  flags.DEFINE_float("vis_scale", 1.3,
                     "Scale of bbox used when extracting meshes.")
  flags.DEFINE_float("level_set", 0.5,
                     "Level set used for extracting surfaces.")

  # Dataset flags
  flags.DEFINE_enum("dataset", "shapenet",
                    list(k for k in datasets.dataset_dict.keys()),
                    "Name of the dataset.")
  flags.DEFINE_integer("image_h", 137, "The height of the color images.")
  flags.DEFINE_integer("image_w", 137, "The width of the color images.")
  flags.DEFINE_integer("image_d", 3, "The channels of color images.")
  flags.DEFINE_integer("depth_h", 224, "The height of depth images.")
  flags.DEFINE_integer("depth_w", 224, "The width of depth images.")
  flags.DEFINE_integer("depth_d", 20, "The number of depth views.")
  flags.DEFINE_integer("n_views", 24, "The number of color images views.")
  flags.DEFINE_string("data_dir", None, "The base directory to load data from.")
  flags.mark_flag_as_required("data_dir")
  flags.DEFINE_string("obj_class", "*", "Object class used from dataset.")

  # Training flags
  flags.DEFINE_float("lr", 1e-4, "Start learning rate.")
  flags.DEFINE_string(
      "train_dir", None, "The base directory to save training info and"
      "checkpoints.")
  flags.DEFINE_integer("save_every", 20000,
                       "The number of steps to save checkpoint.")
  flags.DEFINE_integer("max_steps", 800000, "The number of steps of training.")
  flags.DEFINE_integer("batch_size", 32, "Batch size.")
  flags.DEFINE_integer("sample_bbx", 1024,
                       "The number of bounding box sample points.")
  flags.DEFINE_integer("sample_surf", 1024,
                       "The number of surface sample points.")
  flags.DEFINE_float("weight_overlap", 0.1, "Weight of overlap_loss")
  flags.DEFINE_float("weight_balance", 0.01, "Weight of balance_loss")
  flags.DEFINE_float("weight_center", 0.001, "Weight of center_loss")
  flags.mark_flag_as_required("train_dir")

  # Eval flags
  flags.DEFINE_bool("extract_mesh", False,
                    "Extract meshes and set to disk if True.")
  flags.DEFINE_bool("surface_metrics", False,
                    "Measure surface metrics and save to csv if True.")
  flags.DEFINE_string("mesh_dir", None, "Path to load ground truth meshes.")
  flags.DEFINE_string("trans_dir", None,
                      "Path to load pred-to-target transformations.")
  flags.DEFINE_bool("eval_once", False, "Evaluate the model only once if True.")


def mesh_name_helper(name):
  name = name[0].decode("utf-8")
  split = name.find("-")
  cls_name = name[:split]
  obj_name = name[split + 1:]
  return cls_name, obj_name


def extract_mesh(input_val, params, indicators, input_holder, params_holder,
                 points_holder, sess, args):
  """Extracting meshes from an indicator function.

  Args:
    input_val: np.array, [1, height, width, channel], input image.
    params: tf.Operation, hyperplane parameter hook.
    indicators: tf.Operation, indicator hook.
    input_holder: tf.Placeholder, input image placeholder.
    params_holder: tf.Placeholder, hyperplane parameter placeholder.
    points_holder: tf.Placeholder, query point placeholder.
    sess: tf.Session, running sess.
    args: tf.app.flags.FLAGS, configurations.

  Returns:
    mesh: trimesh.Trimesh, the extracted mesh.
  """
  mesh_extractor = mise.MISE(64, 1, args.level_set)
  points = mesh_extractor.query()
  params_val = sess.run(params, {input_holder: input_val})

  while points.shape[0] != 0:
    orig_points = points
    points = points.astype(np.float32)
    points = (
        (np.expand_dims(points, axis=0) / mesh_extractor.resolution - 0.5) *
        args.vis_scale)
    n_points = points.shape[1]
    values = []
    for i in range(0, n_points, 100000):  # Add this to prevent OOM.
      value = sess.run(indicators, {
          params_holder: params_val,
          points_holder: points[:, i:i + 100000]
      })
      values.append(value)
    values = np.concatenate(values, axis=1)
    values = values[0, :, 0].astype(np.float64)
    mesh_extractor.update(orig_points, values)
    points = mesh_extractor.query()

  value_grid = mesh_extractor.to_dense()
  value_grid = np.pad(value_grid, 1, "constant", constant_values=-1e6)
  verts, faces, normals, unused_var = measure.marching_cubes_lewiner(
      value_grid, min(args.level_set,
                      value_grid.max() * 0.75))
  del normals
  verts -= 1
  verts /= np.array([
      value_grid.shape[0] - 3, value_grid.shape[1] - 3, value_grid.shape[2] - 3
  ],
                    dtype=np.float32)
  verts = args.vis_scale * (verts - 0.5)
  faces = np.stack([faces[..., 1], faces[..., 0], faces[..., 2]], axis=-1)
  return trimesh.Trimesh(vertices=verts, faces=faces)


def transform_mesh(mesh, name, trans_dir):
  """Transform mesh back to the same coordinate of ground truth.

  Args:
    mesh: trimesh.Trimesh, predicted mesh before transformation.
    name: Tensor, hash name of the mesh as recorded in the dataset.
    trans_dir: string, path to the directory for loading transformations.

  Returns:
    mesh: trimesh.Trimesh, the transformed mesh.
  """
  if trans_dir is None:
    raise ValueError("Need to specify args.trans_dir for loading pred-to-target"
                     "transformations.")
  cls_name, obj_name = mesh_name_helper(name)
  with tf.io.gfile.GFile(
      path.join(trans_dir, "test", cls_name, obj_name, "occnet_to_gaps.txt"),
      "r") as fin:
    tx = np.loadtxt(fin).reshape([4, 4])
    mesh.apply_transform(np.linalg.inv(tx))
  return mesh


def save_mesh(mesh, name, eval_dir):
  """Save a mesh to disk.

  Args:
    mesh: trimesh.Trimesh, the mesh to save.
    name: Tensor, hash name of the mesh as recorded in the dataset.
    eval_dir: string, path to the directory to save the mesh.
  """
  cls_name, obj_name = mesh_name_helper(name)
  cls_dir = path.join(eval_dir, "meshes", cls_name)
  if not tf.io.gfile.isdir(cls_dir):
    tf.io.gfile.makedirs(cls_dir)
  with tf.io.gfile.GFile(path.join(cls_dir, obj_name + ".obj"), "w") as fout:
    mesh.export(fout, file_type="obj")


def distance_field_helper(source, target):
  target_kdtree = sp.spatial.cKDTree(target)
  distances, unused_var = target_kdtree.query(source, n_jobs=-1)
  return distances


def compute_surface_metrics(mesh, name, mesh_dir):
  """Compute surface metrics (chamfer distance and f-score) for one example.

  Args:
    mesh: trimesh.Trimesh, the mesh to evaluate.
    name: Tensor, hash name of the mesh as recorded in the dataset.
    mesh_dir: string, path to the directory for loading ground truth meshes.

  Returns:
    chamfer: float, chamfer distance.
    fscore: float, f-score.
  """
  if mesh_dir is None:
    raise ValueError("Need to specify args.mesh_dir for loading ground truth.")
  cls_name, obj_name = mesh_name_helper(name)
  with tf.io.gfile.GFile(
      path.join(mesh_dir, "test", cls_name, obj_name, "model_occnet.ply"),
      "rb",
  ) as fin:
    mesh_gt = trimesh.Trimesh(**trimesh.exchange.ply.load_ply(fin))

  # Chamfer
  eval_points = 100000
  point_gt = mesh_gt.sample(eval_points)
  point_gt = point_gt.astype(np.float32)
  point_pred = mesh.sample(eval_points)
  point_pred = point_pred.astype(np.float32)

  pred_to_gt = distance_field_helper(point_pred, point_gt)
  gt_to_pred = distance_field_helper(point_gt, point_pred)

  chamfer = np.mean(pred_to_gt**2) + np.mean(gt_to_pred**2)

  # Fscore
  tau = 1e-4
  eps = 1e-9

  pred_to_gt = (pred_to_gt**2)
  gt_to_pred = (gt_to_pred**2)

  prec_tau = (pred_to_gt <= tau).astype(np.float32).mean() * 100.
  recall_tau = (gt_to_pred <= tau).astype(np.float32).mean() * 100.

  fscore = (2 * prec_tau * recall_tau) / max(prec_tau + recall_tau, eps)

  # Following the tradition to scale chamfer distance up by 10.
  return chamfer * 100., fscore


def init_stats():
  """Initialize evaluation stats."""
  stats = {}
  for k in SYSNET_CLASSES:
    stats[k] = {
        "cnt": 0,
        "iou": 0.,
        "chamfer": 0.,
        "fscore": 0.,
    }
  return stats


def update_stats(example_stats, name, shapenet_stats):
  """Update evaluation statistics.

  Args:
    example_stats: Stats, the stats of one example.
    name: Tensor, hash name of the example as recorded in the dataset.
    shapenet_stats: dict, the current stats of the whole dataset.
  """
  cls_name, unused_var = mesh_name_helper(name)
  shapenet_stats[cls_name]["cnt"] += 1
  shapenet_stats[cls_name]["iou"] += example_stats.iou
  shapenet_stats[cls_name]["chamfer"] += example_stats.chamfer
  shapenet_stats[cls_name]["fscore"] += example_stats.fscore
  shapenet_stats["all"]["cnt"] += 1
  shapenet_stats["all"]["iou"] += example_stats.iou
  shapenet_stats["all"]["chamfer"] += example_stats.chamfer
  shapenet_stats["all"]["fscore"] += example_stats.fscore


def average_stats(shapenet_stats):
  """Average the accumulated stats of the whole dataset."""
  for k, v in shapenet_stats.items():
    cnt = max(v["cnt"], 1)
    shapenet_stats[k] = {
        "iou": v["iou"] / cnt,
        "chamfer": v["chamfer"] / cnt,
        "fscore": v["fscore"] / cnt,
    }


def write_stats(stats, eval_dir, step):
  """Write stats of the dataset to disk.

  Args:
    stats: dict, statistics to save.
    eval_dir: string, path to the directory to save the statistics.
    step: int, the global step of the checkpoint.
  """
  if not tf.io.gfile.isdir(eval_dir):
    tf.io.gfile.makedirs(eval_dir)
  with tf.io.gfile.GFile(path.join(eval_dir, "stats_{}.csv".format(step)),
                         "w") as fout:
    fout.write("class,iou,chamfer,fscore\n")
    for k in sorted(stats.keys()):
      if k == "all":
        continue
      fout.write("{0},{1},{2},{3}\n".format(
          SYSNET_CLASSES[k],
          stats[k]["iou"],
          stats[k]["chamfer"],
          stats[k]["fscore"],
      ))
    fout.write("all,{0},{1},{2}".format(
        stats["all"]["iou"],
        stats["all"]["chamfer"],
        stats["all"]["fscore"],
    ))
