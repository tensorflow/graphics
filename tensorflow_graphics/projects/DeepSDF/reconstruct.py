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


import argparse
import json
import logging
import os
import random
import time

import numpy as np
import tensorflow as tf

import deep_sdf
import deep_sdf.workspace as ws


def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
):
  def adjust_learning_rate(
      initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
  ):
    lr = initial_lr * ((1 / decreased_by) **
                       (num_iterations // adjust_lr_every))
    for param_group in optimizer.param_groups:
      param_group["lr"] = lr

  decreased_by = 10
  adjust_lr_every = int(num_iterations / 2)

  if type(stat) == type(0.1):
    latent = tf.random.normal([1, latent_size], mean=0, std=stat)
  else:
    latent = tf.convert_to_tensor(np.random.normal(stat[0], stat[1]))

  optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

  loss_num = 0

  def loss_l1(y_pred, y_true):
    return tf.math.reduce_sum((y_pred - y_true))

  for e in range(num_iterations):

    sdf_data = deep_sdf.data.unpack_sdf_samples_from_ram(
        test_sdf, num_samples
    )
    xyz = sdf_data[:, 0:3]
    sdf_gt = tf.expand_dims(sdf_data[:, 3], axis=1)

    sdf_gt = tf.clip_by_value(sdf_gt, -clamp_dist, clamp_dist)

    adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

    with tf.GradientTape() as tape:
      latent_inputs = tf.broadcast_to(latent, [num_samples, -1])

      inputs = tf.concat([latent_inputs, xyz], axis=1)

      pred_sdf = decoder(inputs, training=False)

      # TODO: why is this needed?
      if e == 0:
        pred_sdf = decoder(inputs, training=False)

      pred_sdf = tf.clip_by_value(pred_sdf, -clamp_dist, clamp_dist)

      loss = loss_l1(pred_sdf, sdf_gt)
      if l2reg:
        loss += 1e-4 * tf.reduce_mean(tf.math.pow(latent, 2))

    grad_latent = tape.gradient(loss, latent)

    optimizer.apply_gradients(zip(grad_latent, latent))

    if e % 50 == 0:
      logging.debug(loss.numpy())
      logging.debug(e)
      logging.debug(tf.norm(latent))
    loss_num = loss.numpy()

  return loss_num, latent


if __name__ == "__main__":

  arg_parser = argparse.ArgumentParser(
      description="Use a trained DeepSDF decoder to \
        reconstruct a shape given SDF "
      + "samples."
  )
  arg_parser.add_argument(
      "--experiment",
      "-e",
      dest="experiment_directory",
      required=True,
      help="The experiment directory which includes \
        specifications and saved model "
      + "files to use for reconstruction",
  )
  arg_parser.add_argument(
      "--checkpoint",
      "-c",
      dest="checkpoint",
      default="latest",
      help="The checkpoint weights to use. \
        This can be a number indicated an epoch "
      + "or 'latest' for the latest weights (this is the default)",
  )
  arg_parser.add_argument(
      "--data",
      "-d",
      dest="data_source",
      required=True,
      help="The data source directory.",
  )
  arg_parser.add_argument(
      "--split",
      "-s",
      dest="split_filename",
      required=True,
      help="The split to reconstruct.",
  )
  arg_parser.add_argument(
      "--iters",
      dest="iterations",
      default=800,
      help="The number of iterations of latent code optimization to perform.",
  )
  arg_parser.add_argument(
      "--skip",
      dest="skip",
      action="store_true",
      help="Skip meshes which have already been reconstructed.",
  )
  deep_sdf.add_common_args(arg_parser)

  args = arg_parser.parse_args()

  deep_sdf.configure_logging(args)

  def empirical_stat(latent_vecs, indices):
    lat_mat = tf.zeros(0)
    for ind in indices:
      lat_mat = tf.concat([lat_mat, latent_vecs[ind]], axis=0)
    mean = tf.math.reduce_mean(lat_mat, axis=0)
    var = tf.math.square(tf.math.reduce_std(lat_mat, axis=0))
    return mean, var

  specs_filename = os.path.join(args.experiment_directory, "specs.json")

  if not os.path.isfile(specs_filename):
    raise Exception('The experiment directory does not\
      include specifications file "specs.json"'
                    )

  specs = json.load(open(specs_filename))

  arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

  latent_size = specs["CodeLength"]

  decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

  model_filename = os.path.join(
      args.experiment_directory, ws.model_params_subdir,
      args.checkpoint + ".ckpt"
  )
  ckpt = tf.train.Checkpoint(
      decoder=decoder, epoch=tf.Variable(0, dtype=tf.int64))
  ckpt.restore(model_filename)
  saved_model_epoch = ckpt.epoch

  with open(args.split_filename, "r") as f:
    split = json.load(f)

  npz_filenames = deep_sdf.data.get_instance_filenames(
      args.data_source, split)

  random.shuffle(npz_filenames)

  logging.debug(decoder)

  err_sum = 0.0
  repeat = 1
  save_latvec_only = False
  rerun = 0

  reconstruction_dir = os.path.join(
      args.experiment_directory, ws.reconstructions_subdir, str(
          saved_model_epoch)
  )

  if not os.path.isdir(reconstruction_dir):
    os.makedirs(reconstruction_dir)

  reconstruction_meshes_dir = os.path.join(
      reconstruction_dir, ws.reconstruction_meshes_subdir
  )
  if not os.path.isdir(reconstruction_meshes_dir):
    os.makedirs(reconstruction_meshes_dir)

  reconstruction_codes_dir = os.path.join(
      reconstruction_dir, ws.reconstruction_codes_subdir
  )
  if not os.path.isdir(reconstruction_codes_dir):
    os.makedirs(reconstruction_codes_dir)

  for ii, npz in enumerate(npz_filenames):

    if "npz" not in npz:
      continue

    full_filename = os.path.join(
        args.data_source, ws.sdf_samples_subdir, npz)

    logging.debug("loading %s", npz)

    data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename)

    for k in range(repeat):

      if rerun > 1:
        mesh_filename = os.path.join(
            reconstruction_meshes_dir, npz[:-4] + "-" + str(k + rerun)
        )
        latent_filename = os.path.join(
            reconstruction_codes_dir, npz[:-4] +
            "-" + str(k + rerun) + ".ckpt"
        )
      else:
        mesh_filename = os.path.join(
            reconstruction_meshes_dir, npz[:-4])
        latent_filename = os.path.join(
            reconstruction_codes_dir, npz[:-4] + ".ckpt"
        )

      if (
          args.skip
          and os.path.isfile(mesh_filename + ".ply")
          and os.path.isfile(latent_filename)
      ):
        continue

      logging.info("reconstructing %s", npz)

      data_sdf[0] = data_sdf[0][tf.random.shuffle(
          tf.range(start=0, limit=data_sdf[0].shape[0]))]
      data_sdf[1] = data_sdf[1][tf.random.shuffle(
          tf.range(start=0, limit=data_sdf[1].shape[0]))]

      start = time.time()
      err, latent = reconstruct(
          decoder,
          int(args.iterations),
          latent_size,
          data_sdf,
          0.01,  # [emp_mean,emp_var],
          0.1,
          num_samples=8000,
          lr=5e-3,
          l2reg=True,
      )
      logging.debug("reconstruct time: %s", time.time() - start)
      err_sum += err
      logging.debug("current_error avg: %s", (err_sum / (ii + 1)))
      logging.debug(ii)

      logging.debug("latent: %s", latent.numpy())

      if not os.path.exists(os.path.dirname(mesh_filename)):
        os.makedirs(os.path.dirname(mesh_filename))

      if not save_latvec_only:
        start = time.time()
        deep_sdf.mesh.create_mesh(
            decoder, latent, mesh_filename, N=256, max_batch=int(2 ** 18)
        )
        logging.debug("total time: %s", time.time() - start)

      if not os.path.exists(os.path.dirname(latent_filename)):
        os.makedirs(os.path.dirname(latent_filename))

      ckpt = tf.train.Checkpoint(latent_vec=tf.expand_dims(latent, axis=0))
      ckpt.save(latent_filename)
