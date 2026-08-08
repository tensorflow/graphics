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


import signal
import sys
import os
import logging
import math
import json
import time
import pickle

import tensorflow as tf

import deep_sdf
import deep_sdf.workspace as ws


class LearningRateSchedule:
  def get_learning_rate(self, epoch):
    pass


class ConstantLearningRateSchedule(LearningRateSchedule):
  def __init__(self, value):
    self.value = value

  def get_learning_rate(self, epoch):
    return self.value


class StepLearningRateSchedule(LearningRateSchedule):
  def __init__(self, initial, interval, factor):
    self.initial = initial
    self.interval = interval
    self.factor = factor

  def get_learning_rate(self, epoch):

    return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
  def __init__(self, initial, warmed_up, length):
    self.initial = initial
    self.warmed_up = warmed_up
    self.length = length

  def get_learning_rate(self, epoch):
    if epoch > self.length:
      return self.warmed_up
    return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

  schedule_specs = specs["LearningRateSchedule"]

  schedules = []

  for schedule_specs in schedule_specs:

    if schedule_specs["Type"] == "Step":
      schedules.append(
          StepLearningRateSchedule(
              schedule_specs["Initial"],
              schedule_specs["Interval"],
              schedule_specs["Factor"],
          )
      )
    elif schedule_specs["Type"] == "Warmup":
      schedules.append(
          WarmupLearningRateSchedule(
              schedule_specs["Initial"],
              schedule_specs["Final"],
              schedule_specs["Length"],
          )
      )
    elif schedule_specs["Type"] == "Constant":
      schedules.append(ConstantLearningRateSchedule(
          schedule_specs["Value"]))

    else:
      raise Exception(
          'no known learning rate schedule of type "{}"'.format(
              schedule_specs["Type"]
          )
      )

  return schedules


# need to rewrite
def save_model(experiment_directory, filename, decoder, epoch):

  model_params_dir = ws.get_model_params_dir(experiment_directory, True)
  ckpt = tf.train.Checkpoint(decoder=decoder, epoch=epoch)
  ckpt.save(os.path.join(model_params_dir, filename))

# need to rewrite


def save_optimizer(experiment_directory, filename,
                   optimizer_decoder, optimizer_lat_vecs, epoch):

  optimizer_params_dir = ws.get_optimizer_params_dir(
      experiment_directory, True)
  ckpt = tf.train.Checkpoint(
      optimizer_decoder=optimizer_decoder,
      optimizer_lat_vecs=optimizer_lat_vecs,
      epoch=epoch)
  ckpt.save(os.path.join(optimizer_params_dir, filename))


def load_optimizer(experiment_directory, filename,
                   optimizer_decoder, optimizer_lat_vecs):

  full_filename = os.path.join(
      ws.get_optimizer_params_dir(experiment_directory), filename
  )

  if not os.path.isfile(full_filename):
    raise Exception(
        'optimizer ckpt "{}" does not exist'.format(full_filename)
    )

  ckpt = tf.train.Checkpoint(
      optimizer_decoder=optimizer_decoder,
      optimizer_lat_vecs=optimizer_lat_vecs,
      epoch=tf.Variable(0, dtype=tf.int64))

  ckpt.restore(full_filename)
  epoch = ckpt.epoch

  return epoch

# need to rewrite


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

  latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)
  ckpt = tf.train.Checkpoint(latent_vec=latent_vec, epoch=epoch)
  ckpt.save(os.path.join(latent_codes_dir, filename))


# TODO:: duplicated in workspace
def load_latent_vectors(experiment_directory, filename, lat_vecs):

  full_filename = os.path.join(
      ws.get_latent_codes_dir(experiment_directory), filename
  )

  if not os.path.isfile(full_filename):
    raise Exception(
        'latent state file "{}" does not exist'.format(full_filename))

  ckpt = tf.train.Checkpoint(
      latent_vec=lat_vecs, epoch=tf.Variable(0, dtype=tf.int64))
  ckpt.restore(full_filename)
  epoch = ckpt.epoch

  return epoch


def save_logs(
    experiment_directory,
    loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch,
):

  logs = {
      "epoch": epoch,
      "loss": loss_log,
      "learning_rate": lr_log,
      "timing": timing_log,
      "latent_magnitude": lat_mag_log,
      "param_magnitude": param_mag_log,
  }
  with open(os.path.join(experiment_directory, ws.logs_filename), 'wb') as f:
    pickle.dump(logs, f)


def load_logs(experiment_directory):

  full_filename = os.path.join(experiment_directory, ws.logs_filename)

  if not os.path.isfile(full_filename):
    raise Exception('log file "{}" does not exist'.format(full_filename))

  with open(full_filename, 'rb') as f:
    data = pickle.load(f)

  return (
      data["loss"],
      data["learning_rate"],
      data["timing"],
      data["latent_magnitude"],
      data["param_magnitude"],
      data["epoch"],
  )


def clip_logs(loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):

  iters_per_epoch = len(loss_log) // len(lr_log)

  loss_log = loss_log[: (iters_per_epoch * epoch)]
  lr_log = lr_log[:epoch]
  timing_log = timing_log[:epoch]
  lat_mag_log = lat_mag_log[:epoch]
  for n in param_mag_log:
    param_mag_log[n] = param_mag_log[n][:epoch]

  return (loss_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def get_spec_with_default(specs, key, default):
  try:
    return specs[key]
  except KeyError:
    return default


def get_mean_latent_vector_magnitude(latent_vectors):
  return tf.reduce_mean(tf.norm(latent_vectors.weights, axis=1))


def append_parameter_magnitudes(param_mag_log, model):
  for name, param in model.named_parameters():
    if len(name) > 7 and name[:7] == "module.":
      name = name[7:]
    if name not in param_mag_log.keys():
      param_mag_log[name] = []
    param_mag_log[name].append(param.data.norm().item())


def main_function(experiment_directory, continue_from, batch_split):

  logging.debug("running %s", experiment_directory)

  specs = ws.load_experiment_specifications(experiment_directory)

  logging.info("Experiment description: \n%s", specs["Description"])

  data_source = specs["DataSource"]
  train_split_file = specs["TrainSplit"]

  arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

  logging.debug(specs["NetworkSpecs"])

  latent_size = specs["CodeLength"]

  checkpoints = list(
      range(
          specs["SnapshotFrequency"],
          specs["NumEpochs"] + 1,
          specs["SnapshotFrequency"],
      )
  )

  for checkpoint in specs["AdditionalSnapshots"]:
    checkpoints.append(checkpoint)
  checkpoints.sort()

  lr_schedules = get_learning_rate_schedules(specs)

  grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
  if grad_clip is not None:
    logging.debug("clipping gradients to max norm %f", grad_clip)

  def save_latest(epoch):

    save_model(experiment_directory, "latest.ckpt", decoder, epoch)
    save_optimizer(experiment_directory,
                   "latest.ckpt", optimizer_decoder, optimizer_lat_vecs, epoch)
    save_latent_vectors(experiment_directory,
                        "latest.ckpt", lat_vecs, epoch)

  def save_checkpoints(epoch):

    save_model(experiment_directory, str(epoch) + ".ckpt", decoder, epoch)
    save_optimizer(experiment_directory, str(
        epoch) + ".ckpt", optimizer_decoder, optimizer_lat_vecs, epoch)
    save_latent_vectors(experiment_directory, str(
        epoch) + ".ckpt", lat_vecs, epoch)

  def signal_handler(sig, frame):
    logging.info("Stopping early...")
    sys.exit(0)

  # TODO: optimizer.param_groups
  def adjust_learning_rate(lr_schedules, optimizer_decoder,
                           optimizer_lat_vecs, epoch):
    optimizer_decoder.learning_rate = lr_schedules[0].get_learning_rate(
        epoch)
    optimizer_lat_vecs.learning_rate = lr_schedules[1].get_learning_rate(
        epoch)

  signal.signal(signal.SIGINT, signal_handler)

  num_samp_per_scene = specs["SamplesPerScene"]
  scene_per_batch = specs["ScenesPerBatch"]
  clamp_dist = specs["ClampingDistance"]
  min_t = -clamp_dist
  max_t = clamp_dist
  enforce_minmax = True

  do_code_regularization = get_spec_with_default(
      specs, "CodeRegularization", True)
  code_reg_lambda = get_spec_with_default(
      specs, "CodeRegularizationLambda", 1e-4)

  code_bound = get_spec_with_default(specs, "CodeBound", None)

  decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

  logging.info("training with %d GPU(s)",
               len(tf.config.experimental.list_physical_devices('GPU')))

  num_epochs = specs["NumEpochs"]
  log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

  with open(train_split_file, "r") as f:
    train_split = json.load(f)

  # make dataset class
  sdf_samples = deep_sdf.data.SDFSamples(
      data_source, train_split, num_samp_per_scene,
      batch_size=scene_per_batch, shuffle=True,
      epoch=num_epochs, load_ram=False
  )
  sdf_dataset = sdf_samples.dataset()

  num_data_loader_threads = get_spec_with_default(
      specs, "DataLoaderThreads", 1)
  logging.debug("loading data with %d threads",
                num_data_loader_threads)

  # make dataset class
  num_scenes = len(sdf_samples.__len__())

  logging.info("There are %d scenes", num_scenes)

  logging.debug(decoder)

  num_embeddings = num_scenes
  embedding_dim = latent_size
  em_initializer = tf.keras.initializers.RandomNormal(
      0.0,
      get_spec_with_default(
          specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size))
  em_constraint = tf.keras.constraints.MaxNorm(code_bound)
  lat_vecs = tf.keras.layers.Embedding(
      num_embeddings, embedding_dim,
      embeddings_initializer=em_initializer,
      embeddings_constraint=em_constraint)

  logging.debug(
      "initialized with mean magnitude %f",
      get_mean_latent_vector_magnitude(lat_vecs)
  )

  # CHECK:
  def loss_l1(y_pred, y_true):
    return tf.math.reduce_sum((y_pred - y_true))

  optimizer_decoder = tf.keras.optimizers.Adam(
      learning_rate=lr_schedules[0].get_learning_rate(0))

  optimizer_lat_vecs = tf.keras.optimizers.Adam(
      learning_rate=lr_schedules[1].get_learning_rate(0))

  loss_log = []
  lr_log = []
  lat_mag_log = []
  timing_log = []
  param_mag_log = {}

  start_epoch = 1

  if continue_from is not None:

    logging.info('continuing from "%s"', continue_from)

    lat_epoch = load_latent_vectors(
        experiment_directory, continue_from +
        ".ckpt", lat_vecs
    )

    model_epoch = ws.load_model_parameters(
        experiment_directory, continue_from, decoder
    )

    optimizer_epoch = load_optimizer(
        experiment_directory, continue_from +
        ".ckpt", optimizer_decoder, optimizer_lat_vecs
    )

    loss_log, lr_log, timing_log, lat_mag_log, \
        param_mag_log, log_epoch = load_logs(
            experiment_directory)

    if not log_epoch == model_epoch:
      loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = clip_logs(
          loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, model_epoch
      )

    if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch):
      raise RuntimeError(
          "epoch mismatch: {} vs {} vs {} vs {}".format(
              model_epoch, optimizer_epoch, lat_epoch, log_epoch
          )
      )

    start_epoch = model_epoch + 1

    logging.debug("loaded")

  logging.info("starting from epoch %d", start_epoch)

  logging.info(
      "Number of decoder parameters: %d", decoder.count_params())

  logging.info(
      "Number of shape code parameters: %d (# codes %d, code dim %d)",
      num_embeddings * embedding_dim,
      num_embeddings,
      embedding_dim,
  )

  for epoch in range(start_epoch, num_epochs + 1):

    start = time.time()

    logging.info("epoch %d...", epoch)

    adjust_learning_rate(lr_schedules, optimizer_decoder,
                         optimizer_lat_vecs, epoch)

    for sdf_data, indices in sdf_dataset:

      # Process the input data
      sdf_data = tf.reshape(sdf_data, shape=[-1, 4])

      num_sdf_samples = sdf_data.shape[0]

      # sdf_data.requires_grad = False
      # sdf_data = tf.stop_gradient(sdf_data)

      xyz = sdf_data[:, 0:3]
      sdf_gt = tf.expand_dims(sdf_data[:, 3], axis=1)

      if enforce_minmax:
        sdf_gt = tf.clip_by_value(sdf_gt, min_t, max_t)

      xyz = tf.split(xyz, batch_split)
      indices = tf.split(
          tf.reshape(
              tf.tile(
                  tf.expand_dims(
                      indices, axis=-1),
                  multiples=[1, num_samp_per_scene]),
              shape=[-1]),
          batch_split)

      sdf_gt = tf.split(sdf_gt, batch_split)

      batch_loss = 0.0
      grad_decoder = 0.0
      grad_lat_vecs = 0.0

      for i in range(batch_split):
        with tf.GradientTape() as tape:

          batch_vecs = lat_vecs(indices[i])

          inp = tf.concat([batch_vecs, xyz[i]], axis=1)

          # NN optimization
          pred_sdf = decoder(inp, training=True)

          if enforce_minmax:
            pred_sdf = tf.clip_by_value(pred_sdf, min_t, max_t)

          chunk_loss = loss_l1(
              pred_sdf, sdf_gt[i]) / num_sdf_samples

          if do_code_regularization:
            l2_size_loss = tf.reduce_sum(
                tf.norm(batch_vecs, axis=1))
            reg_loss = (
                code_reg_lambda * min(1,
                                      epoch / 100) * l2_size_loss
            ) / num_sdf_samples

            chunk_loss = chunk_loss + reg_loss

        batch_loss += chunk_loss[0]

        grad_decoder += tape.gradient(
            chunk_loss, decoder.trainable_variables)

        grad_lat_vecs += tape.gradient(
            chunk_loss, lat_vecs.trainable_weights)

      logging.debug("loss = %f", batch_loss)

      loss_log.append(batch_loss)

      if grad_clip is not None:

        grad_decoder = tf.clip_by_norm(grad_decoder, grad_clip)

      optimizer_decoder.apply_gradients(
          zip(grad_decoder, decoder.trainable_variables))
      optimizer_lat_vecs.apply_gradients(
          zip(grad_lat_vecs, lat_vecs.trainable_weights))

    end = time.time()

    seconds_elapsed = end - start
    timing_log.append(seconds_elapsed)

    lr_log.append([schedule.get_learning_rate(epoch)
                   for schedule in lr_schedules])

    lat_mag_log.append(get_mean_latent_vector_magnitude(lat_vecs))

    append_parameter_magnitudes(param_mag_log, decoder)

    if epoch in checkpoints:
      save_checkpoints(epoch)

    if epoch % log_frequency == 0:

      save_latest(epoch)
      save_logs(
          experiment_directory,
          loss_log,
          lr_log,
          timing_log,
          lat_mag_log,
          param_mag_log,
          epoch,
      )


if __name__ == "__main__":

  import argparse

  arg_parser = argparse.ArgumentParser(
      description="Train a DeepSDF autodecoder")
  arg_parser.add_argument(
      "--experiment",
      "-e",
      dest="experiment_directory",
      required=True,
      help="The experiment directory. This directory should include "
      + "experiment specifications in 'specs.json', and logging will be "
      + "done in this directory as well.",
  )
  arg_parser.add_argument(
      "--continue",
      "-c",
      dest="continue_from",
      help="A snapshot to continue from. This can be 'latest' to continue"
      + "from the latest running snapshot, or an integer corresponding to "
      + "an epochal snapshot.",
  )
  arg_parser.add_argument(
      "--batch_split",
      dest="batch_split",
      default=1,
      help="This splits the batch into separate subbatches which are "
      + "processed separately, with gradients accumulated across all "
      + "subbatches. This allows for training with large effective batch "
      + "sizes in memory constrained environments.",
  )

  deep_sdf.add_common_args(arg_parser)

  args = arg_parser.parse_args()

  deep_sdf.configure_logging(args)

  main_function(args.experiment_directory,
                args.continue_from, int(args.batch_split))
