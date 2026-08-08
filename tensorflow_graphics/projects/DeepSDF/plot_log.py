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


import logging
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import deep_sdf
import deep_sdf.workspace as ws


def running_mean(x, n):
  cumsum = np.cumsum(np.insert(x, 0, 0))
  return (cumsum[n:] - cumsum[:-n]) / float(n)


def load_logs(experiment_directory, logs_type):

  full_filename = os.path.join(experiment_directory, ws.logs_filename)
  with open(full_filename, 'rb') as f:
    logs = pickle.load(f)

  logging.info("latest epoch is %s", logs["epoch"])

  num_iters = len(logs["loss"])
  iters_per_epoch = num_iters / logs["epoch"]

  logging.info("%s iters per epoch", iters_per_epoch)

  smoothed_loss_41 = running_mean(logs["loss"], 41)
  smoothed_loss_1601 = running_mean(logs["loss"], 1601)

  _, ax = plt.subplots()

  if logs_type == "loss":

    ax.plot(
        np.arange(num_iters) / iters_per_epoch,
        logs["loss"],
        "#82c6eb",
        np.arange(20, num_iters - 20) / iters_per_epoch,
        smoothed_loss_41,
        "#2a9edd",
        np.arange(800, num_iters - 800) / iters_per_epoch,
        smoothed_loss_1601,
        "#16628b",
    )

    ax.set(xlabel="Epoch", ylabel="Loss", title="Training Loss")

  elif logs_type == "learning_rate":
    combined_lrs = np.array(logs["learning_rate"])

    ax.plot(
        np.arange(combined_lrs.shape[0]),
        combined_lrs[:, 0],
        np.arange(combined_lrs.shape[0]),
        combined_lrs[:, 1],
    )
    ax.set(xlabel="Epoch", ylabel="Learning Rate", title="Learning Rates")

  elif logs_type == "time":
    ax.plot(logs["timing"], "#833eb7")
    ax.set(xlabel="Epoch", ylabel="Time per Epoch (s)", title="Timing")

  elif logs_type == "lat_mag":
    ax.plot(logs["latent_magnitude"])
    ax.set(xlabel="Epoch", ylabel="Magnitude",
           title="Latent Vector Magnitude")

  elif logs_type == "param_mag":
    for _, mags in logs["param_magnitude"].items():
      ax.plot(mags)
    ax.set(xlabel="Epoch", ylabel="Magnitude", title="Parameter Magnitude")
    ax.legend(logs["param_magnitude"].keys())

  else:
    raise Exception('unrecognized plot type "{}"'.format(logs_type))

  ax.grid()
  plt.show()


if __name__ == "__main__":

  import argparse

  arg_parser = argparse.ArgumentParser(
      description="Plot DeepSDF training logs")
  arg_parser.add_argument(
      "--experiment",
      "-e",
      dest="experiment_directory",
      required=True,
      help="The experiment directory. This directory should include experiment "
      + "specifications in 'specs.json', \
        and logging will be done in this directory "
      + "as well",
  )
  arg_parser.add_argument("--type", "-t", dest="type", default="loss")

  deep_sdf.add_common_args(arg_parser)

  args = arg_parser.parse_args()

  deep_sdf.configure_logging(args)

  load_logs(args.experiment_directory, args.type)
