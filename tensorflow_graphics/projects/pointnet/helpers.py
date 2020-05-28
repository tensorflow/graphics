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
"""A collection of training helper utilities."""

from __future__ import print_function

import argparse
import os
import tempfile
import time

import tensorflow as tf
import termcolor


class ArgumentParser(argparse.ArgumentParser):
  """Argument parser with default flags, and tensorboard helpers."""

  def __init__(self, *args, **kwargs):
    argparse.ArgumentParser.__init__(self, *args, **kwargs)

    # --- Query default logdir
    random_logdir = tempfile.mkdtemp(prefix="tensorboard_")
    default_logdir = os.environ.get("TENSORBOARD_DEFAULT_LOGDIR", random_logdir)

    # --- Add the default options
    self.add("--logdir", default_logdir, help="tensorboard dir")
    self.add("--tensorboard", True, help="should generate summaries?")
    self.add("--assert_gpu", True, help="asserts on missing GPU accelerator")
    self.add("--tf_quiet", True, help="no verbose tf startup")

  def add(self, name, default, **kwargs):
    """More compact argumentparser 'add' flag method."""
    helpstring = kwargs["help"] if "help" in kwargs else ""
    metavar = kwargs["metavar"] if "metavar" in kwargs else name

    # --- Fixes problems with bool arguments
    def str2bool(string):
      if isinstance(string, bool):
        return str
      if string.lower() in ("true", "yes"):
        return True
      if string.lower() in ("false", "no"):
        return False
      raise argparse.ArgumentTypeError("Bad value for boolean flag")

    mytype = type(default)
    if isinstance(default, bool):
      mytype = str2bool

    self.add_argument(
        name, metavar=metavar, default=default, help=helpstring, type=mytype)

  def parse_args(self, args=None, namespace=None):
    """WARNING: programmatically changes the logdir flags."""
    flags = super(ArgumentParser, self).parse_args(args)

    # --- setup automatic logdir (timestamp)
    if "timestamp" in flags.logdir:
      timestamp = time.strftime("%a%d_%H:%M:%S")  # "Tue19_12:02:26"
      flags.logdir = flags.logdir.replace("timestamp", timestamp)

    if flags.tf_quiet:
      set_tensorflow_log_level(3)

    if flags.assert_gpu:
      assert_gpu_available()

    # --- ensure logdir ends in /
    if flags.logdir[-1] != "/":
      flags.logdir += "/"

    return flags


def assert_gpu_available():
  """Verifies a GPU accelerator is available."""
  physical_devices = tf.config.list_physical_devices("GPU")
  num_gpus = len(physical_devices)
  assert num_gpus >= 1, "execution requires one GPU"


def set_tensorflow_log_level(level=3):
  """Sets the log level of TensorFlow."""
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(level)


def summary_command(parser, flags, log_to_file=True, log_to_summary=True):
  """Cache the command used to reproduce experiment in summary folder."""
  if not flags.tensorboard:
    return
  exec_string = "python " + parser.prog + " \\\n"
  nflags = len(vars(flags))
  for i, arg in enumerate(vars(flags)):
    exec_string += "  --{} ".format(arg)
    exec_string += "{}".format(getattr(flags, arg))
    if i + 1 < nflags:
      exec_string += " \\\n"
  exec_string += "\n"
  if log_to_file:
    with tf.io.gfile.GFile(
        os.path.join(flags.logdir, "command.txt"), mode="w") as fid:
      fid.write(exec_string)
  if log_to_summary and flags.tensorboard:
    tf.summary.text("command", exec_string, step=0)


def setup_tensorboard(flags):
  """Creates summary writers, and setups default tensorboard paths."""
  if not flags.tensorboard:
    return

  # --- Do not allow experiment with same name
  assert (not tf.io.gfile.exists(flags.logdir) or
          not tf.io.gfile.listdir(flags.logdir)), \
    "CRITICAL: folder {} already exists".format(flags.logdir)

  # --- Log where summary can be found
  print("View results with: ")
  termcolor.cprint("  tensorboard --logdir {}".format(flags.logdir), "red")
  writer = tf.summary.create_file_writer(flags.logdir, flush_millis=10000)
  writer.set_as_default()

  # --- Log dir name tweak for "hypertune"
  log_dir = ""
  trial_id = int(os.environ.get("CLOUD_ML_TRIAL_ID", 0))
  if trial_id != 0:
    if log_dir.endswith(os.sep):
      log_dir = log_dir[:-1]  # removes trailing "/"
    log_dir += "_trial{0:03d}/".format(trial_id)


def handle_keyboard_interrupt(flags):
  """Informs user how to delete stale summaries."""
  print("Keyboard interrupt by user")
  if flags.logdir.startswith("gs://"):
    bucketpath = flags.logdir[5:]
    print("Delete these summaries with: ")
    termcolor.cprint("  gsutil rm -rf {}".format(flags.logdir), "red")
    baseurl = "  https://pantheon.google.com/storage/browser/{}"
    print("Or by visiting: ")
    termcolor.cprint(baseurl.format(bucketpath), "red")
  else:
    print("Delete these summaries with: ")
    termcolor.cprint("  rm -rf {}".format(flags.logdir), "red")
