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

import tensorflow as tf


def add_common_args(arg_parser):
  arg_parser.add_argument(
      "--debug",
      dest="debug",
      default=False,
      action="store_true",
      help="If set, debugging messages will be printed",
  )
  arg_parser.add_argument(
      "--quiet",
      "-q",
      dest="quiet",
      default=False,
      action="store_true",
      help="If set, only warnings will be printed",
  )
  arg_parser.add_argument(
      "--log",
      dest="logfile",
      default=None,
      help="If set, the log will be saved using the specified filename.",
  )


def configure_logging(args):
  logger = logging.getLogger()
  if args.debug:
    logger.setLevel(logging.DEBUG)
  elif args.quiet:
    logger.setLevel(logging.WARNING)
  else:
    logger.setLevel(logging.INFO)
  logger_handler = logging.StreamHandler()
  formatter = logging.Formatter("DeepSdf - %(levelname)s - %(message)s")
  logger_handler.setFormatter(formatter)
  logger.addHandler(logger_handler)

  if args.logfile is not None:
    file_logger_handler = logging.FileHandler(args.logfile)
    file_logger_handler.setFormatter(formatter)
    logger.addHandler(file_logger_handler)


def decode_sdf(decoder, latent_vector, queries):
  num_samples = queries.shape[0]

  if latent_vector is None:
    inputs = queries
  else:
    latent_repeat = tf.broadcast_to(latent_vector, [num_samples, -1])
    inputs = tf.concat([latent_repeat, queries], 1)

  sdf = decoder(inputs, training=False)

  return sdf
