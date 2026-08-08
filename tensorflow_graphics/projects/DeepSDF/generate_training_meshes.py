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

import os
import numpy as np
import tensorflow as tf

import deep_sdf
import deep_sdf.workspace as ws


def code_to_mesh(experiment_directory, checkpoint, keep_normalized=False):

  specs_filename = os.path.join(experiment_directory, "specs.json")

  if not os.path.isfile(specs_filename):
    raise Exception(
        'The experiment directory does not include\
           specifications file "specs.json"'
    )

  specs = json.load(open(specs_filename))

  arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

  latent_size = specs["CodeLength"]

  decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

  model_filename = os.path.join(experiment_directory,
                                ws.model_params_subdir, checkpoint + ".ckpt")
  ckpt = tf.train.Checkpoint(
      decoder=decoder, epoch=tf.Variable(0, dtype=tf.int64))
  ckpt.restore(model_filename)
  saved_model_epoch = ckpt.epoch

  latent_vectors = ws.load_latent_vectors(experiment_directory, checkpoint)

  train_split_file = specs["TrainSplit"]

  with open(train_split_file, "r") as f:
    train_split = json.load(f)

  data_source = specs["DataSource"]

  instance_filenames = deep_sdf.data.get_instance_filenames(
      data_source, train_split)

  print(len(instance_filenames), " vs ", len(latent_vectors))

  for i, latent_vector in enumerate(latent_vectors):

    dataset_name, class_name, instance_name = instance_filenames[i].split(
        "/")
    instance_name = instance_name.split(".")[0]

    print("{} {} {}".format(dataset_name, class_name, instance_name))

    mesh_dir = os.path.join(
        experiment_directory,
        ws.training_meshes_subdir,
        str(saved_model_epoch),
        dataset_name,
        class_name,
    )
    print(mesh_dir)

    if not os.path.isdir(mesh_dir):
      os.makedirs(mesh_dir)

    mesh_filename = os.path.join(mesh_dir, instance_name)

    print(instance_filenames[i])

    offset = None
    scale = None

    if not keep_normalized:

      normalization_params = np.load(
          ws.get_normalization_params_filename(
              data_source, dataset_name, class_name, instance_name
          )
      )
      offset = normalization_params["offset"]
      scale = normalization_params["scale"]

    deep_sdf.mesh.create_mesh(
        decoder,
        latent_vector,
        mesh_filename,
        N=256,
        max_batch=int(2 ** 18),
        offset=offset,
        scale=scale,
    )


if __name__ == "__main__":

  arg_parser = argparse.ArgumentParser(
      description="Use a trained DeepSDF decoder to\
         generate a mesh given a latent code."
  )
  arg_parser.add_argument(
      "--experiment",
      "-e",
      dest="experiment_directory",
      required=True,
      help="The experiment directory which includes\
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
      "--keep_normalization",
      dest="keep_normalized",
      default=False,
      action="store_true",
      help="If set, keep the meshes in the normalized scale.",
  )
  deep_sdf.add_common_args(arg_parser)

  args = arg_parser.parse_args()

  deep_sdf.configure_logging(args)

  code_to_mesh(args.experiment_directory,
               args.checkpoint, args.keep_normalized)
