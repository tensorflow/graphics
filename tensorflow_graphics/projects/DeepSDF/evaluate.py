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


import os
import argparse
import logging
import json
import numpy as np
import trimesh

import deep_sdf
import deep_sdf.workspace as ws


def evaluate(experiment_directory, checkpoint, data_dir, split_filename):

  with open(split_filename, "r") as f:
    split = json.load(f)

  chamfer_results = []

  for dataset in split:
    for class_name in split[dataset]:
      for instance_name in split[dataset][class_name]:
        logging.debug(
            "evaluating %s", os.path.join(dataset, class_name, instance_name)
        )

        reconstructed_mesh_filename = ws.get_reconstructed_mesh_filename(
            experiment_directory, checkpoint, dataset, class_name, instance_name
        )

        logging.debug(
            'reconstructed mesh is "%s', reconstructed_mesh_filename
        )

        ground_truth_samples_filename = os.path.join(
            data_dir,
            "SurfaceSamples",
            dataset,
            class_name,
            instance_name + ".ply",
        )

        logging.debug(
            "ground truth samples are %s", ground_truth_samples_filename
        )

        normalization_params_filename = os.path.join(
            data_dir,
            "NormalizationParameters",
            dataset,
            class_name,
            instance_name + ".npz",
        )

        logging.debug(
            "normalization params are %s", ground_truth_samples_filename
        )

        ground_truth_points = trimesh.load(ground_truth_samples_filename)
        reconstruction = trimesh.load(reconstructed_mesh_filename)

        normalization_params = np.load(normalization_params_filename)

        chamfer_dist = deep_sdf.metrics.chamfer.compute_trimesh_chamfer(
            ground_truth_points,
            reconstruction,
            normalization_params["offset"],
            normalization_params["scale"],
        )

        logging.debug("chamfer distance: %s", str(chamfer_dist))

        chamfer_results.append(
            (os.path.join(dataset, class_name, instance_name), chamfer_dist)
        )

  with open(
      os.path.join(
          ws.get_evaluation_dir(experiment_directory,
                                checkpoint, True), "chamfer.csv"
      ),
      "w",
  ) as f:
    f.write("shape, chamfer_dist\n")
    for result in chamfer_results:
      f.write("{}, {}\n".format(result[0], result[1]))


if __name__ == "__main__":

  arg_parser = argparse.ArgumentParser(
      description="Evaluate a DeepSDF autodecoder")
  arg_parser.add_argument(
      "--experiment",
      "-e",
      dest="experiment_directory",
      required=True,
      help="The experiment directory. This directory should include \
        experiment specifications in "
      + '"specs.json", and logging will be done in this directory as well.',
  )
  arg_parser.add_argument(
      "--checkpoint",
      "-c",
      dest="checkpoint",
      default="latest",
      help="The checkpoint to test.",
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
      help="The split to evaluate.",
  )

  deep_sdf.add_common_args(arg_parser)

  args = arg_parser.parse_args()

  deep_sdf.configure_logging(args)

  evaluate(
      args.experiment_directory,
      args.checkpoint,
      args.data_source,
      args.split_filename,
  )
