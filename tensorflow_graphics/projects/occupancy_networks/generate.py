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
""" NO COMMENT STILL"""

import os
import shutil
import argparse
from tqdm import tqdm
import time
from collections import defaultdict
import pandas as pd
import tensorflow as tf
from im2mesh import config
from im2mesh.checkpoints import CheckpointIO
from im2mesh.utils.io import export_pointcloud
from im2mesh.utils.visualize import visualize_data
from im2mesh.utils.voxels import VoxelGrid


parser = argparse.ArgumentParser(
    description="Extract meshes from occupancy process.")
parser.add_argument("config", type=str, help="Path to config file.")
parser.add_argument("--no-cuda", action="store_true", help="Do not use cuda.")

args = parser.parse_args()
cfg = config.load_config(args.config, "configs/default.yaml")

out_dir = cfg["training"]["out_dir"]
generation_dir = os.path.join(out_dir, cfg["generation"]["generation_dir"])
out_time_file = os.path.join(generation_dir, "time_generation_full.pkl")
out_time_file_class = os.path.join(generation_dir, "time_generation.pkl")

batch_size = cfg["generation"]["batch_size"]
input_type = cfg["data"]["input_type"]
vis_n_outputs = cfg["generation"]["vis_n_outputs"]
if vis_n_outputs is None:
  vis_n_outputs = -1


# Model
# Dataset
dataset = config.get_dataset(
    'test', cfg, batch_size=1, shuffle=False, repeat_count=1, epoch=1)
# Loader
dataloader = dataset.loader()

model = config.get_model(cfg, dataset=dataset)
dummy_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-08)

checkpoint_io = CheckpointIO(model, dummy_optimizer, checkpoint_dir=out_dir)

checkpoint_io.load(cfg['test']['model_file'])

# Generator
generator = config.get_generator(model, cfg)

# Determine what to generate
generate_mesh = cfg["generation"]["generate_mesh"]
generate_pointcloud = cfg["generation"]["generate_pointcloud"]

if generate_mesh and not hasattr(generator, "generate_mesh"):
  generate_mesh = False
  print("Warning: generator does not support mesh generation.")

if generate_pointcloud and not hasattr(generator, "generate_pointcloud"):
  generate_pointcloud = False
  print("Warning: generator does not support pointcloud generation.")


# Statistics
time_dicts = []


# Count how many models already created
model_counter = defaultdict(int)

for it, data in enumerate(tqdm(dataloader)):
  # Output folders
  mesh_dir = os.path.join(generation_dir, "meshes")
  pointcloud_dir = os.path.join(generation_dir, "pointcloud")
  in_dir = os.path.join(generation_dir, "input")
  generation_vis_dir = os.path.join(generation_dir, "vis",)

  # Get index etc.
  # idx = data["idx"].item()
  idx = it

  try:
    model_dict = dataset.get_model_dict(idx)
  except AttributeError:
    model_dict = {"model": str(idx), "category": "n/a"}

  modelname = model_dict["model"]
  category_id = model_dict.get("category", "n/a")

  try:
    category_name = dataset.metadata[category_id].get("name", "n/a")
  except AttributeError:
    category_name = "n/a"

  if category_id != "n/a":
    mesh_dir = os.path.join(mesh_dir, str(category_id))
    pointcloud_dir = os.path.join(pointcloud_dir, str(category_id))
    in_dir = os.path.join(in_dir, str(category_id))

    folder_name = str(category_id)
    if category_name != "n/a":
      folder_name = str(folder_name) + "_" + category_name.split(",")[0]

    generation_vis_dir = os.path.join(generation_vis_dir, folder_name)

  # Create directories if necessary
  if vis_n_outputs >= 0 and not os.path.exists(generation_vis_dir):
    os.makedirs(generation_vis_dir)

  if generate_mesh and not os.path.exists(mesh_dir):
    os.makedirs(mesh_dir)

  if generate_pointcloud and not os.path.exists(pointcloud_dir):
    os.makedirs(pointcloud_dir)

  if not os.path.exists(in_dir):
    os.makedirs(in_dir)

  # Timing dict
  time_dict = {
      "idx": idx,
      "class id": category_id,
      "class name": category_name,
      "modelname": modelname,
  }
  time_dicts.append(time_dict)

  # Generate outputs
  out_file_dict = {}

  # Also copy ground truth
  if cfg["generation"]["copy_groundtruth"]:
    modelpath = os.path.join(
        dataset.dataset_folder,
        category_id,
        modelname,
        cfg["data"]["watertight_file"],
    )
    out_file_dict["gt"] = modelpath

  if generate_mesh:
    t0 = time.time()
    out = generator.generate_mesh(data)
    time_dict["mesh"] = time.time() - t0

    # Get statistics
    try:
      mesh, stats_dict = out
    except TypeError:
      mesh, stats_dict = out, {}
    time_dict.update(stats_dict)

    # Write output
    mesh_out_file = os.path.join(mesh_dir, "%s.off" % modelname)
    mesh.export(mesh_out_file)
    out_file_dict["mesh"] = mesh_out_file

  if generate_pointcloud:
    t0 = time.time()
    pointcloud = generator.generate_pointcloud(data)
    time_dict["pcl"] = time.time() - t0
    pointcloud_out_file = os.path.join(
        pointcloud_dir, "%s.ply" % modelname)
    export_pointcloud(pointcloud, pointcloud_out_file)
    out_file_dict["pointcloud"] = pointcloud_out_file

  if cfg["generation"]["copy_input"]:
    # Save inputs
    if input_type == "img":
      inputs_path = os.path.join(in_dir, "%s.jpg" % modelname)
      inputs = tf.squeeze(data["inputs"], axis=0)
      visualize_data(inputs, "img", inputs_path)
      out_file_dict["in"] = inputs_path
    elif input_type == "voxels":
      inputs_path = os.path.join(in_dir, "%s.off" % modelname)
      inputs = tf.squeeze(data["inputs"], axis=0)
      voxel_mesh = VoxelGrid(inputs).to_mesh()
      voxel_mesh.export(inputs_path)
      out_file_dict["in"] = inputs_path
    elif input_type == "pointcloud":
      inputs_path = os.path.join(in_dir, "%s.ply" % modelname)
      inputs = tf.squeeze(data["inputs"], axis=0).numpy()
      export_pointcloud(inputs, inputs_path, False)
      out_file_dict["in"] = inputs_path

  # Copy to visualization directory for first vis_n_output samples
  c_it = model_counter[category_id]
  if c_it < vis_n_outputs:
    # Save output files
    img_name = "%02d.off" % c_it
    for k, filepath in out_file_dict.items():
      ext = os.path.splitext(filepath)[1]
      out_file = os.path.join(
          generation_vis_dir, "%02d_%s%s" % (c_it, k, ext))
      shutil.copyfile(filepath, out_file)

  model_counter[category_id] += 1

# Create pandas dataframe and save
time_df = pd.DataFrame(time_dicts)
time_df.set_index(["idx"], inplace=True)
time_df.to_pickle(out_time_file)

# Create pickle files  with main statistics
time_df_class = time_df.groupby(by=["class name"]).mean()
time_df_class.to_pickle(out_time_file_class)

# Print results
time_df_class.loc["mean"] = time_df_class.mean()
print("Timings [s]:")
print(time_df_class)
