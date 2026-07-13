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
import os
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import trimesh
from im2mesh import config, data
from im2mesh.eval import MeshEvaluator
from im2mesh.utils.io import load_pointcloud


parser = argparse.ArgumentParser(
    description='Evaluate mesh algorithms.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--eval_input', action='store_true',
                    help='Evaluate inputs instead.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')

# Shorthands
out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
if not args.eval_input:
  out_file = os.path.join(generation_dir, 'eval_meshes_full.pkl')
  out_file_class = os.path.join(generation_dir, 'eval_meshes.csv')
else:
  out_file = os.path.join(generation_dir, 'eval_input_full.pkl')
  out_file_class = os.path.join(generation_dir, 'eval_input.csv')

# Dataset
points_field = data.PointsField(
    cfg['data']['points_iou_file'],
    unpackbits=cfg['data']['points_unpackbits'],
)
pointcloud_field = data.PointCloudField(
    cfg['data']['pointcloud_chamfer_file']
)
fields = {
    'points_iou': points_field,
    'pointcloud_chamfer': pointcloud_field,
    'idx': data.IndexField(),
}

print('Test split: ', cfg['data']['test_split'])


# Dataset
dataset_folder = cfg['data']['path']
dataset = data.Shapes3dDataset(
    dataset_folder, fields,
    cfg['data']['test_split'], batch_size=1,
    shuffle=False, repeat_count=1, epoch=1,
    categories=cfg['data']['classes'])

# Loader
dataloader = dataset.loader()

# Evaluator
evaluator = MeshEvaluator(n_points=100000)

# Evaluate all classes
eval_dicts = []
print('Evaluating meshes...')
for it, batch in enumerate(tqdm(dataloader)):
  if batch is None:
    print('Invalid data.')
    continue

  # Output folders
  if not args.eval_input:
    mesh_dir = os.path.join(generation_dir, 'meshes')
    pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
  else:
    mesh_dir = os.path.join(generation_dir, 'input')
    pointcloud_dir = os.path.join(generation_dir, 'input')

  # Get index etc.
  # idx = batch['idx']
  idx = it

  try:
    model_dict = dataset.get_model_dict(idx)
  except AttributeError:
    model_dict = {'model': str(idx), 'category': 'n/a'}

  modelname = model_dict['model']
  category_id = model_dict['category']

  try:
    category_name = dataset.metadata[category_id].get('name', 'n/a')
  except AttributeError:
    category_name = 'n/a'

  if category_id != 'n/a':
    mesh_dir = os.path.join(mesh_dir, category_id)
    pointcloud_dir = os.path.join(pointcloud_dir, category_id)

  # Evaluate
  pointcloud_tgt = tf.squeeze(batch['pointcloud_chamfer'], axis=0).numpy()
  normals_tgt = tf.squeeze(
      batch['pointcloud_chamfer.normals'], axis=0).numpy()
  points_tgt = tf.squeeze(batch['points_iou'], axis=0).numpy()
  occ_tgt = tf.squeeze(batch['points_iou.occ'], axis=0).numpy()

  # Evaluating mesh and pointcloud
  # Start row and put basic informatin inside
  eval_dict = {
      'idx': idx,
      'class id': category_id,
      'class name': category_name,
      'modelname': modelname,
  }
  eval_dicts.append(eval_dict)

  # Evaluate mesh
  if cfg['test']['eval_mesh']:
    mesh_file = os.path.join(mesh_dir, '%s.off' % modelname)

    if os.path.exists(mesh_file):
      mesh = trimesh.load(mesh_file, process=False)
      eval_dict_mesh = evaluator.eval_mesh(
          mesh, pointcloud_tgt, normals_tgt, points_tgt, occ_tgt)
      for k, v in eval_dict_mesh.items():
        eval_dict[k + ' (mesh)'] = v
    else:
      print('Warning: mesh does not exist: %s' % mesh_file)

  # Evaluate point cloud
  if cfg['test']['eval_pointcloud']:
    pointcloud_file = os.path.join(
        pointcloud_dir, '%s.ply' % modelname)

    if os.path.exists(pointcloud_file):
      pointcloud = load_pointcloud(pointcloud_file)
      eval_dict_pcl = evaluator.eval_pointcloud(
          pointcloud, pointcloud_tgt)
      for k, v in eval_dict_pcl.items():
        eval_dict[k + ' (pcl)'] = v
    else:
      print('Warning: pointcloud does not exist: %s'
            % pointcloud_file)

# Create pandas dataframe and save
eval_df = pd.DataFrame(eval_dicts)
eval_df.set_index(['idx'], inplace=True)
eval_df.to_pickle(out_file)

# Create CSV file  with main statistics
eval_df_class = eval_df.groupby(by=['class name']).mean()
eval_df_class.to_csv(out_file_class)

# Print results
eval_df_class.loc['mean'] = eval_df_class.mean()
print(eval_df_class)
