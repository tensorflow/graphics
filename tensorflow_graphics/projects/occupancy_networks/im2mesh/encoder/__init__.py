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

from im2mesh.encoder import (
    conv,
    # pix2mesh_cond, pointnet,
    # psgn_cond, r2n2, voxels,
)


encoder_dict = {
    'simple_conv': conv.ConvEncoder,
    'resnet18': conv.Resnet18,
    # 'resnet34': conv.Resnet34,
    'resnet50': conv.Resnet50,
    'resnet101': conv.Resnet101,
    # 'r2n2_simple': r2n2.SimpleConv,
    # 'r2n2_resnet': r2n2.Resnet,
    # 'pointnet_simple': pointnet.SimplePointnet,
    # 'pointnet_resnet': pointnet.ResnetPointnet,
    # 'psgn_cond': psgn_cond.PCGN_Cond,
    # 'voxel_simple': voxels.VoxelEncoder,
    # 'pixel2mesh_cond': pix2mesh_cond.Pix2mesh_Cond,
}
