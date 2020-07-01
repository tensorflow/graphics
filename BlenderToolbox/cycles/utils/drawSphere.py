# Copyright 2020 The TensorFlow Authors, Derek Liu
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
import bpy
from include import *

def drawSphere(ptSize, ptColor, ptLoc = (1e10,1e10,1e10)):
    bpy.ops.mesh.primitive_uv_sphere_add(radius = ptSize)
    sphere = bpy.context.object
    sphere.location = ptLoc
    bpy.ops.object.shade_smooth()

    mat = bpy.data.materials.new('sphere_mat')
    sphere.data.materials.append(mat)
    mat.use_nodes = True
    tree = mat.node_tree

    BCNode = initColorNode(tree, ptColor)

    tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 1.0
    tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = 0
    tree.links.new(BCNode.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])
    return sphere