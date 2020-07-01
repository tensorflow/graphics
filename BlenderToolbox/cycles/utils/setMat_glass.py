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

def setMat_glass(mesh, C1, roughness, transparancy = 0.5):
    mat = bpy.data.materials.new('MeshMaterial')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    # init color node
    C1 = initColorNode(tree, C1)

    # construct car paint node
    LW = tree.nodes.new('ShaderNodeLayerWeight')
    LW.inputs[0].default_value = transparancy
    LW.location.x -= 200
    LW.location.y -= 200
    GLO = tree.nodes.new('ShaderNodeBsdfGlossy')
    GLO.inputs[1].default_value = roughness
    GLO.location.x -= 400
    GLO.location.y -= 200
    TRAN = tree.nodes.new('ShaderNodeBsdfTransparent')
    tree.links.new(C1.outputs['Color'], TRAN.inputs['Color'])
    tree.links.new(C1.outputs['Color'], GLO.inputs['Color'])

    MIX = tree.nodes.new('ShaderNodeMixShader')
    tree.links.new(LW.outputs['Facing'], MIX.inputs['Fac'])
    tree.links.new(TRAN.outputs['BSDF'], MIX.inputs[1])
    tree.links.new(GLO.outputs['BSDF'], MIX.inputs[2])

    tree.links.new(MIX.outputs[0], tree.nodes['Material Output'].inputs['Surface'])