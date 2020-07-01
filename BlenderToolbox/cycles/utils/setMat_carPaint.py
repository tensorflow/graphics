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

def setMat_carPaint(mesh, C1, C2):
    mat = bpy.data.materials.new('MeshMaterial')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    # init color node
    C1_BS = initColorNode(tree, C1)
    C2_BS = initColorNode(tree, C2, [200, 400], [200, 200])

    # construct car paint node
    LW = tree.nodes.new('ShaderNodeLayerWeight')
    CR = tree.nodes.new('ShaderNodeValToRGB')
    tree.links.new(LW.outputs['Facing'], CR.inputs['Fac'])

    MIX = tree.nodes.new('ShaderNodeMixRGB')
    tree.links.new(CR.outputs['Color'], MIX.inputs['Fac'])
    tree.links.new(C1_BS.outputs['Color'], MIX.inputs['Color1'])
    tree.links.new(C2_BS.outputs['Color'], MIX.inputs['Color2'])

    GLO = tree.nodes.new('ShaderNodeBsdfGlossy')
    GLO.inputs['Roughness'].default_value = 0.224
    DIF = tree.nodes.new('ShaderNodeBsdfDiffuse')
    tree.links.new(MIX.outputs['Color'], DIF.inputs['Color'])

    MIXS = tree.nodes.new('ShaderNodeMixShader')
    MIXS.inputs['Fac'].default_value = 0.25
    tree.links.new(DIF.outputs['BSDF'], MIXS.inputs[1])
    tree.links.new(GLO.outputs['BSDF'], MIXS.inputs[2])

    tree.links.new(MIXS.outputs[0], tree.nodes['Material Output'].inputs['Surface'])