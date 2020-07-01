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

def setMat_chrome(mesh, roughness):
    mat = bpy.data.materials.new('MeshMaterial')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    # construct car paint node
    LW = tree.nodes.new('ShaderNodeLayerWeight')
    LW.inputs[0].default_value = 0.7
    CR = tree.nodes.new('ShaderNodeValToRGB')
    CR.color_ramp.elements[0].position = 0.9
    CR.color_ramp.elements[1].position = 1
    CR.color_ramp.elements[0].color = (1,1,1,1)
    CR.color_ramp.elements[1].color = (0,0,0,1)
    GLO = tree.nodes.new('ShaderNodeBsdfGlossy')
    GLO.inputs[1].default_value = roughness
    GLO.location.x -= 200
    
    tree.links.new(LW.outputs[1], CR.inputs['Fac'])
    tree.links.new(CR.outputs['Color'], GLO.inputs['Color'])
    tree.links.new(GLO.outputs[0], tree.nodes['Material Output'].inputs['Surface'])
