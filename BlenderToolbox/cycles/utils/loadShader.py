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
import os
# pwd = os.getcwd()
pwd = os.path.dirname(os.path.realpath(__file__))

def loadShader(shaderName, mesh):
    # switch to different shader names
    if shaderName is "EeveeToon":
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.render.alpha_mode = 'TRANSPARENT'
        matName = "ToonShade_EV"
        blenderFileName = 'EeveeToon.blend'
    elif shaderName is "ColoredSteel":
        matName = "Blued_Steel"
        blenderFileName = 'ColoredSteel.blend'
    elif shaderName is "Wax":
        matName = "Wax_PBR_SSS"
        blenderFileName = 'Wax.blend'
    elif shaderName is "Wood":
        matName = "UCP wood-v-1-1"
        blenderFileName = 'UCPWood.blend' # createy by Elbriga

    # load shaders to the mesh
    path = pwd + '/../../shaders/' + blenderFileName + "\\Material\\"
    bpy.ops.wm.append(filename=matName, directory=path)
    mat = bpy.data.materials.get(matName)
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    tree = mat.node_tree
    matNode = tree.nodes[-1]
    return matNode