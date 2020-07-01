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

def initColorNode(tree, color, xloc = [200,400], yloc = [0,0]):
    HSV = tree.nodes.new('ShaderNodeHueSaturation')
    HSV.inputs['Color'].default_value = color.RGBA
    HSV.inputs['Saturation'].default_value = color.S
    HSV.inputs['Value'].default_value = color.V
    HSV.inputs['Hue'].default_value = color.H
    HSV.location.x -= xloc[0]
    HSV.location.y -= yloc[0]
    BS = tree.nodes.new('ShaderNodeBrightContrast')
    BS.inputs['Bright'].default_value = color.B
    BS.inputs['Contrast'].default_value = color.C
    BS.location.x -= xloc[1]
    BS.location.y -= yloc[1]
    tree.links.new(HSV.outputs['Color'], BS.inputs['Color'])
    return BS