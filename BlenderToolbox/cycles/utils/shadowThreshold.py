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

def shadowThreshold(alphaThreshold, interpolationMode = 'CARDINAL'):
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    RAMP = tree.nodes.new('CompositorNodeValToRGB')
    RAMP.color_ramp.elements[0].color[3] = 0
    RAMP.color_ramp.elements[0].position = alphaThreshold
    RAMP.color_ramp.interpolation = interpolationMode

    REND = tree.nodes["Render Layers"]
    OUT = tree.nodes["Composite"]
    tree.links.new(REND.outputs[1], RAMP.inputs[0])
    tree.links.new(RAMP.outputs[1], OUT.inputs[1])