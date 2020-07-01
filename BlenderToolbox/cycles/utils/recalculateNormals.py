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

def recalculateNormals(mesh):
    bpy.ops.object.select_all(action = 'DESELECT')
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.set_normals_from_faces()
    bpy.ops.object.editmode_toggle()