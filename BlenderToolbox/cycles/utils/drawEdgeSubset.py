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
import math
import numpy as np
from include import *

def drawEdgeSubset(mesh, E, r, edgeColor):
    bpy.ops.mesh.primitive_cylinder_add(radius = r, location = (1e10,1e10,1e10))
    cylinder = bpy.context.object

    mat = bpy.data.materials.new('MeshMaterial')
    cylinder.data.materials.append(mat)
    cylinder.active_material = mat
    mat.diffuse_color = edgeColor

    for ii in range(E.shape[0]): 
        p1Idx = E[ii,0]
        p2Idx = E[ii,1]
        p1 = mesh.matrix_world @ mesh.data.vertices[p1Idx].co
        p2 = mesh.matrix_world @ mesh.data.vertices[p2Idx].co
        x1 = p1[0]
        y1 = p1[1]
        z1 = p1[2]
        x2 = p2[0]
        y2 = p2[1]
        z2 = p2[2]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1    
        dist = math.sqrt(dx**2 + dy**2 + dz**2) 

        bpy.ops.object.duplicate({"object" : cylinder}, linked=True)
        objCopy = bpy.context.object
        objCopy.dimensions = (r,r,dist)
        objCopy.location = (dx/2 + x1, dy/2 + y1, dz/2 + z1)
        phi = math.atan2(dy, dx) 
        theta = math.acos(dz/dist) 
        bpy.context.object.rotation_euler[1] = theta 
        bpy.context.object.rotation_euler[2] = phi 