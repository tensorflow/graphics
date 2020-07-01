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
import mathutils
from utils.lookAt import *

def setCamera_orthographic(camLocation, lookAtLocation, top, bottom, left, right):
	# scale the resolution y using resolution x
  assert(abs(left-right)>0)
  assert(abs(top-bottom)>0)
  aspectRatio = abs(right - left)*1.0 / abs(top - bottom)
  bpy.context.scene.render.resolution_y = bpy.context.scene.render.resolution_x / aspectRatio

  bpy.ops.object.camera_add(location = camLocation)
  cam = bpy.context.object
  bpy.context.object.data.type = 'ORTHO'
  cam.data.ortho_scale = abs(left-right)
  loc = mathutils.Vector(lookAtLocation)
  lookAt(cam, loc)
  return cam