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

import sys
sys.path.append('/Users/hsuehtil/Dropbox/BlenderToolbox/cycles')
from include import *
import bpy

outputPath = './results/demo_threePtLights.png'

# # init blender
imgRes_x = 720  # increase this for paper figures
imgRes_y = 720  # increase this for paper figures
numSamples = 50 # usually increase it to >200 for paper figures
exposure = 1.5
blenderInit(imgRes_x, imgRes_y, numSamples, exposure)

# read mesh 
meshPath = '../meshes/spot.ply'
location = (-0.3, 0.6, -0.04)
rotation = (90, 0,0)
scale = (1.5,1.5,1.5)
mesh = readPLY(meshPath, location, rotation, scale)

# # set shading
bpy.ops.object.shade_smooth()
# bpy.ops.object.shade_flat()

# # subdivision
level = 2
subdivision(mesh, level)

# # set material
# colorObj(RGBA, H, S, V, Bright, Contrast)
meshColor = colorObj(derekBlue, 0.5, 1.0, 1.0, 0.0, 2.0)
AOStrength = 0.5
setMat_singleColor(mesh, meshColor, AOStrength)

# # set invisible plane (shadow catcher)
groundCenter = (0,0,0)
shadowBrightness = 0.8
groundSize = 20
invisibleGround(groundCenter, groundSize, shadowBrightness)

# # set camera
camLocation = (2,2,2)
lookAtLocation = (0,0,0.5)
focalLength = 45
cam = setCamera(camLocation, lookAtLocation, focalLength)

# set three lighting system
keyLoc = 'right'
setLight_threePoints(radius=4, height=10, intensity=1700, softness=6, keyLoc=keyLoc)

# # set ambient light
ambientColor = (0.1,0.1,0.1,1)
setLight_ambient(ambientColor)

## set gray shadow to completely white with a threshold (optional)
alphaThreshold = 0.05
shadowThreshold(alphaThreshold, interpolationMode = 'CARDINAL')

# # save blender file
bpy.ops.wm.save_mainfile(filepath='./test.blend')

# # save rendering
renderImage(outputPath, cam)