import sys
sys.path.append('/Users/hsuehtil/Dropbox/BlenderToolbox/cycles')
from include import *
import bpy

outputPath = './results/demo_drawLines.png'

# # init blender
imgRes_x = 720  # increase this for paper figures
imgRes_y = 720  # increase this for paper figures
numSamples = 50 # usually increase it to >200 for paper figures
exposure = 1.0
blenderInit(imgRes_x, imgRes_y, numSamples, exposure)

# create original local frame
orig = np.array([0.,0.,0.])
lineVec = np.array([[1,0,0],[0,1,0],[0,0,1]])

p1List = np.tile(orig[None,:], (lineVec.shape[0],1))
p2List = p1List + lineVec
colorList = np.array([[1,0,0,1],[0,1,0,1],[0,0,1,1]])
radius = 0.1
drawLines(p1List, p2List, radius, colorList)

# # set camera
camLocation = (1.9,2,2.2)
lookAtLocation = (0,0,0.5)
focalLength = 45
cam = setCamera(camLocation, lookAtLocation, focalLength)

# # set sunlight
lightAngle = (-15,-34,-155) 
strength = 2
shadowSoftness = 0.1
sun = setLight_sun(lightAngle, strength, shadowSoftness)

# # set ambient light
ambientColor = (0.2,0.2,0.2,1)
setLight_ambient(ambientColor)

# # save blender file
bpy.ops.wm.save_mainfile(filepath='./test.blend')

# # save rendering
renderImage(outputPath, cam)