import sys
sys.path.append('/Users/hsuehtil/Dropbox/BlenderToolbox/cycles')

from include import *
import bpy

outputPath = './results/demo_readImage.png'

# # init blender
imgRes_x = 720  # increase this for paper figures
imgRes_y = 720  # increase this for paper figures
numSamples = 50 # usually increase it to >200 for paper figures
exposure = 1.0
blenderInit(imgRes_x, imgRes_y, numSamples, exposure)

# # read image
imagePath = '../meshes/fmap.png'
location = (0.06,0.12,0.86)
rotation = (90, 0, -15.4)
radius = 1.5
brightness = 2.0
img = readImagePlane(imagePath, location, rotation, radius, brightness)

# # set invisible plane (shadow catcher)
groundCenter = (0,0,0)
shadowDarkeness = 0.8
groundSize = 20
invisibleGround(groundCenter, groundSize, shadowDarkeness)

# # set camera
camLocation = (1.9,2,2.2)
lookAtLocation = (0,0,0.7)
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

# # # save rendering
renderImage(outputPath, cam)