import sys
sys.path.append('/Users/hsuehtil/Dropbox/BlenderToolbox/cycles')
from include import *
import bpy

outputPath = './results/demo_edgeNormals.png'

# # init blender
imgRes_x = 720  # increase this for paper figures
imgRes_y = 720  # increase this for paper figures
numSamples = 50 # usually increase it to >200 for paper figures
exposure = 1.0
blenderInit(imgRes_x, imgRes_y, numSamples, exposure)

# read mesh 
meshPath = '../meshes/frog_paparazzi_l0.obj'
location = (-0.42, -0.28, 0.29)
rotation = (0, 0,90.1)
scale = (1,1,1)
mesh = readOBJ(meshPath, location, rotation, scale)

# # set shading
angle = 10
edgeNormals(mesh, angle)

# # subdivision
# level = 1
# subdivision(mesh, level)

# # set material
# colorObj(RGBA, H, S, V, Bright, Contrast)
meshColor = colorObj(derekBlue, 0.5, 1.0, 1.0, 0.0, 2.0)
AOStrength = 0.5
setMat_singleColor(mesh, meshColor, AOStrength)

# # set invisible plane (shadow catcher)
groundCenter = (0,0,0)
shadowDarkeness = 0.7
groundSize = 20
invisibleGround(groundCenter, groundSize, shadowDarkeness)

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