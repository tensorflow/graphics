import sys
sys.path.append('/Users/hsuehtil/Dropbox/BlenderToolbox/cycles')
from include import *
import bpy

outputPath = './results/demo_edgeSubset.png'

# # init blender
imgRes_x = 720  # increase this for paper figures
imgRes_y = 720  # increase this for paper figures
numSamples = 50 # usually increase it to >200 for paper figures
exposure = 1.0
blenderInit(imgRes_x, imgRes_y, numSamples, exposure)

# read mesh 
meshPath = '../meshes/spot.ply'
location = (-0.3, 0.6, -0.04)
rotation = (90, 0,0)
scale = (1.5,1.5,1.5)
mesh = readPLY(meshPath, location, rotation, scale)

# # set shading
# bpy.ops.object.shade_smooth()
bpy.ops.object.shade_flat()

# # subdivision
level = 0
subdivision(mesh, level)

# # set material (single color first)
# colorObj(RGBA, H, S, V, Bright, Contrast)
meshColor = colorObj((.5,.5,.5,1), 0.5, 1.0, 1.0, 0.0, 0.0)
AOStrength = 0.5
setMat_singleColor(mesh, meshColor, AOStrength)

# # draw edge subset
E = np.loadtxt('../meshes/Es.txt').astype(np.int32)
radius = 0.01
edgeColor = derekBlue
drawEdgeSubset(mesh, E,radius, edgeColor)

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