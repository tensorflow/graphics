import bpy

def blenderInit(resolution_x, resolution_y, numSamples = 128, exposure = 1.5, useBothCPUGPU = False):
	# clear all
	bpy.ops.wm.read_homefile()
	bpy.ops.object.select_all(action = 'SELECT')
	bpy.ops.object.delete() 
	# use cycle
	bpy.context.scene.render.engine = 'CYCLES'
	bpy.context.scene.render.resolution_x = resolution_x 
	bpy.context.scene.render.resolution_y = resolution_y 
	# bpy.context.scene.cycles.film_transparent = True
	bpy.context.scene.render.film_transparent = True
	bpy.context.scene.cycles.samples = numSamples 
	bpy.context.scene.cycles.max_bounces = 6
	bpy.context.scene.cycles.film_exposure = exposure
	bpy.data.scenes[0].view_layers['View Layer']['cycles']['use_denoising'] = 1

	# set devices
	cyclePref  = bpy.context.preferences.addons['cycles'].preferences
	cyclePref.compute_device_type = 'CUDA'
	for dev in cyclePref.devices:
		if dev.type == "CPU" and useBothCPUGPU is False:
			dev.use = False
		else:
			dev.use = True
	bpy.context.scene.cycles.device = 'GPU'

	for dev in cyclePref.devices:
		print (dev)
		print (dev.use)

	return 0