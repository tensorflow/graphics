import bpy 

def lookAt(camera, point):
	direction = point - camera.location
	rotQuat = direction.to_track_quat('-Z', 'Y')
	camera.rotation_euler = rotQuat.to_euler()