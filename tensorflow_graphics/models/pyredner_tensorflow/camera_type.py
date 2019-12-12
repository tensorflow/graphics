import redner

class CameraType:
    def __init__(self):
        self.perspective = redner.CameraType.perspective
        self.orthographic = redner.CameraType.orthographic
        self.fisheye = redner.CameraType.fisheye
        self.panorama = redner.CameraType.panorama

camera_type = CameraType()