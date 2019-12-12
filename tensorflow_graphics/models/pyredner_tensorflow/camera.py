import numpy as np
import tensorflow as tf
import pyredner_tensorflow.transform as transform
import redner
import pyredner_tensorflow as pyredner
import math
from typing import Tuple, Optional, List

class Camera:
    """
        redner supports four types of cameras: perspective, orthographic, fisheye, and panorama.
        The camera takes a look at transform or a cam_to_world matrix to
        transform from camera local space to world space. It also can optionally
        take an intrinsic matrix that models field of view and camera skew.

        Args
        ----------
            position: Optional[tf.Tensor]
                the origin of the camera, 1-d tensor with size 3 and type float32
            look_at: Optional[tf.Tensor]
                the point camera is looking at, 1-d tensor with size 3 and type float32
            up: Optional[tf.tensor]
                the up vector of the camera, 1-d tensor with size 3 and type float32
            fov: Optional[tf.Tensor]
                the field of view of the camera in angle
                no effect if the camera is a fisheye or panorama camera
                1-d tensor with size 1 and type float32
            clip_near: float
                the near clipping plane of the camera, need to > 0
            resolution: Tuple[int, int]
                the size of the output image in (height, width)
            cam_to_world: Optional[tf.Tensor]
                overrides position, look_at, up vectors
                4x4 matrix, optional
            intrinsic_mat: Optional[tf.Tensor]
                a matrix that transforms a point in camera space before the point
                is projected to 2D screen space
                used for modelling field of view and camera skewing
                after the multiplication the point should be in
                [-1, 1/aspect_ratio] x [1, -1/aspect_ratio] in homogeneous coordinates
                the projection is then carried by the specific camera types
                perspective camera normalizes the homogeneous coordinates
                while orthogonal camera drop the Z coordinate.
                ignored by fisheye or panorama cameras
                overrides fov
                3x3 matrix, optional
            camera_type: render.camera_type
                the type of the camera (perspective, orthographic, or fisheye)
            fisheye: bool
                whether the camera is a fisheye camera
                (legacy parameter just to ensure compatibility).

    """
    def __init__(self,
                 position: Optional[tf.Tensor] = None,
                 look_at: Optional[tf.Tensor] = None,
                 up: Optional[tf.Tensor] = None,
                 fov: Optional[tf.Tensor] = None,
                 clip_near: float = 1e-4,
                 resolution: Tuple[int] = (256, 256),
                 cam_to_world: Optional[tf.Tensor] = None,
                 intrinsic_mat: Optional[tf.Tensor] = None,
                 camera_type = pyredner.camera_type.perspective,
                 fisheye: bool = False):
        assert(tf.executing_eagerly())
        if position is not None:
            assert(position.dtype == tf.float32)
            assert(len(position.shape) == 1 and position.shape[0] == 3)
        if look_at is not None:
            assert(look_at.dtype == tf.float32)
            assert(len(look_at.shape) == 1 and look_at.shape[0] == 3)
        if up is not None:
            assert(up.dtype == tf.float32)
            assert(len(up.shape) == 1 and up.shape[0] == 3)
        if fov is not None:
            assert(fov.dtype == tf.float32)
            assert(len(fov.shape) == 1 and fov.shape[0] == 1)
        assert(isinstance(clip_near, float))
        if position is None and look_at is None and up is None:
            assert(cam_to_world is  not None)
        
        self.position = position
        self.look_at = look_at
        self.up = up
        self.fov = fov
        with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
            if cam_to_world is not None:
                self.cam_to_world = cam_to_world
            else:
                self.cam_to_world = None
            if intrinsic_mat is None:
                if camera_type == redner.CameraType.perspective:
                    fov_factor = 1.0 / tf.tan(transform.radians(0.5 * fov))
                    o = tf.ones([1], dtype=tf.float32)
                    diag = tf.concat([fov_factor, fov_factor, o], 0)
                    self._intrinsic_mat = tf.linalg.tensor_diag(diag)
                else:
                    self._intrinsic_mat = tf.eye(3, dtype=tf.float32)   
            else:
                self._intrinsic_mat = intrinsic_mat
            self.intrinsic_mat_inv = tf.linalg.inv(self._intrinsic_mat)
        self.clip_near = clip_near
        self.resolution = resolution
        self.camera_type = camera_type
        if fisheye:
            self.camera_type = redner.CameraType.fisheye

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, value):
        if value is not None:
            self._fov = value
            with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
                fov_factor = 1.0 / tf.tan(transform.radians(0.5 * self._fov))
                o = tf.ones([1], dtype=tf.float32)
                diag = tf.concat([fov_factor, fov_factor, o], 0)
                self._intrinsic_mat = tf.linalg.tensor_diag(diag)
                self.intrinsic_mat_inv = tf.linalg.inv(self._intrinsic_mat)
        else:
            self._fov = None

    @property
    def intrinsic_mat(self):
        return self._intrinsic_mat

    @intrinsic_mat.setter
    def intrinsic_mat(self, value):
        if value is not None:
            self._intrinsic_mat = value
            with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
                self.intrinsic_mat_inv = tf.linalg.inv(self._intrinsic_mat)
        else:
            assert(self.fov is not None)
            self.fov = self._fov

    @property
    def cam_to_world(self):
        return self._cam_to_world

    @cam_to_world.setter
    def cam_to_world(self, value):
        if value is not None:
            self._cam_to_world = value
            with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
                self.world_to_cam = tf.linalg.inv(self.cam_to_world)
        else:
            self._cam_to_world = None
            self.world_to_cam = None

    def state_dict(self):
        return {
            'position': self.position,
            'look_at': self.look_at,
            'up': self.up,
            'fov': self.fov,
            'cam_to_world': self._cam_to_world,
            'world_to_cam': self.world_to_cam,
            'intrinsic_mat': self._intrinsic_mat,
            'clip_near': self.clip_near,
            'resolution': self.resolution,
            'camera_type': self.camera_type
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        out = cls.__new__(Camera)
        out.position = state_dict['position']
        out.look_at = state_dict['look_at']
        out.up = state_dict['up']
        out.fov = state_dict['fov']
        out.cam_to_world = state_dict['cam_to_world']
        out.intrinsic_mat = state_dict['intrinsic_mat']
        out.clip_near = state_dict['clip_near']
        out.resolution = state_dict['resolution']
        out.camera_type = state_dict['camera_type']
        return out

def automatic_camera_placement(shapes: List,
                               resolution: Tuple[int, int]):
    """
        Given a list of shapes, generates camera parameters automatically
        using the bounding boxes of the shapes. Place the camera at
        some distances from the shapes, so that it can see all of them.
        Inspired by https://github.com/mitsuba-renderer/mitsuba/blob/master/src/librender/scene.cpp#L286
    """
    assert(tf.executing_eagerly())
    aabb_min = tf.constant((float('inf'), float('inf'), float('inf')))
    aabb_max = -tf.constant((float('inf'), float('inf'), float('inf')))
    for shape in shapes:
        v = shape.vertices    
        v_min = tf.reduce_min(v, 0)
        v_max = tf.reduce_max(v, 0)
        with tf.device('/device:cpu:' + str(pyredner.get_cpu_device_id())):
            v_min = tf.identity(v_min)
            v_max = tf.identity(v_max)
        aabb_min = tf.minimum(aabb_min, v_min)
        aabb_max = tf.maximum(aabb_max, v_max)
    assert(tf.reduce_all(tf.math.is_finite(aabb_min)) and tf.reduce_all(tf.math.is_finite(aabb_max)))
    center = (aabb_max + aabb_min) * 0.5
    extents = aabb_max - aabb_min
    max_extents_xy = tf.maximum(extents[0], extents[1])
    distance = max_extents_xy / (2 * math.tan(45 * 0.5 * math.pi / 180.0))
    max_extents_xyz = tf.maximum(extents[2], max_extents_xy)    
    return Camera(position = tf.stack((center[0], center[1], aabb_min[2] - distance)),
                  look_at = center,
                  up = tf.constant((0.0, 1.0, 0.0)),
                  fov = tf.constant([45.0]),
                  clip_near = 0.001 * float(distance),
                  resolution = resolution)

def generate_intrinsic_mat(fx: tf.Tensor,
                           fy: tf.Tensor,
                           skew: tf.Tensor,
                           x0: tf.Tensor,
                           y0: tf.Tensor):
    """
        | Generate the following 3x3 intrinsic matrix given the parameters.
        | fx, skew, x0
        |  0,   fy, y0
        |  0,    0,  1

        Parameters
        ==========
        fx: tf.Tensor
            Focal length at x dimension. 1D tensor with size 1.
        fy: tf.Tensor
            Focal length at y dimension. 1D tensor with size 1.
        skew: tf.Tensor
            Axis skew parameter describing shearing transform. 1D tensor with size 1.
        x0: tf.Tensor
            Principle point offset at x dimension. 1D tensor with size 1.
        y0: tf.Tensor
            Principle point offset at y dimension. 1D tensor with size 1.

        Returns
        =======
        tf.Tensor
            3x3 intrinsic matrix
    """
    z = tf.zeros_like(fx)
    o = tf.ones_like(fx)
    row0 = tf.concat([fx, skew, x0], axis=0)
    row1 = tf.concat([ z,   fy, y0], axis=0)
    row2 = tf.concat([ z,    z,  o], axis=0)
    return tf.stack([row0, row1, row2])
