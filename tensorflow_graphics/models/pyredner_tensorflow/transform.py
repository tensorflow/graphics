import math
import numpy as np
import tensorflow as tf

def radians(deg):
    return (math.pi / 180.0) * deg

def normalize(v):
    """

    NOTE: torch.norm() uses Frobineus norm which is Euclidean and L2
    """
    return v / tf.norm(v)

def gen_look_at_matrix(pos, look, up):
    d = normalize(look - pos)
    right = normalize(tf.linalg.cross(d, normalize(up)))
    new_up = normalize(tf.linalg.cross(right, d))
    z = tf.constant(np.zeros([1]), dtype=tf.float32)
    o = tf.convert_to_tensor(np.ones([1], dtype=np.float32), dtype=tf.float32)
    return tf.transpose(tf.stack([tf.concat([right , z], 0),
                                  tf.concat([new_up, z], 0),
                                  tf.concat([d     , z], 0),
                                  tf.concat([pos   , o], 0)]))

def gen_scale_matrix(scale):
    o = tf.convert_to_tensor(np.ones([1], dtype=np.float32), dtype=tf.float32)
    return tf.linalg.tensor_diag(tf.concat([scale, o], 0))

def gen_translate_matrix(translate):
    z = tf.constant(np.zeros([1]), dtype=tf.float32)
    o = tf.convert_to_tensor(np.ones([1], dtype=np.float32), dtype=tf.float32)
    return tf.stack([tf.concat([o, z, z, translate[0:1]], 0),
                        tf.concat([z, o, z, translate[1:2]], 0),
                        tf.concat([z, z, o, translate[2:3]], 0),
                        tf.concat([z, z, z, o], 0)])

def gen_perspective_matrix(fov, clip_near, clip_far):
    clip_dist = clip_far - clip_near
    cot = 1 / tf.tan(radians(fov / 2.0))
    z = tf.constant(np.zeros([1]), dtype=tf.float32)
    o = tf.convert_to_tensor(np.ones([1], dtype=np.float32), dtype=tf.float32)
    return tf.stack([tf.concat([cot,   z,             z,                       z], 0),
                     tf.concat([  z, cot,             z,                       z], 0),
                     tf.concat([  z,   z, 1 / clip_dist, - clip_near / clip_dist], 0),
                     tf.concat([  z,   z,             o,                       z], 0)])

def gen_rotate_matrix(angles: tf.Tensor) -> tf.Tensor:
    """
        Given a 3D Euler angle vector, outputs a rotation matrix.

        Args
        ====
            angles: torch.Tensor
                3D Euler angle

        Returns
        =======
            tf.Tensor
                3x3 rotation matrix
    """
    theta = angles[0]
    phi = angles[1]
    psi = angles[2]

    rot_x = tf.stack([
        tf.constant([1.0,0,0]),
        tf.constant([0.0, 1.0, 0.0]) * tf.cos(theta) + tf.constant([0.0, 0.0, 1.0]) * tf.sin(theta),
        tf.constant([0.0, 1.0, 0.0]) * -tf.sin(theta) + tf.constant([0.0, 0.0, 1.0]) * tf.cos(theta),
        ]
    )

    rot_y = tf.stack([
        tf.constant([1.0, 0.0, 0.0]) * tf.cos(phi) + tf.constant([0.0, 0.0, 1.0]) * -tf.sin(phi),
        tf.constant([0.0, 1.0, 0.0]),
        tf.constant([1.0, 0.0, 0.0]) * tf.sin(phi) + tf.constant([0.0, 0.0, 1.0]) * tf.cos(phi),
        ]
    )

    rot_z = tf.stack([
        tf.constant([1.0, 0.0, 0.0]) * tf.cos(psi) + tf.constant([0.0, 1.0, 0.0]) * -tf.sin(psi),
        tf.constant([1.0, 0.0, 0.0]) * tf.sin(psi) + tf.constant([0.0, 1.0, 0.0]) * tf.cos(psi),
        tf.constant([0.0, 0.0, 1.0]),
        ]
    )
    return rot_z @ (rot_y @ rot_x)
  
