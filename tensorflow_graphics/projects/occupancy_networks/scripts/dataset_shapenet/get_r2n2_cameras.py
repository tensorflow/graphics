import argparse
import numpy as np


parser = argparse.ArgumentParser(
    description='Convert camera from Choy et al. format to npz file.')
parser.add_argument('in_file', type=str,
                    help='Location of metadata input file.')
parser.add_argument('out_file', type=str,
                    help='Location of output npz file.')

F_MM = 35.  # Focal length
SENSOR_SIZE_MM = 32.

PIXEL_ASPECT_RATIO = 1.  # pixel_aspect_x / pixel_aspect_y
RESOLUTION_PCT = 100
SKEW = 0.
CAM_MAX_DIST = 1.75

IMG_W = 127 + 10  # Rendering image size. Network input size + cropping margin.
IMG_H = 127 + 10


CAM_ROT = np.matrix(((1.910685676922942e-15, 4.371138828673793e-08, 1.0),
                     (1.0, -4.371138828673793e-08, -0.0),
                     (4.371138828673793e-08, 1.0, -4.371138828673793e-08)))

blender_T = np.array([
    [1, 0., 0],
    [0, 0, -1],
    [0, 1, 0.],
])


def getBlenderProj(az, el, distance_ratio, img_w=IMG_W, img_h=IMG_H):
    """Calculate 4x3 3D to 2D projection matrix given viewpoint parameters."""

    # Calculate intrinsic matrix.
    scale = RESOLUTION_PCT / 100
    f_u = F_MM * img_w * scale / SENSOR_SIZE_MM
    f_v = F_MM * img_h * scale * PIXEL_ASPECT_RATIO / SENSOR_SIZE_MM
    u_0 = img_w * scale / 2
    v_0 = img_h * scale / 2
    K = np.matrix(((f_u, SKEW, u_0), (0, f_v, v_0), (0, 0, 1)))

    # Calculate rotation and translation matrices.
    # Step 1: World coordinate to object coordinate.
    sa = np.sin(np.radians(-az))
    ca = np.cos(np.radians(-az))
    se = np.sin(np.radians(-el))
    ce = np.cos(np.radians(-el))
    R_world2obj = np.transpose(np.matrix(((ca * ce, -sa, ca * se),
                                          (sa * ce, ca, sa * se),
                                          (-se, 0, ce))))

    # Step 2: Object coordinate to camera coordinate.
    R_obj2cam = np.transpose(np.matrix(CAM_ROT))
    R_world2cam = R_obj2cam * R_world2obj
    cam_location = np.transpose(np.matrix((distance_ratio * CAM_MAX_DIST,
                                           0,
                                           0)))
    T_world2cam = -1 * R_obj2cam * cam_location

    # Step 3: Fix blender camera's y and z axis direction.
    R_camfix = np.matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    R_world2cam = R_camfix * R_world2cam
    T_world2cam = R_camfix * T_world2cam

    RT = np.hstack((R_world2cam, T_world2cam))

    return K, RT


def main(args):
    with open(args.in_file, 'r') as f:
        lines = f.readlines()

    data = np.array([[float(e) for e in l.split()] for l in lines])
    out_dict = {}
    for idx in range(data.shape[0]):
        K, RT = getBlenderProj(data[idx, 0], data[idx, 1], data[idx, 3])
        RT = np.asarray(RT)
        K = np.asarray(K)
        M = RT[:, :3] @ blender_T
        c = RT[:, 3:]
        out_dict['camera_mat_%d' % idx] = K
        out_dict['world_mat_%d' % idx] = np.concatenate([M, c], axis=1)

    np.savez(args.out_file, **out_dict)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
