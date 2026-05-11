import argparse
import cv2
import copy

def add_display_scale_argument(parser):
    """Adds the --display_scale argument to an existing argparse.ArgumentParser."""
    parser.add_argument('--display_scale', type=float, default=1.0, 
                        help='Scale factor for the OpenCV visualization image (1.0 = native, 2.0 = double size, 0.5 = half size).')

def add_suction_cup_argument(parser):
    """Adds the --disable_suction_cups argument to an existing argparse.ArgumentParser."""
    parser.add_argument('--disable_suction_cups', action='store_true', 
                        help='Disable rendering of the 3D suction cups.')

def apply_display_scale(image, scale, camera_info=None):
    """
    Scales the image by the given scale factor prior to annotation.
    If camera_info is provided, scales the intrinsic camera matrix logically to match the new image dimensions.
    Returns:
        scaled_image: The resized image.
        scaled_camera_info (optional): The mathematically matched visual camera info dict.
    """
    if scale == 1.0:
        if camera_info is not None:
            return image, camera_info
        return image

    scaled_image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    if camera_info is not None:
        scaled_camera_info = copy.deepcopy(camera_info)
        # Scale focal lengths and principal center points linearly
        scaled_camera_info['camera_matrix'][0, 0] *= scale
        scaled_camera_info['camera_matrix'][1, 1] *= scale
        scaled_camera_info['camera_matrix'][0, 2] *= scale
        scaled_camera_info['camera_matrix'][1, 2] *= scale
        return scaled_image, scaled_camera_info
        
    return scaled_image

import numpy as np
from stretch4_gripper_modeling_and_control import gripper_camera as gc

def draw_predicted_frames(predicted_fingertips, image, camera_info, axis_length_in_m=0.02, draw_origins=True):
    """Draw predicted frames with desaturated/darker colors compared to visually estimated frames."""
    sides = ['left', 'right']
    axes = [('x_axis', (0, 0, 128)),     # Dark Red
            ('y_axis', (0, 128, 0)),     # Dark Green
            ('z_axis', (128, 0, 0))]     # Dark Blue
    thickness = 2
    origin_radius = 5
            
    for side in sides: 
        f = predicted_fingertips.get(side, None)
        if f is not None:
            to_draw = []
            origin = f['pos']
            origin_camera = gc.pixel_from_3d(origin, camera_info)
            origin_image = np.round(origin_camera).astype(np.int32)
            to_draw.append({'type': 'origin',
                            'z': origin[2],
                            'pix': origin_image})

            for axis, color in axes:
                axis_tip = (axis_length_in_m * (f[axis] - origin)) + origin
                axis_tip_camera = gc.pixel_from_3d(axis_tip, camera_info)
                axis_tip_image = np.round(axis_tip_camera).astype(np.int32)
                to_draw.append({'type': 'axis',
                                'z': axis_tip[2],
                                'base_pix': origin_image,
                                'tip_pix': axis_tip_image,
                                'color': color})

            to_draw_by_z = sorted(to_draw, key=lambda element: element['z'], reverse=True)

            for d in to_draw_by_z:
                t = d['type']
                if (t == 'origin') and draw_origins:
                    color = (128, 128, 128) # Gray origin
                    cv2.circle(image, d['pix'], origin_radius, color, -1, lineType=cv2.LINE_AA)
                if (t == 'axis'): 
                    cv2.line(image, d['base_pix'], d['tip_pix'], d['color'], thickness, lineType=cv2.LINE_AA)
