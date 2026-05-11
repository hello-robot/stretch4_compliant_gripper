import numpy as np
import pathlib
import os
import errno
import yourdfpy as urdf_loader
import cv2
from stretch4_gripper_modeling_and_control import gripper_camera as gc

def load_urdf(file_name):
    if not os.path.isfile(file_name):
        print()
        print('*****************************')
        print('ERROR: ' + file_name + ' was not found. OptasIK requires a specialized URDF saved with this file name. prepare_base_rotation_ik_urdf.py can be used to generate this specialized URDF.')
        print('*****************************')
        print()
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_name)
    import warnings
    warnings.filterwarnings('ignore', message='Unable to resolve filename')
    urdf = urdf_loader.URDF.load(file_name, load_meshes=False, build_collision_scene_graph=False)
    return(urdf)

# key heights along the central axis of the suction cup in meters
# total height from the bottom to the top of the suction cup should be 0.015 m
suctioncup_height = {
    'cup_top': 0.015, # Top of the suction cup, which corresponds with the circular edge of the cup
    'cup_bottom': 0.0074, # Bottom of the interior of the suction cup, which has thickness that places it above the top of the cylinder
    'cylinder_top': 0.00415, # Top of the cylindrical base of the suction cup
    'cylinder_bottom': 0.0 # Bottom of the cylindrical base of the suction cup
}

# diameter and radius of the top most part of the suction cup in meters
suctioncup_diameter = 0.037 
suctioncup_radius = suctioncup_diameter/2.0

# diameter and radius of the cylindrical base of the suction cup in meters
cylinder_diameter = 0.0154
cylinder_radius = cylinder_diameter/2.0

# The suction cup matches a minor spherical cap from a sphere with this diameter in meters. 
# The spherical cap rests on the top of the cylindrical base, such that the outer circle of 
# the cylinder's top end is on the surface of the spherical cap.
suctioncup_sphere_diameter = 0.04
suctioncup_sphere_radius = suctioncup_sphere_diameter / 2.0

# ArUco to suction cup correction in meters
aruco_to_suction_cup_bottom_correction = 0.002

class ArucoToFingertips:
    def __init__(self, urdf_filename=None, default_height_above_mounting_surface=None):

        self.default_height_above_mounting_surface = default_height_above_mounting_surface
        
        if urdf_filename is None: 
            self.urdf_filename = os.path.join(os.path.dirname(__file__), 'minimal_stretch_gripper.urdf')
        else:
            self.urdf_filename = urdf_filename

        urdf = load_urdf(self.urdf_filename)

        self.marker_left_name ='finger_left'
        self.marker_right_name = 'finger_right'
        self.marker_names = [self.marker_left_name, self.marker_right_name]
        
        self.fingertip_basename = 'link_gripper_fingertip_'
        self.aruco_basename = 'link_aruco_fingertip_'

        self.sides = ['left', 'right']

        self.transforms = {}
        self.translations = {}
        self.rotations = {}
                
        for side in self.sides:
            fingertip_link_name = self.fingertip_basename + side
            aruco_link_name = self.aruco_basename + side

            fingertip_transform = urdf.scene.graph.get(fingertip_link_name)[0]
            aruco_transform = urdf.scene.graph.get(aruco_link_name)[0]

            F = fingertip_transform
            A = aruco_transform

            # A T = F
            # T = A^(-1) F

            A_inv = np.linalg.inv(A)
            T = np.matmul(A_inv, F)
            
            # Apply physical suction cup bottom correction along local Z-axis
            Transl = np.eye(4)
            Transl[2, 3] = aruco_to_suction_cup_bottom_correction
            T = np.matmul(T, Transl)

            aruco_to_fingertip_transform = T

            self.transforms[side] = T

            self.translations[side] = np.copy(T[:3,3].flatten())

            self.rotations[side] = np.copy(T[:3,:3])

            
    def get_transforms(self):
        return self.transforms

    def get_rotations(self):
        return self.rotations

    def get_translations(self):
        return self.translations

    def get_fingertips(self, markers, height_above_mounting_surface=None):
        # Find the fingertip poses using finger ArUco markers observed from a gripper camera.

        fingertips = {}
        
        for k in markers:
            m = markers[k]
            name = m['info']['name']
            if name in self.marker_names:
                marker_pos = m['pos']
                marker_x_axis = m['x_axis']
                marker_y_axis = m['y_axis']
                marker_z_axis = m['z_axis']

                if 'left' in name:
                    side = 'left'
                else:
                    side = 'right'

                t = self.translations[side]

                A = np.zeros((3,3))
                A[:,0] = marker_x_axis.flatten()
                A[:,1] = marker_y_axis.flatten()
                A[:,2] = marker_z_axis.flatten()

                T = self.rotations[side]

                F = np.matmul(A, T)
                
                fingertip_x_axis = F[:,0].flatten()
                fingertip_y_axis = F[:,1].flatten()
                fingertip_z_axis = F[:,2].flatten()

                if (height_above_mounting_surface is None) and (self.default_height_above_mounting_surface is None):
                    # Use the bottom of the rubber cylinder at the
                    # base of the suction cup fingertip, which is also
                    # the top surface of the mounting surface for
                    # other fingertips.
                    fingertip_pos = marker_pos + np.matmul(A, t)
                elif height_above_mounting_surface is not None:
                    fingertip_pos = (marker_pos + np.matmul(A, t))  + (height_above_mounting_surface * fingertip_z_axis)
                elif self.default_height_above_mounting_surface is not None:
                    fingertip_pos = (marker_pos + np.matmul(A, t))  + (self.default_height_above_mounting_surface * fingertip_z_axis)
                    
                fingertips[side] = {'pos': fingertip_pos,
                                    'x_axis': fingertip_x_axis,
                                    'y_axis': fingertip_y_axis,
                                    'z_axis': fingertip_z_axis}

                if m.get('alt_rvec') is not None:
                    A_alt, _ = cv2.Rodrigues(m['alt_rvec'])
                    marker_pos_alt = m['alt_tvec'].reshape(-1) / 1000.0
                    F_alt = np.matmul(A_alt, T)
                    
                    if (height_above_mounting_surface is None) and (self.default_height_above_mounting_surface is None):
                        fingertip_pos_alt = marker_pos_alt + np.matmul(A_alt, t)
                    elif height_above_mounting_surface is not None:
                        fingertip_pos_alt = (marker_pos_alt + np.matmul(A_alt, t)) + (height_above_mounting_surface * F_alt[:,2].flatten())
                    elif self.default_height_above_mounting_surface is not None:
                        fingertip_pos_alt = (marker_pos_alt + np.matmul(A_alt, t)) + (self.default_height_above_mounting_surface * F_alt[:,2].flatten())
                        
                    fingertips[side]['alt'] = {
                        'pos': fingertip_pos_alt,
                        'x_axis': F_alt[:,0].flatten(),
                        'y_axis': F_alt[:,1].flatten(),
                        'z_axis': F_alt[:,2].flatten()
                    }

        return fingertips

    
    def draw_fingertip_origins(self, fingertips, image, camera_info):

        origins_3d = []
        sides = ['left', 'right']
        for side in sides: 
            f = fingertips.get(side, None)
            if f is not None: 
                origins_3d.append(f['pos'])
        
        origin_pixels = [gc.pixel_from_3d(p, camera_info) for p in origins_3d]

        origins_image = image
        radius = 6
        color = (255, 255, 255)
        thickness = 2
        for p in origin_pixels:
            center = np.round(p).astype(np.int32)
            cv2.circle(origins_image, center, radius, color, thickness) 

            
    def draw_fingertip_frames(self, fingertips, image, camera_info, axis_length_in_m=0.02, draw_origins=True, write_coordinates=False, desaturated=False):

        # colors are in BGR format
        sides = ['left', 'right']
        if desaturated:
            axes = [('x_axis', (100, 100, 150)),
                    ('y_axis', (100, 150, 100)),
                    ('z_axis', (150, 100, 100))]
        else:
            axes = [('x_axis', (0, 0, 255)),
                    ('y_axis', (0, 255, 0)),
                    ('z_axis', (255, 0, 0))]
        thickness = 3
        origin_radius = 6
                
        for side in sides: 
            f = fingertips.get(side, None)
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
                        color = (255, 255, 255)
                        cv2.circle(image, d['pix'], origin_radius, color, -1, lineType=cv2.LINE_AA)
                    if (t == 'axis'): 
                        cv2.line(image, d['base_pix'], d['tip_pix'], d['color'], thickness, lineType=cv2.LINE_AA)
                    
                if write_coordinates:
                    x,y,z = origin * 100.0
                    text = "{:.1f}, {:.1f}, {:.1f} cm".format(x,y,z)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_size = 0.4

                    text_size = cv2.getTextSize(text, font, font_size, 2)
                    (text_width, text_height), text_baseline = text_size

                    shift = int(2.5 * origin_radius)
                    
                    if side == 'right': 
                        location = origin_image + np.array([shift, int(text_height/2)])
                    else:
                        location = origin_image + np.array([-(text_width + shift), int(text_height/2)])
                    cv2.putText(image, text, location, font, font_size, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, text, location, font, font_size, (255, 255, 255), 1, cv2.LINE_AA)
                    
    color=(0, 165, 255) #gold
    def draw_fingertip_suction_cups(self, fingertips, image, camera_info, color=(255, 0, 0), thickness=1, alpha=0.4):
        num_points = 16
        num_cap_rings = 5
        
        overlay = image.copy()
        draw_occurred = False
        
        for side in self.sides:
            f = fingertips.get(side, None)
            if f is not None:
                draw_occurred = True
                
                current_height = self.default_height_above_mounting_surface if self.default_height_above_mounting_surface is not None else 0.0
                base_pos = f['pos'] - current_height * f['z_axis']
                
                x_axis = f['x_axis']
                y_axis = f['y_axis']
                z_axis = f['z_axis']
                
                h_cup_top = suctioncup_height.get('cup_top', 0.015)
                h_cyl_top = suctioncup_height.get('cylinder_top', 0.00415)
                h_cyl_bot = suctioncup_height.get('cylinder_bottom', 0.0)
                
                R_s = suctioncup_sphere_radius
                
                # The spherical cap connects perfectly to the fixed rigid cylinder boundary
                z_c_offset = np.sqrt(max(0, R_s**2 - cylinder_radius**2))
                Z_c_local = h_cyl_top + z_c_offset
                
                rings_3d = []
                
                # 1. Cylinder bottom ring
                ring_bot = []
                for i in range(num_points):
                    theta = 2.0 * np.pi * i / num_points
                    ring_bot.append((h_cyl_bot, cylinder_radius, theta))
                rings_3d.append(ring_bot)
                
                # 2. Cylinder top ring (rigid connection interface)
                ring_cyl_top = []
                for i in range(num_points):
                    theta = 2.0 * np.pi * i / num_points
                    ring_cyl_top.append((h_cyl_top, cylinder_radius, theta))
                rings_3d.append(ring_cyl_top)
                
                # 3. Spherical cap rings
                ring_z_vals = np.linspace(h_cyl_top, h_cup_top, num_cap_rings + 1)
                for z_val in ring_z_vals[1:]: 
                    r_val = np.sqrt(max(0, R_s**2 - (Z_c_local - z_val)**2))
                    ring = []
                    for i in range(num_points):
                        theta = 2.0 * np.pi * i / num_points
                        ring.append((z_val, r_val, theta))
                    rings_3d.append(ring)
                    
                # Convert to explicit 3D positions
                rings_actual_3d = []
                for ring in rings_3d:
                    ring_pts = []
                    for (z_val, r_val, theta) in ring:
                        pt = base_pos + z_val * z_axis + r_val * np.cos(theta) * x_axis + r_val * np.sin(theta) * y_axis
                        ring_pts.append(pt)
                    rings_actual_3d.append(ring_pts)
                
                faces = []
                
                # Top cap
                faces.append({
                    'points_3d': rings_actual_3d[-1],
                    'center': np.mean(rings_actual_3d[-1], axis=0)
                })
                # Bottom cap
                faces.append({
                    'points_3d': rings_actual_3d[0],
                    'center': np.mean(rings_actual_3d[0], axis=0)
                })
                
                # Side quads between successive rings
                for r_idx in range(len(rings_actual_3d) - 1):
                    ring_lower = rings_actual_3d[r_idx]
                    ring_upper = rings_actual_3d[r_idx + 1]
                    for i in range(num_points):
                        next_i = (i + 1) % num_points
                        quad_3d = [ring_upper[i], ring_upper[next_i], ring_lower[next_i], ring_lower[i]]
                        faces.append({
                            'points_3d': quad_3d,
                            'center': np.mean(quad_3d, axis=0)
                        })
                
                # Sort faces by depth (z-coordinate in camera frame) descending
                faces.sort(key=lambda face: face['center'][2], reverse=True)
                
                outline_color = (0, 0, 0)
                for face in faces:
                    pts_2d = [np.round(gc.pixel_from_3d(p, camera_info)).astype(np.int32) for p in face['points_3d']]
                    pts_2d_np = np.array([pts_2d], dtype=np.int32)
                    
                    cv2.fillPoly(overlay, pts_2d_np, color, lineType=cv2.LINE_AA)
                    cv2.polylines(overlay, pts_2d_np, True, outline_color, thickness, lineType=cv2.LINE_AA)            
        
        if draw_occurred:
            if alpha < 1.0:
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            else:
                np.copyto(image, overlay)
                    
if __name__ == '__main__':
    
    aruco_to_fingertips = ArucoToFingertips(default_height_above_mounting_surface=suctioncup_height['cup_bottom'])
    aruco_to_fingertip_transforms = aruco_to_fingertips.get_transforms()
    aruco_to_fingertip_translations = aruco_to_fingertips.get_translations()
    aruco_to_fingertip_rotations = aruco_to_fingertips.get_rotations()
    
    np.set_printoptions(precision=3, linewidth=100, suppress=True)

    print('------------------')
    print('aruco_to_fingertip_transforms:')
    print()
    print('left =')
    print(aruco_to_fingertip_transforms['left'])
    print()
    print('right =')
    print(aruco_to_fingertip_transforms['right'])

    print()
    
    print('------------------')
    print('aruco_to_fingertip_translations:')
    print()
    print('left =')
    print(aruco_to_fingertip_translations['left'])
    print()
    print('right =')
    print(aruco_to_fingertip_translations['right'])

    print()
    
    print('------------------')
    print('aruco_to_fingertip_rotations:')
    print()
    print('left =')
    print(aruco_to_fingertip_rotations['left'])
    print()
    print('right =')
    print(aruco_to_fingertip_rotations['right'])
