#!/usr/bin/env python3

import argparse
import time
import collections
import cv2
import zmq
import numpy as np

from stretch4_gripper_modeling_and_control import gripper_networking as gn
from stretch4_gripper_modeling_and_control import gripper_camera as gc
from recv_and_detect_fingertips import FingertipDetector, add_fingertip_detector_args, process_fingertip_detector_args
from visualize_fingertip_model import FingertipVisualizer
from stretch4_gripper_modeling_and_control.temporal_estimation import AdaptiveBaselineEstimator
from stretch4_gripper_modeling_and_control import visualization_utils as vu
from stretch4_gripper_modeling_and_control import calibration_utils as cu

def _old_draw_predicted_frames(predicted_fingertips, image, camera_info, axis_length_in_m=0.02, draw_origins=True):
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


def draw_bar_graph(image, left_disp, right_disp, left_twist, right_twist, aperture_diff, vis_scale=1.0):
    """Draw a compact real-time bar graph on the image showing loading metrics."""
    h, w, _ = image.shape
    
    # Box parameters
    box_w = w - 20
    box_h = 160
    box_x = 10
    box_y = 10
    
    # Draw semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    
    # Padding and text setup
    padding = 15
    y_offset = box_y + 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    
    pixels_per_unit = 12.0 * vis_scale
    bar_x = box_x + 105
    bar_max_w = box_w - 150
    
    def draw_bar(label, val, y_pos, color):
        cv2.putText(image, label, (box_x + padding, y_pos), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Calculate pixel width based on fixed scale
        bar_w = int(abs(val) * pixels_per_unit)
        # Clamp to available width to prevent drawing outside the box
        bar_w = min(bar_w, bar_max_w)
        
        cv2.rectangle(image, (bar_x, y_pos - 10), (bar_x + bar_max_w, y_pos), (50, 50, 50), -1)
        
        # Color intensity based on sign for aperture diff, or just solid for magnitudes
        bar_color = color if val >= 0 else (0, 0, 255) # Red if aperture diff is negative
        
        cv2.rectangle(image, (bar_x, y_pos - 10), (bar_x + bar_w, y_pos), bar_color, -1)
        
        val_text = f"{val:4.1f}"
        cv2.putText(image, val_text, (bar_x + bar_w + 5, y_pos), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # Drawing the bars
    draw_bar("L Disp (mm)", left_disp, y_offset, (0, 255, 255))
    y_offset += 25
    draw_bar("R Disp (mm)", right_disp, y_offset, (0, 255, 255))
    y_offset += 25
    draw_bar("L Twist (deg)", left_twist, y_offset, (255, 0, 255))
    y_offset += 25
    draw_bar("R Twist (deg)", right_twist, y_offset, (255, 0, 255))
    y_offset += 25
    draw_bar("Ap Diff (mm)", aperture_diff, y_offset, (0, 255, 0))

def draw_magnitude_analysis(predicted_fingertips, vis_fingertips, image, vis_scale=1.0):
    left_disp, right_disp = 0.0, 0.0
    left_twist, right_twist = 0.0, 0.0
    aperture_diff = 0.0
    
    # Displacements and Twists
    for side in ['left', 'right']:
        if side in predicted_fingertips and side in vis_fingertips:
            f_pred = predicted_fingertips[side]
            f_vis = vis_fingertips[side]
            
            # Translation (mm)
            disp = np.linalg.norm(f_pred['pos'] - f_vis['pos']) * 1000.0
            
            # Twist (degrees)
            R_pred = np.column_stack((f_pred['x_axis'], f_pred['y_axis'], f_pred['z_axis']))
            R_vis = np.column_stack((f_vis['x_axis'], f_vis['y_axis'], f_vis['z_axis']))
            R_diff = R_pred.T @ R_vis
            trace = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
            twist_deg = np.degrees(np.arccos(trace))
            
            if side == 'left':
                left_disp, left_twist = disp, twist_deg
            else:
                right_disp, right_twist = disp, twist_deg

    # Aperture Difference (mm)
    if 'left' in predicted_fingertips and 'right' in predicted_fingertips and \
       'left' in vis_fingertips and 'right' in vis_fingertips:
        ap_pred = np.linalg.norm(predicted_fingertips['left']['pos'] - predicted_fingertips['right']['pos']) * 1000.0
        ap_vis = np.linalg.norm(vis_fingertips['left']['pos'] - vis_fingertips['right']['pos']) * 1000.0
        aperture_diff = ap_pred - ap_vis
        
    draw_bar_graph(image, left_disp, right_disp, left_twist, right_twist, aperture_diff, vis_scale)


def draw_plane_analysis(predicted_fingertips, vis_fingertips, image, visualizer, vis_scale=1.0):
    normal = visualizer.normal.copy()
    centroid = visualizer.centroid.copy()
    
    if np.dot(-centroid, normal) < 0:
        normal = -normal
        
    z_axis = normal
    
    cam_z = np.array([0.0, 0.0, 1.0])
    x_axis_unnorm = cam_z - np.dot(cam_z, z_axis) * z_axis
    if np.linalg.norm(x_axis_unnorm) > 1e-6:
        x_axis = x_axis_unnorm / np.linalg.norm(x_axis_unnorm)
    else:
        cam_y = np.array([0.0, 1.0, 0.0])
        y_proj = cam_y - np.dot(cam_y, z_axis) * z_axis
        x_axis = y_proj / np.linalg.norm(y_proj)
        
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    left_x, left_y, left_z = 0.0, 0.0, 0.0
    right_x, right_y, right_z = 0.0, 0.0, 0.0
    
    for side in ['left', 'right']:
        if side in predicted_fingertips and side in vis_fingertips:
            f_pred = predicted_fingertips[side]
            f_vis = vis_fingertips[side]
            
            E = (f_vis['pos'] - f_pred['pos']) * 1000.0 # difference in mm
            
            # Signed displacements
            dx = np.dot(E, x_axis)        # forward/backward
            dy = np.dot(E, -y_axis)       # rightward/leftward
            dz = np.dot(E, z_axis)        # upward/downward
            
            if side == 'left':
                left_x, left_y, left_z = dx, dy, dz
            else:
                right_x, right_y, right_z = dx, dy, dz

    h, w, _ = image.shape
    box_w = w - 20
    box_h = 185
    box_x = 10
    box_y = 10
    
    overlay = image.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    
    padding = 15
    y_offset = box_y + 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    
    pixels_per_unit = 12.0 * vis_scale
    center_x = box_x + 150 + (box_w - 150) // 2
    max_w = (box_w - 150) // 2 - 45 # Space for drawing bar and text
    
    def draw_symmetric_bar(label, val, y_pos, color):
        cv2.putText(image, label, (box_x + padding, y_pos), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(image, (center_x, y_pos - 15), (center_x, y_pos + 5), (100, 100, 100), 2)
        
        bar_w = int(abs(val) * pixels_per_unit)
        bar_w = min(bar_w, max_w)
        
        if val >= 0:
            cv2.rectangle(image, (center_x, y_pos - 10), (center_x + bar_w, y_pos), color, -1)
            val_text = f"+{val:4.1f}"
            cv2.putText(image, val_text, (center_x + bar_w + 5, y_pos), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(image, (center_x - bar_w, y_pos - 10), (center_x, y_pos), color, -1)
            val_text = f"{val:4.1f}"
            tw = cv2.getTextSize(val_text, font, font_scale, 1)[0][0]
            cv2.putText(image, val_text, (center_x - bar_w - tw - 5, y_pos), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # Drawing the bars
    draw_symmetric_bar("L Fwd/Bkwd (X)", left_x, y_offset, (0, 255, 255))
    y_offset += 25
    draw_symmetric_bar("R Fwd/Bkwd (X)", right_x, y_offset, (0, 255, 255))
    y_offset += 25
    draw_symmetric_bar("L Up/Dwn (Z)", left_z, y_offset, (255, 0, 255))
    y_offset += 25
    draw_symmetric_bar("R Up/Dwn (Z)", right_z, y_offset, (255, 0, 255))
    y_offset += 25
    draw_symmetric_bar("L R/L (-Y)", left_y, y_offset, (0, 255, 0))
    y_offset += 25
    draw_symmetric_bar("R R/L (-Y)", right_y, y_offset, (0, 255, 0))


def draw_cross_hud_analysis(predicted_fingertips, vis_fingertips, image, visualizer, vis_scale=1.0):
    normal = visualizer.normal.copy()
    centroid = visualizer.centroid.copy()
    if np.dot(-centroid, normal) < 0: normal = -normal
    z_axis = normal
    
    cam_z = np.array([0.0, 0.0, 1.0])
    x_axis_unnorm = cam_z - np.dot(cam_z, z_axis) * z_axis
    if np.linalg.norm(x_axis_unnorm) > 1e-6:
        x_axis = x_axis_unnorm / np.linalg.norm(x_axis_unnorm)
    else:
        cam_y = np.array([0.0, 1.0, 0.0])
        y_proj = cam_y - np.dot(cam_y, z_axis) * z_axis
        x_axis = y_proj / np.linalg.norm(y_proj)
        
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    left_x, left_y, left_z = 0.0, 0.0, 0.0
    right_x, right_y, right_z = 0.0, 0.0, 0.0
    
    for side in ['left', 'right']:
        if side in predicted_fingertips and side in vis_fingertips:
            f_pred = predicted_fingertips[side]
            f_vis = vis_fingertips[side]
            E = (f_vis['pos'] - f_pred['pos']) * 1000.0
            # Calculate translations
            dx, dy, dz = np.dot(E, x_axis), np.dot(E, -y_axis), np.dot(E, z_axis)
            
            # Calculate rotation around z_axis (normal to finger plane)
            R_pred = np.column_stack((f_pred['x_axis'], f_pred['y_axis'], f_pred['z_axis']))
            R_vis = np.column_stack((f_vis['x_axis'], f_vis['y_axis'], f_vis['z_axis']))
            R_diff = R_vis @ R_pred.T
            rvec, _ = cv2.Rodrigues(R_diff)
            twist_z_rad = np.dot(rvec.flatten(), z_axis)
            twist_z_deg = np.degrees(twist_z_rad)
            
            if side == 'left':
                left_x, left_y, left_z = twist_z_deg, dy, dz
            else:
                # Flip sign for right fingertip to make inward/outward rotations mirror symmetric
                right_x, right_y, right_z = -twist_z_deg, dy, dz

    h, w, _ = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    w_bar = 20
    gap = 15
    max_len = 65
    
    ppu_z = (max_len / 10.0) * vis_scale  # [-10mm, 10mm]
    ppu_y = (max_len / 20.0) * vis_scale  # [-20mm, 20mm]
    ppu_rot = (max_len / 5.0) * vis_scale  # [-5 deg, 5 deg]
    
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (320, 250), (0, 0, 0), -1)
    cv2.rectangle(overlay, (w - 320, 0), (w, 250), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    def draw_hud_cross(val_z, val_y, val_rot, cx, cy, x_fwd_offset, label_prefix):
        # Z (up/down) Vertical
        bz = min(int(abs(val_z) * ppu_z), max_len)
        cv2.rectangle(image, (cx - w_bar//2, cy - gap - max_len), (cx + w_bar//2, cy - gap), (100,100,100), -1)
        cv2.rectangle(image, (cx - w_bar//2, cy + gap), (cx + w_bar//2, cy + gap + max_len), (100,100,100), -1)
        if val_z >= 0:
            cv2.rectangle(image, (cx - w_bar//2, cy - gap - bz), (cx + w_bar//2, cy - gap), (255, 0, 255), -1)
        else:
            cv2.rectangle(image, (cx - w_bar//2, cy + gap), (cx + w_bar//2, cy + gap + bz), (255, 0, 255), -1)
        
        # Y (right/left) Horizontal
        by = min(int(abs(val_y) * ppu_y), max_len)
        cv2.rectangle(image, (cx + gap, cy - w_bar//2), (cx + gap + max_len, cy + w_bar//2), (100,100,100), -1)
        cv2.rectangle(image, (cx - gap - max_len, cy - w_bar//2), (cx - gap, cy + w_bar//2), (100,100,100), -1)
        if val_y >= 0:
            cv2.rectangle(image, (cx + gap, cy - w_bar//2), (cx + gap + by, cy + w_bar//2), (0, 255, 0), -1)
        else:
            cv2.rectangle(image, (cx - gap - by, cy - w_bar//2), (cx - gap, cy + w_bar//2), (0, 255, 0), -1)
        
        # Rot Z (inward/outward) Vertical at xf
        xf = cx + x_fwd_offset
        bx = min(int(abs(val_rot) * ppu_rot), max_len)
        cv2.rectangle(image, (xf - w_bar//2, cy - max_len), (xf + w_bar//2, cy + max_len), (100,100,100), -1)
        if val_rot >= 0:
            cv2.rectangle(image, (xf - w_bar//2, cy - bx), (xf + w_bar//2, cy), (0, 255, 255), -1)
        else:
            cv2.rectangle(image, (xf - w_bar//2, cy), (xf + w_bar//2, cy + bx), (0, 255, 255), -1)
            
        # Origin/Center outline
        cv2.rectangle(image, (cx - gap, cy - gap), (cx + gap, cy + gap), (200, 200, 200), 1)
        cv2.line(image, (xf - w_bar//2 - 5, cy), (xf + w_bar//2 + 5, cy), (200, 200, 200), 1)
        
        # Labels
        cv2.putText(image, f"{label_prefix} Finger", (cx - 30, cy - gap - max_len - 15), font, font_scale+0.1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f"Z: {val_z:+.1f}", (cx - 20, cy + gap + max_len + 15), font, font_scale, (255, 0, 255), 1, cv2.LINE_AA)
        
        val_y_str = f"Y: {val_y:+.1f}"
        tw = cv2.getTextSize(val_y_str, font, font_scale, 1)[0][0]
        if x_fwd_offset < 0:
            cv2.putText(image, val_y_str, (cx + gap + max_len + 5, cy + 4), font, font_scale, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(image, val_y_str, (cx - gap - max_len - tw - 5, cy + 4), font, font_scale, (0, 255, 0), 1, cv2.LINE_AA)
            
        cv2.putText(image, f"RotZ:{val_rot:+.1f}dg", (xf - 35, cy + max_len + 15), font, font_scale, (0, 255, 255), 1, cv2.LINE_AA)

    # Cross Displays at Top Left/Right Corners
    draw_hud_cross(left_z, left_y, left_x, cx=150, cy=105, x_fwd_offset=-110, label_prefix="Left")
    draw_hud_cross(right_z, right_y, right_x, cx=w-150, cy=105, x_fwd_offset=110, label_prefix="Right")


def main():
    parser = argparse.ArgumentParser(description='Estimate fingertip loads using visually tracked vs kinematically predicted frames.')
    parser.add_argument('--model', type=str, default=None, help='Path to the fingertip kinematic model YAML file. If not provided, defaults to the latest fleet calibration model.')
    parser.add_argument('-r', '--remote', action='store_true', help='Use this argument when running on a remote computer.')
    parser.add_argument('--interpolate', action='store_true', help='Use linear interpolation for pos_pct instead of nearest neighbor.')
    parser.add_argument('--mode', type=str, choices=['magnitude', 'plane', 'cross_hud'], default='magnitude', help='Visualization and calculation mode')
    parser.add_argument('-t', '--temporal', action='store_true', help='Use temporal estimation for baseline instead of kinematic model')
    parser.add_argument('--vis_scale_kinematic', type=float, default=1.0, help='Visualization scale multiplier for kinematic mode')
    parser.add_argument('--vis_scale_temporal', type=float, default=5.0, help='Visualization scale multiplier for temporal mode')
    add_fingertip_detector_args(parser)
    vu.add_suction_cup_argument(parser)
    vu.add_display_scale_argument(parser)
    args = parser.parse_args()

    if args.model is None:
        args.model = cu.get_default_model_path()
        if args.model:
            print(f"No model path provided. Defaulting to fleet calibration: {args.model}")
        else:
            print("Error: No model path provided and could not locate latest_model_planar.yaml in fleet directory.")
            import sys
            sys.exit(1)

    vis_scale = args.vis_scale_temporal if args.temporal else args.vis_scale_kinematic

    print("Loading Fingertip Kinematic Model...")
    visualizer = FingertipVisualizer(args.model)

    if args.temporal:
        print("Initializing Temporal Baseline Estimator...")
        estimator = AdaptiveBaselineEstimator()

    robot = None

    # Initialize FingertipDetector
    print(f"Initializing Fingertip Detector... (smooth={getattr(args, 'smooth', False)})")
    detector = process_fingertip_detector_args(args)

    # Setup ZMQ subscriber for gripper images
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1)
    
    port = gn.gripper_and_joints_port
    address = f"tcp://{gn.robot_ip if args.remote else '127.0.0.1'}:{port}"
    print(f"Connecting ZMQ Subscriber to {address}")
    socket.connect(address)

    # Camera Intrinsics 
    rgb_camera_info = {
        'camera_matrix': np.array([
            [425.0, 0.0, 320.0],
            [0.0, 425.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64),
        'distortion_coefficients': np.zeros(5, dtype=np.float64)
    }

    # State history buffer
    status_history = collections.deque(maxlen=1000) # Holds ~5 seconds of state
    
    def get_pos_pct_at_time(target_ts):
        """Retrieve pos_pct at target_ts using interpolation or nearest neighbor."""
        if len(status_history) == 0:
            return None, 'closing', 0.0
            
        timestamps = np.array([s['ts'] for s in status_history])
        pcts = np.array([s['pct'] for s in status_history])

        # Direction estimate over recent window
        if len(status_history) >= 10:
            recent_pcts = pcts[-10:]
            diff = recent_pcts[-1] - recent_pcts[0]
            direction = 'opening' if diff > 0 else 'closing'
        else:
            direction = 'closing'
        
        if not args.interpolate:
            idx = np.argmin(np.abs(timestamps - target_ts))
            return pcts[idx], direction, target_ts - timestamps[idx]
        else:
            # Linear interpolation
            if target_ts <= timestamps[0]:
                return pcts[0], direction, target_ts - timestamps[0]
            elif target_ts >= timestamps[-1]:
                return pcts[-1], direction, target_ts - timestamps[-1]
            else:
                pct_interp = np.interp(target_ts, timestamps, pcts)
                return pct_interp, direction, 0.0

    print("Running Load Estimation... Press 'q' or 'Esc' to quit.")
    
    cv2.namedWindow("Fingertip Loading Estimator", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # Try to grab an image (NOBLOCK)
            try:
                output_dict = socket.recv_pyobj(flags=zmq.NOBLOCK)
                
                if 'camera_matrix' in output_dict and 'distortion_coefficients' in output_dict:
                    rgb_camera_info['camera_matrix'] = output_dict['camera_matrix']
                    rgb_camera_info['distortion_coefficients'] = output_dict['distortion_coefficients']

                if 'color_image_compressed' in output_dict:
                    color_image = cv2.imdecode(np.frombuffer(output_dict['color_image_compressed'], np.uint8), cv2.IMREAD_COLOR)
                else:
                    color_image = output_dict['color_image']
                sys_timestamp = output_dict['system_timestamp']
                
                for st in output_dict.get('joint_state_history', []):
                    status_history.append({'ts': st['timestamp'], 'pct': st['gripper']['pos_pct']})

                closest_joint_state = output_dict.get('closest_joint_state')
                if closest_joint_state is not None:
                    matched_pct = closest_joint_state['gripper']['pos_pct']
                    debug_diff = closest_joint_state['time_relative_to_image']
                    
                    if len(status_history) >= 10:
                        pcts = np.array([s['pct'] for s in status_history])
                        recent_pcts = pcts[-10:]
                        diff = recent_pcts[-1] - recent_pcts[0]
                        direction = 'opening' if diff > 0 else 'closing'
                    else:
                        direction = 'closing'
                else:
                    matched_pct = None
                    direction = 'closing'
                    debug_diff = 0.0
                
                if matched_pct is not None:
                    print(f"Timing Debug -> Image vs Status Diff: {debug_diff:.4f} s | Current Age: {(time.time() - sys_timestamp):.4f} s")
                
                predicted_fingertips = {}
                if matched_pct is not None:
                    for side in ['left', 'right']:
                        pos_pred, rot_pred = visualizer.predict(side, matched_pct, direction)
                        if pos_pred is not None and rot_pred is not None:
                            predicted_fingertips[side] = {
                                'pos': pos_pred,
                                'x_axis': rot_pred[:, 0],
                                'y_axis': rot_pred[:, 1],
                                'z_axis': rot_pred[:, 2]
                            }

                # Visually estimate loaded frames
                vis_fingertips = detector.process_image(color_image, rgb_camera_info, pos_pct=matched_pct)
                
                if args.temporal:
                    predicted_fingertips = estimator.update(matched_pct, vis_fingertips)
                    
                # Visualizations
                # 1. Base image copy
                display_img = color_image.copy()
                
                display_img, scaled_camera_info = vu.apply_display_scale(display_img, args.display_scale, camera_info=rgb_camera_info)

                # 2. Draw Kinematics (Desaturated)
                if predicted_fingertips:
                    vu.draw_predicted_frames(predicted_fingertips, display_img, scaled_camera_info)
                    if not args.disable_suction_cups:
                        detector.aruco_to_fingertips.draw_fingertip_suction_cups(
                            predicted_fingertips, display_img, scaled_camera_info,
                            color=(128, 0, 0), alpha=0.4
                        )
                
                # 3. Draw Visually Estimated (Bright)
                detector.aruco_to_fingertips.draw_fingertip_frames(
                    vis_fingertips, display_img, scaled_camera_info,
                    axis_length_in_m=0.02, draw_origins=True, write_coordinates=False
                )
                if not args.disable_suction_cups and vis_fingertips:
                    detector.aruco_to_fingertips.draw_fingertip_suction_cups(
                        vis_fingertips, display_img, scaled_camera_info,
                        color=(255, 0, 0), alpha=0.4
                    )
                
                # 4. Draw Appropriate Analysis UI
                if args.mode == 'plane':
                    draw_plane_analysis(predicted_fingertips, vis_fingertips, display_img, visualizer, vis_scale=vis_scale)
                elif args.mode == 'cross_hud':
                    draw_cross_hud_analysis(predicted_fingertips, vis_fingertips, display_img, visualizer, vis_scale=vis_scale)
                else:
                    draw_magnitude_analysis(predicted_fingertips, vis_fingertips, display_img, vis_scale=vis_scale)
                
                cv2.imshow("Fingertip Loading Estimator", display_img)
                key = cv2.waitKey(1)
                if key in (27, ord('q')):
                    break

            except zmq.Again:
                # No new image, loop without doing anything to let robot status update
                time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if robot is not None:
            robot.stop()
        print("\nStopped.")

if __name__ == '__main__':
    main()
