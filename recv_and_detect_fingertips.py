#!/usr/bin/env python3

import argparse
import cv2
import zmq
import numpy as np
import yaml
from yaml.loader import SafeLoader

from stretch4_gripper_modeling_and_control import gripper_networking as gn
from stretch4_gripper_modeling_and_control import aruco_detector as ad
from stretch4_gripper_modeling_and_control import aruco_to_fingertips as af
from stretch4_gripper_modeling_and_control import fingertip_filter as ff
from stretch4_gripper_modeling_and_control.aruco_config import ArucoConfig
from stretch4_gripper_modeling_and_control import visualization_utils as vu
from stretch4_gripper_modeling_and_control import calibration_utils as cu

import os

from stretch4_gripper_modeling_and_control.fingertip_detector import FingertipDetector, add_fingertip_detector_args, process_fingertip_detector_args

def find_closest_joint_state(joint_state_history, target_ts):
    """Find the joint state closest to the target timestamp from the history buffer."""
    if not joint_state_history:
        return None
    time_diffs = [abs(state['timestamp'] - target_ts) for state in joint_state_history]
    min_idx = np.argmin(time_diffs)
    return joint_state_history[min_idx]

def main(use_remote_computer, args, model_path=None, use_robot_state=False, display_scale=1.0, disable_suction_cups=False, side='both', swept_volume_to_pos=None, swept_volume_sampling_method='pos_pct', swept_volume_samples=30, swept_volume_mesh_visibility=1.0):
    validator = None
    if model_path is not None:
        from stretch4_gripper_modeling_and_control import aruco_kinematic_validator as akv
        print(f"Loading Kinematic Validator with model: {model_path}")
        validator = akv.ArucoKinematicValidator(model_path)
        
    if swept_volume_to_pos is not None and validator is None:
        print("WARNING: --swept_volume_to_pos requires a kinematic model to be provided via --model. Swept volume will not be modeled.")
        
    robot = None
    if use_robot_state:
        print("Connecting to local Stretch robot hardware to pipeline actuator state...")
        import stretch_body_ii.robot.robot_client as rc
        robot = rc.RobotClient()
        robot.startup()
        assert robot.is_homed(), "Robot is not homed! Cannot use absolute actuator states."

    # Initialize the detector
    detector = process_fingertip_detector_args(args, validator=validator)

    # Setup ZMQ subscriber for gripper images
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1)
    
    if use_remote_computer:
        address = 'tcp://' + gn.robot_ip + ':' + str(gn.gripper_and_joints_port)
    else:
        address = 'tcp://127.0.0.1:' + str(gn.gripper_and_joints_port)
        
    print(f"Connecting ZMQ Subscriber to {address}")
    socket.connect(address)

    # Placeholder camera intrinsics for the OAK-D SR (Stretch 4)
    # This will be updated later when proper calibration integration is added.
    camera_matrix = np.array([
        [425.0, 0.0, 320.0],
        [0.0, 425.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    distortion_coefficients = np.zeros(5, dtype=np.float64)

    rgb_camera_info = {
        'camera_matrix': camera_matrix,
        'distortion_coefficients': distortion_coefficients
    }

    print("Receiving frames... Press 'q' or 'Esc' to quit.")
    try:
        while True:
            output_dict = socket.recv_pyobj()
            if 'color_image_compressed' in output_dict:
                color_image = cv2.imdecode(np.frombuffer(output_dict['color_image_compressed'], np.uint8), cv2.IMREAD_COLOR)
            else:
                color_image = output_dict['color_image']
            
            # Dynamically pull factory calibration if available across network
            if 'camera_matrix' in output_dict and 'distortion_coefficients' in output_dict:
                rgb_camera_info['camera_matrix'] = output_dict['camera_matrix']
                rgb_camera_info['distortion_coefficients'] = output_dict['distortion_coefficients']

            pos_pct = None
            if 'joint_state_history' in output_dict:
                closest_js = find_closest_joint_state(output_dict['joint_state_history'], output_dict['system_timestamp'])
                if closest_js is not None:
                    pos_pct = closest_js['gripper']['pos_pct']
            elif use_robot_state and robot is not None:
                robot.pull_status()
                raw_gripper_status = robot.end_of_arm.status.get('stretch_gripper', {})
                pos_pct = raw_gripper_status.get('pos_pct', None)

            # Detect ArUco markers and find fingertips
            fingertips = detector.process_image(color_image, rgb_camera_info, pos_pct=pos_pct)
            
            if side != 'both':
                fingertips = {s: f for s, f in fingertips.items() if s == side}

            scaled_image, scaled_camera_info = vu.apply_display_scale(color_image, display_scale, camera_info=rgb_camera_info)

            if not disable_suction_cups:
                detector.aruco_to_fingertips.draw_fingertip_suction_cups(fingertips, scaled_image, scaled_camera_info)
                if getattr(args, 'draw_both_ippe', False):
                    alt_fingertips = {s: f['alt'] for s, f in fingertips.items() if 'alt' in f}
                    if alt_fingertips:
                        detector.aruco_to_fingertips.draw_fingertip_suction_cups(alt_fingertips, scaled_image, scaled_camera_info, color=(0, 0, 255))

            if swept_volume_to_pos is not None and validator is not None:
                from stretch4_gripper_modeling_and_control.swept_volume_model import SweptVolumeModel
                from stretch4_gripper_modeling_and_control import gripper_camera as gc
                
                overlay = scaled_image.copy()
                draw_occurred = False
                for f_side in ['left', 'right']:
                    if f_side in fingertips:
                        sv_model = SweptVolumeModel(validator.model, f_side, fingertips[f_side], swept_volume_to_pos)
                        if sv_model.valid:
                            pcts = sv_model.get_sampled_pcts(sampling_method=swept_volume_sampling_method, num_samples=swept_volume_samples)
                            prev_pts_2d = None
                            prev_pts_3d = None
                            for pct in pcts:
                                pos, rot = sv_model.get_frame(pct)
                                pts_3d = sv_model.get_circle_points(pct)
                                if pts_3d is not None and pos is not None:
                                    pts_2d = [np.round(gc.pixel_from_3d(p, scaled_camera_info)).astype(np.int32) for p in pts_3d]
                                    pts_2d_np = np.array([pts_2d], dtype=np.int32)
                                    color = (180, 140, 70)
                                    
                                    # Form a convex hull between previous and current circle to cleanly fill the volume
                                    if prev_pts_2d is not None:
                                        all_pts = np.vstack((prev_pts_2d, pts_2d))
                                        hull = cv2.convexHull(all_pts)
                                        cv2.fillPoly(overlay, [hull], color, lineType=cv2.LINE_AA)
                                        
                                    prev_pts_2d = pts_2d
                                    prev_pts_3d = pts_3d
                                    prev_pos = pos
                                    draw_occurred = True
                
                if draw_occurred:
                    alpha = 0.15
                    cv2.addWeighted(overlay, alpha, scaled_image, 1 - alpha, 0, scaled_image)
                    
                    if swept_volume_mesh_visibility > 0.0:
                        wire_overlay = scaled_image.copy()
                        # Second Pass: Draw Front-Facing visible surface lines with High Contrast
                        outline_color = (180, 120, 40)
                        for f_side in ['left', 'right']:
                            if f_side in fingertips:
                                sv_model = SweptVolumeModel(validator.model, f_side, fingertips[f_side], swept_volume_to_pos)
                                if sv_model.valid:
                                    pcts = sv_model.get_sampled_pcts(sampling_method=swept_volume_sampling_method, num_samples=swept_volume_samples)
                                    prev_pts_2d = None
                                    prev_pts_3d = None
                                    prev_pos = None
                                    for pct in pcts:
                                        pos, rot = sv_model.get_frame(pct)
                                        pts_3d = sv_model.get_circle_points(pct)
                                        if pts_3d is not None and pos is not None:
                                            pts_2d = [np.round(gc.pixel_from_3d(p, scaled_camera_info)).astype(np.int32) for p in pts_3d]
                                            num_points = len(pts_3d)
                                            
                                            # Draw front-facing circle segments
                                            for i in range(num_points):
                                                next_i = (i + 1) % num_points
                                                mid_pt_3d = (pts_3d[i] + pts_3d[next_i]) / 2.0
                                                normal_3d = mid_pt_3d - pos
                                                if np.dot(normal_3d, mid_pt_3d) < 0: # Facing camera
                                                    cv2.line(wire_overlay, pts_2d[i], pts_2d[next_i], outline_color, 2, lineType=cv2.LINE_AA)
                                            
                                            # Draw front-facing longitudinal lines
                                            if prev_pts_2d is not None:
                                                for i in range(num_points):
                                                    mid_pt_3d = (pts_3d[i] + prev_pts_3d[i]) / 2.0
                                                    mid_pos = (pos + prev_pos) / 2.0
                                                    normal_3d = mid_pt_3d - mid_pos
                                                    if np.dot(normal_3d, mid_pt_3d) < 0:
                                                        cv2.line(wire_overlay, prev_pts_2d[i], pts_2d[i], outline_color, 1, lineType=cv2.LINE_AA)
                                            
                                            prev_pts_2d = pts_2d
                                            prev_pts_3d = pts_3d
                                            prev_pos = pos
                        
                        cv2.addWeighted(wire_overlay, swept_volume_mesh_visibility, scaled_image, 1 - swept_volume_mesh_visibility, 0, scaled_image)

            # Draw fingertip frames
            task_relevant_image = detector.draw_fingertip_frames(fingertips, scaled_image, scaled_camera_info, draw_both_ippe=getattr(args, 'draw_both_ippe', False))
            
            cv2.namedWindow("Stretch 4 Fingertip Detection", cv2.WINDOW_NORMAL)
            cv2.imshow("Stretch 4 Fingertip Detection", task_relevant_image)
            
            key = cv2.waitKey(1)
            if key in (27, ord('q')): # Esc or q
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if robot is not None:
            robot.stop()
        print("\nStopped receiving.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Detect and Draw Fingertips',
        description='Receives gripper camera images, detects ArUco markers, and draws fingertip frames.'
    )
    parser.add_argument('-r', '--remote', action='store_true', help='Use this argument when running the code on a remote computer. Configure gripper_networking.py first.') 
    add_fingertip_detector_args(parser)

    parser.add_argument('--model', type=str, default=None, help='Path to the model planar YAML file. Activates strict geometric validation against ArUco predictions. If not provided, defaults to the latest fleet calibration model.')
    parser.add_argument('--use_robot_state', action='store_true', help='Directly queries the physical stretch stretch_gripper pos_pct to further constrain geometric validation.')
    vu.add_suction_cup_argument(parser)
    parser.add_argument('--draw_both_ippe', action='store_true', help='Render both poses returned by the IPPE algorithm.')
    parser.add_argument('--side', type=str, choices=['both', 'left', 'right'], default='both', help='Which side to render (both, left, right).')
    parser.add_argument('--swept_volume_to_pos', nargs='?', const=0.0, type=float, default=None, help='Visualize the volume swept out by the fingertips from their current configurations to the specified pos_pct (defaults to 0.0).')
    parser.add_argument('--swept_volume_sampling_method', type=str, choices=['pos_pct', 'arc_length'], default='pos_pct', help='Method to uniformly sample the swept volume.')
    parser.add_argument('--swept_volume_samples', type=int, default=30, help='Number of uniform samples for the swept volume.')
    parser.add_argument('--swept_volume_mesh_visibility', type=float, default=1.0, help='Visibility of the swept volume wiremesh (0.0 means not visible, 1.0 means maximum visibility).')
    vu.add_display_scale_argument(parser)

    args = parser.parse_args()

    if args.model is None:
        args.model = cu.get_default_model_path()
        if args.model:
            print(f"No model path provided. Defaulting to fleet calibration: {args.model}")
    
    use_remote_computer = args.remote
    model_path = args.model
    use_robot_state = args.use_robot_state
    swept_volume_to_pos = args.swept_volume_to_pos
    swept_volume_sampling_method = args.swept_volume_sampling_method
    swept_volume_samples = args.swept_volume_samples
    swept_volume_mesh_visibility = args.swept_volume_mesh_visibility
    
    main(use_remote_computer, args, model_path=model_path, use_robot_state=use_robot_state, display_scale=args.display_scale, disable_suction_cups=args.disable_suction_cups, side=args.side, swept_volume_to_pos=swept_volume_to_pos, swept_volume_sampling_method=swept_volume_sampling_method, swept_volume_samples=swept_volume_samples, swept_volume_mesh_visibility=swept_volume_mesh_visibility)
