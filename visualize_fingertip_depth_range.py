#!/usr/bin/env python3

import argparse
import cv2
import zmq
import numpy as np

from stretch4_gripper_modeling_and_control import gripper_networking as gn
from recv_and_detect_fingertips import FingertipDetector, add_fingertip_detector_args, process_fingertip_detector_args, find_closest_joint_state
from stretch4_gripper_modeling_and_control import aruco_to_fingertips as af
from stretch4_gripper_modeling_and_control.aruco_config import ArucoConfig
from stretch4_gripper_modeling_and_control import visualization_utils as vu
from stretch4_gripper_modeling_and_control import calibration_utils as cu

def main(use_remote_computer, args, model_path=None, use_robot_state=False, bg_visibility=0.6, fg_amplification=0.5, 
         display_scale=1.0, disable_suction_cups=False, swept_volume_to_pos=None, swept_volume_sampling_method='pos_pct', swept_volume_samples=30,
         swept_volume_mesh_visibility=1.0):
    if model_path is None:
        model_path = cu.get_default_model_path()
        
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
            depth_image = output_dict.get('depth_image', None)
            
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

            # Determine the depth range from the top circular edge of the suction cups
            h, w = color_image.shape[:2]
            min_depth_img = np.full((h, w), float('inf'), dtype=np.float32)
            max_depth_img = np.full((h, w), float('-inf'), dtype=np.float32)
            tube_mask_2d = np.zeros((h, w), dtype=np.uint8)
            has_valid_depth_range = False
            
            if swept_volume_to_pos is not None and validator is not None:
                from stretch4_gripper_modeling_and_control.swept_volume_model import SweptVolumeModel
                from stretch4_gripper_modeling_and_control import gripper_camera as gc
                for f_side in ['left', 'right']:
                    if f_side in fingertips:
                        sv_model = SweptVolumeModel(validator.model, f_side, fingertips[f_side], swept_volume_to_pos)
                        if sv_model.valid:
                            pcts = sv_model.get_sampled_pcts(sampling_method=swept_volume_sampling_method, num_samples=swept_volume_samples)
                            prev_pts_2d = None
                            prev_f_min = None
                            prev_f_max = None
                            for pct in pcts:
                                pos, rot = sv_model.get_frame(pct)
                                if pos is not None:
                                    x_axis = rot[:, 0]
                                    y_axis = rot[:, 1]
                                    dz = af.suctioncup_radius * np.sqrt(x_axis[2]**2 + y_axis[2]**2)
                                    f_min = pos[2] - dz
                                    f_max = pos[2] + dz
                                    
                                pts_3d = sv_model.get_circle_points(pct)
                                if pts_3d is not None:
                                    pts_2d = [np.round(gc.pixel_from_3d(p, rgb_camera_info)).astype(np.int32) for p in pts_3d]
                                    
                                    if prev_pts_2d is not None:
                                        all_pts = np.vstack((prev_pts_2d, pts_2d))
                                        hull = cv2.convexHull(all_pts)
                                        cv2.fillPoly(tube_mask_2d, [hull], 255, lineType=cv2.LINE_AA)
                                        
                                        seg_mask = np.zeros((h, w), dtype=np.uint8)
                                        cv2.fillPoly(seg_mask, [hull], 255, lineType=cv2.LINE_AA)
                                        b_mask = seg_mask > 0
                                        
                                        seg_min = min(prev_f_min, f_min)
                                        seg_max = max(prev_f_max, f_max)
                                        
                                        min_depth_img[b_mask] = np.minimum(min_depth_img[b_mask], seg_min)
                                        max_depth_img[b_mask] = np.maximum(max_depth_img[b_mask], seg_max)
                                        has_valid_depth_range = True
                                        
                                    prev_pts_2d = pts_2d
                                    prev_f_min = f_min
                                    prev_f_max = f_max
            else:
                glob_min = float('inf')
                glob_max = float('-inf')
                for side in ['left', 'right']:
                    f = fingertips.get(side)
                    if f is not None:
                        pos = f['pos']  # This is the 'cup_top' center point initialized in detector
                        x_axis = f['x_axis']
                        y_axis = f['y_axis']
                        
                        # The circular top edge is normal to z_axis and has radius `suctioncup_radius`
                        # We compute the max deviation in the Z depth axis caused by the circle's tilt
                        # dz = r * sin(angle_between_suctioncup_z_and_camera_z)
                        # Since x and y vectors are in the plane, the max deviation is given by:
                        dz = af.suctioncup_radius * np.sqrt(x_axis[2]**2 + y_axis[2]**2)
                        
                        finger_min = pos[2] - dz
                        finger_max = pos[2] + dz
                        glob_min = min(glob_min, finger_min)
                        glob_max = max(glob_max, finger_max)
                
                if glob_min != float('inf'):
                    min_depth_img[:] = glob_min
                    max_depth_img[:] = glob_max
                    tube_mask_2d[:] = 255
                    has_valid_depth_range = True
                    
            if depth_image is not None and has_valid_depth_range:
                # Convert depth image (mm) to meters for comparison
                depth_meters = depth_image / 1000.0
                
                # Create mask where depth falls within the pixel-wise valid range
                mask = (depth_meters >= min_depth_img) & (depth_meters <= max_depth_img) & (depth_meters > 0.0)
                
                if swept_volume_to_pos is not None and validator is not None:
                    # Enforce that segmented pixels must fall strictly within the 2D swept footprint
                    mask = mask & (tube_mask_2d > 0)
                
                # Darken background pixels to highlight the valid depth region
                dimmed_bg = cv2.addWeighted(color_image, bg_visibility, np.zeros_like(color_image), 0, 0)
                
                # Amplify foreground brightness if requested
                if fg_amplification > 0.0:
                    alpha = 1.0 + fg_amplification  # Scale factor up to 2.0
                    beta = int(127 * fg_amplification)  # Offset up to 127
                    bright_fg = cv2.convertScaleAbs(color_image, alpha=alpha, beta=beta)
                else:
                    bright_fg = color_image
                
                segmented_rgb = np.where(mask[:, :, np.newaxis], bright_fg, dimmed_bg)
            else:
                segmented_rgb = color_image.copy()

            # Draw fingertip frames
            # Using the returned segmented_rgb image which highlights the depth range
            segmented_rgb, scaled_camera_info = vu.apply_display_scale(segmented_rgb, display_scale, camera_info=rgb_camera_info)
            if not disable_suction_cups:
                detector.aruco_to_fingertips.draw_fingertip_suction_cups(fingertips, segmented_rgb, scaled_camera_info)

            if swept_volume_to_pos is not None and validator is not None:
                from stretch4_gripper_modeling_and_control.swept_volume_model import SweptVolumeModel
                from stretch4_gripper_modeling_and_control import gripper_camera as gc
                
                overlay = segmented_rgb.copy()
                draw_occurred = False
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
                                    pts_2d_np = np.array([pts_2d], dtype=np.int32)
                                    color = (180, 140, 70)
                                    
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
                    cv2.addWeighted(overlay, alpha, segmented_rgb, 1 - alpha, 0, segmented_rgb)
                    
                    if swept_volume_mesh_visibility > 0.0:
                        wire_overlay = segmented_rgb.copy()
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
                                            
                                            for i in range(num_points):
                                                next_i = (i + 1) % num_points
                                                mid_pt_3d = (pts_3d[i] + pts_3d[next_i]) / 2.0
                                                normal_3d = mid_pt_3d - pos
                                                if np.dot(normal_3d, mid_pt_3d) < 0:
                                                    cv2.line(wire_overlay, pts_2d[i], pts_2d[next_i], outline_color, 2, lineType=cv2.LINE_AA)
                                                    
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
                        
                        cv2.addWeighted(wire_overlay, swept_volume_mesh_visibility, segmented_rgb, 1 - swept_volume_mesh_visibility, 0, segmented_rgb)

            display_image = detector.draw_fingertip_frames(fingertips, segmented_rgb, scaled_camera_info)
            
            cv2.namedWindow("Fingertip Depth Range Segmentation", cv2.WINDOW_NORMAL)
            cv2.imshow("Fingertip Depth Range Segmentation", display_image)
            
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
        prog='Visualize Fingertip Depth Range',
        description='Segments RGB image based on the depth range bounding the suction cup fingertip edges.'
    )
    parser.add_argument('-r', '--remote', action='store_true', help='Use this argument when running the code on a remote computer. Configure gripper_networking.py first.') 
    add_fingertip_detector_args(parser)
    parser.add_argument('--model', type=str, default=None, help='Path to the model planar YAML file. Activates strict geometric validation against ArUco predictions. If not provided, defaults to the latest fleet calibration model.')
    parser.add_argument('--use_robot_state', action='store_true', help='Directly queries the physical stretch stretch_gripper pos_pct to further constrain geometric validation.')
    parser.add_argument('--bg_visibility', type=float, default=0.6, help='Background visibility for non-segmented regions (0.0 to 1.0, where 0.0 is completely black and 1.0 is fully visible).')
    parser.add_argument('--fg_amplification', type=float, default=0.5, help='Amplifies brightness of the segmented foreground region (0.0 keeps original, 1.0 sets maximum available brightening).')
    vu.add_suction_cup_argument(parser)
    parser.add_argument('--swept_volume_to_pos', nargs='?', const=0.0, type=float, default=None, help='Visualize and segment the volume swept out by the fingertips from their current configurations to the specified pos_pct (defaults to 0.0).')
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
    
    main(use_remote_computer, args, model_path=model_path, use_robot_state=use_robot_state, bg_visibility=args.bg_visibility, fg_amplification=args.fg_amplification, display_scale=args.display_scale, disable_suction_cups=args.disable_suction_cups, swept_volume_to_pos=swept_volume_to_pos, swept_volume_sampling_method=swept_volume_sampling_method, swept_volume_samples=swept_volume_samples, swept_volume_mesh_visibility=swept_volume_mesh_visibility)
