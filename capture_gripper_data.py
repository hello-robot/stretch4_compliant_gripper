#!/usr/bin/env python3

import os
import time
import zmq
import yaml
import numpy as np
import cv2
import argparse
import stretch_body_ii.robot.robot_client as rc
from stretch4_gripper_modeling_and_control import gripper_networking as gn
from stretch4_gripper_modeling_and_control import telemetry_utils as tu
from recv_and_detect_fingertips import FingertipDetector, add_fingertip_detector_args, process_fingertip_detector_args

def format_se3(se3_obj):
    """Convert an SE3 numpy matrix or dictionary (including nested ones) to a list of lists for YAML formatting."""
    if isinstance(se3_obj, np.ndarray):
        return se3_obj.tolist()
    elif isinstance(se3_obj, dict):
        return {k: format_se3(v) for k, v in se3_obj.items()}
    elif isinstance(se3_obj, (list, tuple)):
        return [format_se3(item) for item in se3_obj]
    elif np.isscalar(se3_obj) and hasattr(se3_obj, 'item'):
        return se3_obj.item()
    return se3_obj

def main():
    parser = argparse.ArgumentParser(description='Collect synchronized gripper and camera data.')
    add_fingertip_detector_args(parser)
    args = parser.parse_args()

    # Status keys to save
    status_keys_to_print = ['current_mA', 'effort', 'pos_pct', 'vel', 'is_moving', 'timestamp_pc', 'gripper_conversion']
    
    # Initialize the robot
    print("Connecting to robot...")
    robot = rc.RobotClient()
    robot.startup()
    assert robot.is_homed()

    # Initialize FingertipDetector
    print("Initializing Fingertip Detector...")
    detector = process_fingertip_detector_args(args)

    # Setup ZMQ subscriber for gripper images
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    # We want the newest image to be read
    socket.setsockopt(zmq.CONFLATE, 1)

    # Use local gripper and joints port
    address = 'tcp://127.0.0.1:' + str(gn.gripper_and_joints_port)
    print(f"Connecting ZMQ Subscriber to {address}")
    socket.connect(address)
    
    # Wait for the first message to guarantee the stream is active
    print("Waiting for camera stream... (Ensure send_gripper_images_and_joint_states.py is running)")
    try:
        # Use blocking recv
        socket.recv_pyobj()
        print("Camera stream active! First image received.")
    except Exception as e:
        import sys
        print(f"Error connecting to camera stream: {e}")
        robot.stop()
        sys.exit(1)

    # Creating directories
    current_time_str = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"gripper_calibration_{current_time_str}"
    closing_dir = os.path.join(output_dir, "closing_images")
    opening_dir = os.path.join(output_dir, "opening_images")
    os.makedirs(closing_dir, exist_ok=True)
    os.makedirs(opening_dir, exist_ok=True)
    print(f"Created calibration directory: {output_dir}")

    # Motion parameters
    loop_sleep_time = 0.001
    pos_pct_epsilon = 5.0

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

    # Helper function to move and collect
    def collect_data(motion_direction):
        fully_done = False
        jsh = tu.JointStateHistory(maxlen=100000, warn_on_discontinuity=False)
        images_data = []

        motion_verb = "close" if motion_direction == 'closing' else "open"
        print(f"\nCommanding gripper to {motion_verb}...")
        start_time = time.time()

        robot.pull_status()
        raw_gripper_status = robot.end_of_arm.status['stretch_gripper']
        last_image_pos_pct = raw_gripper_status['pos_pct']

        while not fully_done:
            # Command movement
            step_size = 1
            if motion_direction == 'closing':
                robot.end_of_arm.move_by('stretch_gripper', -step_size)
            else:
                robot.end_of_arm.move_by('stretch_gripper', step_size)
                
            robot.push_command()
            robot.pull_status()
            raw_gripper_status = robot.end_of_arm.status['stretch_gripper']
            
            # Try to grab an image (NOBLOCK)
            try:
                output_dict = socket.recv_pyobj(flags=zmq.NOBLOCK)
                
                if 'joint_state_history' in output_dict:
                    jsh.add_states(output_dict['joint_state_history'])
                
                # Dynamically pull factory calibration if available
                if 'camera_matrix' in output_dict and 'distortion_coefficients' in output_dict:
                    rgb_camera_info['camera_matrix'] = output_dict['camera_matrix']
                    rgb_camera_info['distortion_coefficients'] = output_dict['distortion_coefficients']

                if 'color_image_compressed' in output_dict:
                    color_image = cv2.imdecode(np.frombuffer(output_dict['color_image_compressed'], np.uint8), cv2.IMREAD_COLOR)
                else:
                    color_image = output_dict['color_image']
                sys_timestamp = output_dict['system_timestamp']
                img_num = output_dict['image_number']
                
                rgb_camera_info['image_resolution'] = [color_image.shape[1], color_image.shape[0]]
                
                # Update tracking variable since we got an image!
                last_image_pos_pct = raw_gripper_status['pos_pct']

                # Use the synced closest pos_pct for detector input, or fallback to polled
                closest_js = output_dict.get('closest_joint_state')
                sync_pos_pct = closest_js['gripper']['pos_pct'] if closest_js else raw_gripper_status['pos_pct']

                fingertips = detector.process_image(color_image, rgb_camera_info, pos_pct=sync_pos_pct)
                if 'left' in fingertips and 'right' in fingertips:
                    images_data.append({
                        'left_frame': format_se3(fingertips['left']),
                        'right_frame': format_se3(fingertips['right']),
                        'fingertips_raw': fingertips,
                        'image': color_image,
                        'image_timestamp': sys_timestamp,
                        'image_number': img_num
                    })
            except zmq.Again:
                if abs(raw_gripper_status['pos_pct'] - last_image_pos_pct) > 20.0:
                    import sys
                    error_msg = f"\nERROR: Camera image stream connection lost. " \
                                f"Gripper traversed >5% of its range (from {last_image_pos_pct:.1f} to {raw_gripper_status['pos_pct']:.1f}) without receiving a single image. " \
                                f"\nAborting calibration."
                    print(error_msg)
                    robot.stop()
                    sys.exit(1)

            # Check if fully done
            if motion_direction == 'closing':
                if raw_gripper_status['pos_pct'] <= -(100 - pos_pct_epsilon):
                    fully_done = True
            else:
                if raw_gripper_status['pos_pct'] >= 300 - pos_pct_epsilon:
                    fully_done = True

            time.sleep(loop_sleep_time)

        print(f"Motion completed. Synchronizing and annotating {len(images_data)} captured images...")
        
        # Get all assembled joint states
        all_joint_states = jsh.get_history_list()

        # Synchronize data and annotate
        synchronized_data = []
        for img_entry in images_data:
            img_ts = img_entry['image_timestamp']
            
            # Note: the joint states sent over ZMQ store their PC timestamp in 'timestamp_pc' within the 'gripper' dict
            # Wait, no, the root dict has 'timestamp_pc'. Let's verify. RobotStatePoller adds `timestamp_pc = gripper_st.get('timestamp_pc', 0.0)` ... wait.
            # No, RobotStatePoller has data = {'gripper': {...}, ...}. Does it add 'timestamp_pc' to the root dict?
            # It uses `state['timestamp'] = time.time()` and `state['monotonic_timestamp'] = time.monotonic()`
            
            # Find closest before
            before_candidates = [s for s in all_joint_states if s.get('timestamp') is not None and s['timestamp'] <= img_ts]
            # Find closest after
            after_candidates = [s for s in all_joint_states if s.get('timestamp') is not None and s['timestamp'] > img_ts]
            
            if before_candidates and after_candidates:
                closest_before_state = max(before_candidates, key=lambda s: s['timestamp'])
                closest_after_state = min(after_candidates, key=lambda s: s['timestamp'])
                
                # Format to match the legacy 'gripper_status_before' shape for fit_fingertip_model.py
                # fit_fingertip_model.py expects: entry.get('gripper_status_before').get('pos_pct')
                # So we just provide a dict with 'pos_pct' mapped from closest_before_state['gripper']['pos_pct']
                img_entry['gripper_status_before'] = {'pos_pct': closest_before_state['gripper']['pos_pct']}
                img_entry['gripper_status_after'] = {'pos_pct': closest_after_state['gripper']['pos_pct']}
                
                # Remove objects that should not be serialized
                img_entry.pop('fingertips_raw')
                color_image = img_entry.pop('image')
                
                # Save Raw Image
                target_dir = closing_dir if motion_direction == 'closing' else opening_dir
                raw_filename = f"raw_image_{img_entry['image_number']:05d}.png"
                img_entry['image_filename'] = raw_filename
                
                raw_path = os.path.join(target_dir, raw_filename)
                cv2.imwrite(raw_path, color_image)
                
                synchronized_data.append(img_entry)

        return synchronized_data


    # First: orient the wrist and fully open the gripper without capturing (prep)
    print("Orienting wrist and fully opening the gripper (preparation)...")
    robot.end_of_arm.move_to('wrist_yaw', 0.0)
    robot.end_of_arm.move_to('wrist_pitch', 0.0)
    robot.end_of_arm.move_to('wrist_roll', 0.0)
    robot.end_of_arm.move_to('stretch_gripper', 300)
    robot.push_command()
    time.sleep(3)

    # Now capture data while closing
    closing_data = collect_data('closing')

    # Now capture data while opening
    opening_data = collect_data('opening')

    robot.stop()

    print(f"Captured {len(closing_data)} valid synchronous frames during closing.")
    print(f"Captured {len(opening_data)} valid synchronous frames during opening.")

    # Save to yaml files
    robot_id = os.environ.get('HELLO_FLEET_ID', 'unknown_robot')
    
    metadata = {
        'robot_id': robot_id,
        'capture_time': current_time_str,
        'image_resolution': rgb_camera_info.get('image_resolution', [640, 400]),
        'camera_matrix': rgb_camera_info['camera_matrix'].tolist() if isinstance(rgb_camera_info['camera_matrix'], np.ndarray) else rgb_camera_info['camera_matrix'],
        'distortion_coefficients': rgb_camera_info['distortion_coefficients'].tolist() if isinstance(rgb_camera_info['distortion_coefficients'], np.ndarray) else rgb_camera_info['distortion_coefficients']
    }
    
    closing_output = {
        'metadata': metadata,
        'data': closing_data
    }
    opening_output = {
        'metadata': metadata,
        'data': opening_data
    }

    closing_filename = os.path.join(output_dir, f'closing_gripper_data_{current_time_str}.yaml')
    opening_filename = os.path.join(output_dir, f'opening_gripper_data_{current_time_str}.yaml')

    with open(closing_filename, 'w') as f:
        yaml.dump(closing_output, f, sort_keys=False)
    print(f"\nSaved {closing_filename}")

    with open(opening_filename, 'w') as f:
        yaml.dump(opening_output, f, sort_keys=False)
    print(f"Saved {opening_filename}")

if __name__ == '__main__':
    main()
