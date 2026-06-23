#!/usr/bin/env python3
import argparse
import os
import time
import threading
import collections
import zmq
import numpy as np
import copy

import stretch4_body.robot.robot_client as rc
from stretch4_gripper_modeling_and_control.gripper_camera import GripperCamera, add_camera_args, process_camera_args
from stretch4_gripper_modeling_and_control import gripper_networking as gn

class RobotStatePoller:
    def __init__(self, robot):
        self.robot = robot
        self.history_buffer = []
        self.lock = threading.Lock()
        self.running = True
        self.last_ts = None
        self.state_counter = 0
        
        self.thread = threading.Thread(target=self._poll_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def _poll_loop(self):
        while self.running:
            self.robot.pull_status()
            st = self.robot.status
            eoa = st.get('end_of_arm', {})
            gripper_st = eoa.get('stretch_gripper', {})
            
            # The stretch_gripper timestamp is a good pulse check for new lower-level samples
            curr_ts = gripper_st.get('timestamp_pc', 0.0)
            
            if self.last_ts is not None and curr_ts == self.last_ts:
                # No new data update from lower level hardware
                time.sleep(0.002)
                continue
                
            self.last_ts = curr_ts
            
            base = st.get('base', {})
            lift_st = st.get('lift', {})
            arm_st = st.get('arm', {})
            
            wrist_yaw = eoa.get('wrist_yaw', {})
            wrist_pitch = eoa.get('wrist_pitch', {})
            wrist_roll = eoa.get('wrist_roll', {})

            data = {
                'gripper': {
                    'pos_pct': gripper_st.get('pos_pct', 0.0), 
                    'effort': gripper_st.get('effort', 0.0)
                },
                'lift': {
                    'height': lift_st.get('pos', 0.0)
                },
                'arm': {
                    'extension': arm_st.get('pos', 0.0)
                },
                'wrist_yaw': {
                    'angle': wrist_yaw.get('pos', 0.0), 
                    'effort': wrist_yaw.get('effort', 0.0)
                },
                'wrist_pitch': {
                    'angle': wrist_pitch.get('pos', 0.0), 
                    'effort': wrist_pitch.get('effort', 0.0)
                },
                'wrist_roll': {
                    'angle': wrist_roll.get('pos', 0.0), 
                    'effort': wrist_roll.get('effort', 0.0)
                },
                'base_odometry': {
                    'x': base.get('x', 0.0), 
                    'y': base.get('y', 0.0), 
                    'theta': base.get('theta', 0.0)
                },
                'timestamp': time.time(),
                'monotonic_timestamp': time.monotonic(),
                'state_number': self.state_counter
            }
            
            with self.lock:
                self.history_buffer.append(data)
                self.state_counter += 1
                
            time.sleep(0.002) # Ensure we don't thrash CPU, ~500Hz max polling
            
    def stop(self):
        self.running = False
        self.thread.join()
        
    def get_and_clear_history(self):
        with self.lock:
            history = list(self.history_buffer)
            self.history_buffer.clear()
            return history


def main(use_remote_computer, device_id, center_device_id, use_gripper, use_center, image_size, compress, auto_fps, oak_buffer_size):
    print("Starting Robot Client...")
    robot = rc.RobotClient()
    robot.startup()
    
    if not robot.is_homed():
        print("WARNING: Robot is not homed. Joint values may be incorrect.")
        
    poller = RobotStatePoller(robot)

    camera = None
    try: 
        print(f"Initializing depthai pipeline (Gripper: {use_gripper}, Center: {use_center}, Size: {image_size}, FPS: {auto_fps}, Compress: {compress}, Buffer Size: {oak_buffer_size})...")
        camera = GripperCamera(device_id=device_id, center_device_id=center_device_id, fps=auto_fps, image_size=image_size, use_gripper=use_gripper, use_center=use_center, compress=compress, oak_buffer_size=oak_buffer_size)
        camera.start()
        
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.setsockopt(zmq.RCVHWM, 1)
        
        if use_remote_computer:
            address = 'tcp://*:' + str(gn.gripper_and_joints_port)
        else:
            address = 'tcp://127.0.0.1:' + str(gn.gripper_and_joints_port)
            
        print(f"Binding ZMQ Publisher to {address}")
        socket.bind(address)
        
        gn.print_network_info()
        
        M_right, D_right = None, None
        if use_gripper:
            M_right, D_right = camera.get_gripper_intrinsics()
            
        # Keep 1.0 second of history (max 500 polls at 500Hz) to guarantee we cover camera latency
        sliding_window = collections.deque(maxlen=500)
            
        robot_id = os.environ.get('HELLO_FLEET_ID')
        print("Sending synced frames and joint states... Press Ctrl+C to stop.")
        while True:
            # This is a blocking call to get frames from depthai queue
            img_left, img_right, depth_img, img_center, cam_timestamp, seq_num = camera.get_frames_with_metadata()
            
            # Use fallback time if camera doesn't provide it
            if cam_timestamp is None:
                cam_timestamp = time.monotonic()

            system_boot_epoch = time.time() - time.monotonic()
            sys_timestamp = system_boot_epoch + cam_timestamp
                
            new_history = poller.get_and_clear_history()
            sliding_window.extend(new_history)
            
            closest_joint_state = None
            min_diff = float('inf')
            
            # Find the best match from the full sliding window
            for state in sliding_window:
                time_relative = state['monotonic_timestamp'] - cam_timestamp
                # Update the relative tracking on the queued state itself
                state['time_relative_to_image'] = time_relative
                if abs(time_relative) < min_diff:
                    min_diff = abs(time_relative)
                    closest_joint_state = copy.deepcopy(state)
                    
            if closest_joint_state is not None:
                closest_joint_state['time_relative_to_image'] = closest_joint_state['monotonic_timestamp'] - cam_timestamp
            
            # Since we revert to only sending incremental new history, extract it matching its relative time
            for state in new_history:
                state['time_relative_to_image'] = state['monotonic_timestamp'] - cam_timestamp
                    
            output_dict = {
                'robot_id': robot_id,
                'image_number': seq_num,
                'camera_timestamp': cam_timestamp,
                'system_timestamp': sys_timestamp,
                'joint_state_history': new_history,
                'closest_joint_state': closest_joint_state
            }
            
            if M_right is not None:
                output_dict['camera_matrix'] = M_right
                output_dict['distortion_coefficients'] = D_right
            
            if use_center and not use_gripper:
                if img_center is not None:
                    output_dict['color_image'] = img_center
                    output_dict['depth_image'] = None
                    socket.send_pyobj(output_dict)
            elif use_gripper and use_center:
                if img_right is not None and img_center is not None:
                    if compress:
                        output_dict['color_image_compressed'] = np.array(img_right)
                    else:
                        output_dict['color_image'] = img_right
                    output_dict['depth_image'] = depth_img
                    output_dict['center_color_image'] = img_center
                    socket.send_pyobj(output_dict)
            else:
                if img_right is not None:
                    if compress:
                        output_dict['color_image_compressed'] = np.array(img_right)
                    else:
                        output_dict['color_image'] = img_right
                    output_dict['depth_image'] = depth_img
                    socket.send_pyobj(output_dict)
                
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"Camera Initialization Error: {e}")
    finally:
        poller.stop()
        if camera is not None:
            camera.stop()
        robot.stop()
        print("\nStopped transmitting.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Send Synced Gripper Images and Joint States',
        description='Send images and perfectly synchronized joint state histories via ZMQ.'
    )
    parser.add_argument('-r', '--remote', action='store_true', help='Use this argument when allowing a remote computer to receive images. Configure gripper_networking.py first.')
    parser.add_argument("--device", type=str, default="3.7.3.1", help="Camera device name or USB port (e.g., '3.7.3.1'). Set to empty string to auto-discover.")
    parser.add_argument("--center_device", type=str, default="3.3.1", help="Center camera device name or USB port (e.g., '3.3.1'). Set to empty string to auto-discover.")
    parser.add_argument('-c', '--center', action='store_true', help='Use the center RGB camera instead of the gripper camera.')
    parser.add_argument('-b', '--both', action='store_true', help='Use both the gripper and center RGB cameras.')
    add_camera_args(parser)
    args = parser.parse_args()

    image_size, auto_fps = process_camera_args(args)

    use_remote_computer = args.remote
    use_both = args.both
    use_center = args.center or use_both
    use_gripper = not args.center or use_both

    device_id = args.device if args.device else None
    center_device_id = args.center_device if args.center_device else None

    if use_center and not use_gripper:
        if args.device != "3.4.3.3" and args.center_device == "3.2":
            center_device_id = device_id
            
    main(use_remote_computer, device_id, center_device_id, use_gripper, use_center, image_size, not args.disable_compression, auto_fps, args.oak_buffer_size)
