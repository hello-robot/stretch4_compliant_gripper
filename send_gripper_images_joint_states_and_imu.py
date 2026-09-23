#!/usr/bin/env python3
"""
Send synchronized gripper camera images, robot joint states, and 9-axis IMU measurements
from the wrist-mounted Luxonis OAK-D-SR camera via ZeroMQ.

Supports a Dual-Tier Low-Latency Architecture:
- Tier 1 (Port 4410): Fast, ultra-low-latency telemetry stream (IMU + Joint States at ~100 Hz, ~4 ms latency)
  for closed-loop control and instant bump/tap reflexes.
- Tier 2 (Port 4409): Synchronized visual-inertial-kinematic stream (Images + Depth + Synced Histories at ~30 fps)
  with explicit tri-modal temporal alignment and sync_metadata.
"""

import argparse
import os
import time
import threading
import collections
import zmq
import numpy as np
import copy

import stretch4_body.robot.robot_client as rc
from stretch4_gripper_modeling_and_control.gripper_camera import (
    GripperCamera, add_camera_args, process_camera_args, add_imu_args
)
from stretch4_gripper_modeling_and_control import gripper_networking as gn
from stretch4_gripper_modeling_and_control import telemetry_utils as tu


class RobotStatePoller:
    """Continuously polls low-level joint states from the robot client."""
    def __init__(self, robot):
        self.robot = robot
        self.history_buffer = []
        self.latest_state = None
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
            
            curr_ts = gripper_st.get('timestamp_pc', 0.0)
            
            if self.last_ts is not None and curr_ts == self.last_ts:
                time.sleep(0.002)
                continue
                
            self.last_ts = curr_ts
            
            base = st.get('base', {})
            lift_st = st.get('lift', {})
            arm_st = st.get('arm', {})
            
            wrist_yaw = eoa.get('wrist_yaw', {})
            wrist_pitch = eoa.get('wrist_pitch', {})
            wrist_roll = eoa.get('wrist_roll', {})

            mono_ts = time.monotonic()
            sys_ts = time.time()

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
                'timestamp': sys_ts,
                'monotonic_timestamp': mono_ts,
                'state_number': self.state_counter
            }
            
            with self.lock:
                self.history_buffer.append(data)
                self.latest_state = data
                self.state_counter += 1
                
            time.sleep(0.002)
            
    def stop(self):
        self.running = False
        self.thread.join()
        
    def get_and_clear_history(self):
        with self.lock:
            history = list(self.history_buffer)
            self.history_buffer.clear()
            return history

    def get_latest_state(self):
        with self.lock:
            return copy.deepcopy(self.latest_state)


class IMUPoller:
    """
    High-rate daemon thread polling IMU packets from DepthAI without blocking camera capture.
    Immediately dispatches fast telemetry (Tier 1) for low-latency closed-loop control.
    """
    def __init__(self, camera, robot_poller, fast_pub_socket=None, bump_detector=None):
        self.camera = camera
        self.robot_poller = robot_poller
        self.fast_pub_socket = fast_pub_socket
        self.bump_detector = bump_detector if bump_detector is not None else tu.BumpDetector()
        
        self.history_buffer = []
        self.latest_measurement = None
        self.lock = threading.Lock()
        self.running = True
        
        self.thread = threading.Thread(target=self._poll_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def _poll_loop(self):
        while self.running:
            packets = self.camera.get_imu_packets()
            if not packets:
                time.sleep(0.002)
                continue
                
            converted_list = []
            for pkt in packets:
                meas_dict = tu.convert_imu_packet_to_dict(pkt)
                converted_list.append(meas_dict)
                
                # Immediate in-process contact reflex evaluation (< 0.05 ms)
                bump_event = self.bump_detector.update(meas_dict)
                
                # Tier 1: Fast Telemetry Broadcast for low-latency closed-loop control
                if self.fast_pub_socket is not None:
                    latest_joint = self.robot_poller.get_latest_state()
                    fast_msg = {
                        'type': 'telemetry',
                        'timestamp': meas_dict['timestamp'],
                        'system_timestamp': meas_dict['system_timestamp'],
                        'sequence_number': meas_dict['sequence_number'],
                        'imu': meas_dict['gripper_frame'],
                        'imu_accuracy': meas_dict['accuracy'],
                        'joint_state': latest_joint,
                        'bump_detected': (bump_event is not None),
                        'bump_event': bump_event
                    }
                    try:
                        self.fast_pub_socket.send_pyobj(fast_msg)
                    except Exception:
                        pass
                        
            with self.lock:
                self.history_buffer.extend(converted_list)
                if converted_list:
                    self.latest_measurement = converted_list[-1]
                    
    def stop(self):
        self.running = False
        self.thread.join()
        
    def get_and_clear_history(self):
        with self.lock:
            history = list(self.history_buffer)
            self.history_buffer.clear()
            return history

    def get_latest_measurement(self):
        with self.lock:
            return copy.deepcopy(self.latest_measurement)


def main(use_remote_computer, device_id, center_device_id, use_gripper, use_center,
         image_size, compress, auto_fps, oak_buffer_size, imu_accel_sensor,
         imu_rotation_sensor, imu_rate, disable_fast_telemetry):
    
    print("Starting Robot Client...")
    robot = rc.RobotClient()
    robot.startup()
    
    if not robot.is_homed():
        print("WARNING: Robot is not homed. Joint values may be incorrect.")
        
    poller = RobotStatePoller(robot)
    camera = None
    imu_poller = None
    
    try:
        context = zmq.Context()
        
        # 1. Tier 2 Socket: Synchronized Multi-Modal Channel (Images + Depth + Histories)
        socket_sync = context.socket(zmq.PUB)
        socket_sync.setsockopt(zmq.SNDHWM, 1)
        socket_sync.setsockopt(zmq.RCVHWM, 1)
        
        host_bind = '*' if use_remote_computer else '127.0.0.1'
        sync_address = f'tcp://{host_bind}:{gn.gripper_and_joints_port}'
        print(f"Binding Tier 2 (Visual-Inertial-Kinematic) Publisher to {sync_address}")
        socket_sync.bind(sync_address)
        
        # 2. Tier 1 Socket: Fast Low-Latency Telemetry Channel (IMU + Joint States)
        socket_fast = None
        if not disable_fast_telemetry:
            socket_fast = context.socket(zmq.PUB)
            socket_fast.setsockopt(zmq.SNDHWM, 1)
            socket_fast.setsockopt(zmq.RCVHWM, 1)
            fast_address = f'tcp://{host_bind}:{gn.gripper_telemetry_port}'
            print(f"Binding Tier 1 (Low-Latency Telemetry) Publisher to {fast_address}")
            socket_fast.bind(fast_address)
            
        gn.print_network_info()

        print(f"Initializing DepthAI pipeline with IMU:")
        print(f"  Gripper: {use_gripper}, Center: {use_center}, Size: {image_size}, FPS: {auto_fps}, Compress: {compress}")
        print(f"  IMU Accel: {imu_accel_sensor}, Rotation: {imu_rotation_sensor}, Rate: {imu_rate} Hz (batch threshold = 1)")
        
        camera = GripperCamera(
            device_id=device_id,
            center_device_id=center_device_id,
            fps=auto_fps,
            image_size=image_size,
            use_gripper=use_gripper,
            use_center=use_center,
            compress=compress,
            oak_buffer_size=oak_buffer_size,
            use_imu=use_gripper,
            imu_accel_sensor=imu_accel_sensor,
            imu_rotation_sensor=imu_rotation_sensor,
            imu_rate=imu_rate
        )
        camera.start()
        
        bump_detector = tu.BumpDetector(threshold=2.5, max_angle_deg=45.0, cooldown_seconds=0.3)
        imu_poller = IMUPoller(camera, poller, fast_pub_socket=socket_fast, bump_detector=bump_detector)
        
        M_right, D_right = None, None
        if use_gripper:
            M_right, D_right = camera.get_gripper_intrinsics()
            
        # Sliding windows covering camera latency jitter
        joint_sliding_window = collections.deque(maxlen=500)
        imu_sliding_window = collections.deque(maxlen=500)
        
        robot_id = os.environ.get('HELLO_FLEET_ID')
        print("\nStreaming synchronized imagery, joint states, and IMU measurements... Press Ctrl+C to stop.")
        
        while True:
            # Blocking call to receive synchronized camera frames
            img_left, img_right, depth_img, img_center, cam_timestamp, seq_num = camera.get_frames_with_metadata()
            
            if cam_timestamp is None:
                cam_timestamp = time.monotonic()

            system_boot_epoch = time.time() - time.monotonic()
            sys_timestamp = system_boot_epoch + cam_timestamp
            
            # Ingest new joint states and IMU measurements
            new_joint_history = poller.get_and_clear_history()
            joint_sliding_window.extend(new_joint_history)
            
            new_imu_history = imu_poller.get_and_clear_history()
            imu_sliding_window.extend(new_imu_history)
            
            # Perform tri-modal cross-alignment
            closest_joint, closest_imu, sync_metadata = tu.cross_align_streams(
                joint_sliding_window, imu_sliding_window, cam_timestamp, image_seq=seq_num
            )
            
            output_dict = {
                'robot_id': robot_id,
                'image_number': seq_num,
                'camera_timestamp': cam_timestamp,
                'system_timestamp': sys_timestamp,
                'joint_state_history': new_joint_history,
                'closest_joint_state': closest_joint,
                'imu_history': new_imu_history,
                'closest_imu_measurement': closest_imu,
                'sync_metadata': sync_metadata,
                'imu_frame_description': {
                    'gripper_frame': {
                        'description': 'Right-handed Cartesian coordinate system (FLU) aligned with robot gripper and OAK-D-SR body',
                        'x_axis': 'Normal to front surface of OAK-D-SR (forward direction of robot gripper)',
                        'y_axis': 'Normal to left surface of OAK-D-SR (left direction)',
                        'z_axis': 'Normal to top surface of OAK-D-SR (up direction)',
                    },
                    'camera_frame': {
                        'description': 'DepthAI native camera optical frame (RDF)',
                        'x_axis': 'Right (+X)',
                        'y_axis': 'Down (+Y)',
                        'z_axis': 'Forward (+Z, optical axis)',
                    },
                    'rotation_camera_to_gripper': tu.R_CAM_TO_GRIPPER.tolist(),
                }
            }
            
            if M_right is not None:
                output_dict['camera_matrix'] = M_right
                output_dict['distortion_coefficients'] = D_right
                
            if use_center and not use_gripper:
                if img_center is not None:
                    output_dict['color_image'] = img_center
                    output_dict['depth_image'] = None
                    socket_sync.send_pyobj(output_dict)
            elif use_gripper and use_center:
                if img_right is not None and img_center is not None:
                    if compress:
                        output_dict['color_image_compressed'] = np.array(img_right)
                    else:
                        output_dict['color_image'] = img_right
                    output_dict['depth_image'] = depth_img
                    output_dict['center_color_image'] = img_center
                    socket_sync.send_pyobj(output_dict)
            else:
                if img_right is not None:
                    if compress:
                        output_dict['color_image_compressed'] = np.array(img_right)
                    else:
                        output_dict['color_image'] = img_right
                    output_dict['depth_image'] = depth_img
                    socket_sync.send_pyobj(output_dict)
                    
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"Runtime / Hardware Error: {e}")
    finally:
        print("\nShutting down sender...")
        if imu_poller is not None:
            imu_poller.stop()
        poller.stop()
        if camera is not None:
            camera.stop()
        robot.stop()
        print("Stopped transmitting.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Send Synced Gripper Images, Joint States, and IMU',
        description='Send images, joint state histories, and high-rate 9-axis IMU measurements via ZMQ with low-latency support.'
    )
    parser.add_argument('-r', '--remote', action='store_true',
                        help='Use this argument when allowing a remote computer to receive streams. Configure gripper_networking.py first.')
    parser.add_argument("--device", type=str, default=None,
                        help="Camera device name or USB port (e.g., '3.4.3.1'). If not specified, automatically searches and uses cached device info.")
    parser.add_argument("--center_device", type=str, default=None,
                        help="Center camera device name or USB port (e.g., '3.1'). If not specified, automatically searches and uses cached device info.")
    parser.add_argument('-c', '--center', action='store_true',
                        help='Use the center RGB camera instead of the gripper camera.')
    parser.add_argument('-b', '--both', action='store_true',
                        help='Use both the gripper and center RGB cameras.')
    parser.add_argument('--disable_fast_telemetry', action='store_true',
                        help='Disable Tier 1 dedicated low-latency telemetry channel (Port 4410).')

    add_camera_args(parser)
    add_imu_args(parser)
    args = parser.parse_args()

    image_size, auto_fps = process_camera_args(args)

    use_remote_computer = args.remote
    use_both = args.both
    use_center = args.center or use_both
    use_gripper = not args.center or use_both

    device_id = args.device if args.device else None
    center_device_id = args.center_device if args.center_device else None

    if use_center and not use_gripper:
        if args.device and not args.center_device:
            center_device_id = device_id

    main(
        use_remote_computer=use_remote_computer,
        device_id=device_id,
        center_device_id=center_device_id,
        use_gripper=use_gripper,
        use_center=use_center,
        image_size=image_size,
        compress=not args.disable_compression,
        auto_fps=auto_fps,
        oak_buffer_size=args.oak_buffer_size,
        imu_accel_sensor=args.imu_accel_sensor,
        imu_rotation_sensor=args.imu_rotation_sensor,
        imu_rate=args.imu_rate,
        disable_fast_telemetry=args.disable_fast_telemetry
    )
