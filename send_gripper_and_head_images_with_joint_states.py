#!/usr/bin/env python3
import argparse
import os
import time
import threading
import collections
import zmq
import numpy as np
import copy
import depthai as dai

import stretch4_body.robot.robot_client as rc
from stretch4_gripper_modeling_and_control.gripper_camera import (
    get_device_port_by_product_name,
    add_camera_args,
    process_camera_args
)
from stretch4_gripper_modeling_and_control import gripper_networking as gn


class RobotStatePoller:
    """
    Background poller thread querying stretch4_body joint states at high rate (~500Hz).
    """
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


class WristCameraPipeline:
    """
    Dedicated low-latency pipeline for the wrist-mounted Luxonis OAK-D Short Range camera.
    Captures ONLY the requested single camera (left: CAM_B or right: CAM_C) to eliminate
    unnecessary sensor capture, USB transfer, and ISP overhead.
    """
    def __init__(self, camera_name='right', device_id=None, fps=30, image_size=(640, 400), compress=True, oak_buffer_size=1):
        self.camera_name = camera_name.lower()
        if self.camera_name not in ['left', 'right']:
            raise ValueError(f"Unsupported wrist camera side: {self.camera_name}. Must be 'left' or 'right'.")
            
        self.board_socket = dai.CameraBoardSocket.CAM_B if self.camera_name == 'left' else dai.CameraBoardSocket.CAM_C
        self.fps = fps
        self.image_size = image_size
        self.compress = compress
        self.oak_buffer_size = oak_buffer_size
        
        self.device_id = device_id
        if not self.device_id:
            self.device_id = get_device_port_by_product_name("OAK-D-SR", force_search=False)
            
        try:
            self.device = dai.Device(maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS, nameOrDeviceId=self.device_id)
        except Exception as e:
            print(f"Warning: Initial connection to wrist camera '{self.device_id}' failed ({e}). Performing active search...")
            self.device_id = get_device_port_by_product_name("OAK-D-SR", force_search=True)
            self.device = dai.Device(maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS, nameOrDeviceId=self.device_id)
            
        self.pipeline = dai.Pipeline(defaultDevice=self.device)
        self.pipeline.setXLinkChunkSize(0)
        
        cam_node = self.pipeline.create(dai.node.Camera)
        cam_node.setSensorType(dai.CameraSensorType.COLOR)
        cam_node.build(boardSocket=self.board_socket, sensorFps=self.fps)
        cam_node.setNumFramesPools(isp=self.oak_buffer_size + 1, raw=self.oak_buffer_size + 1, imgmanip=self.oak_buffer_size + 1)
        
        out_node = cam_node.requestOutput(
            size=self.image_size,
            type=dai.ImgFrame.Type.NV12,
            resizeMode=dai.ImgResizeMode.CROP,
            enableUndistortion=False,
        )
        
        if self.compress:
            videoEnc = self.pipeline.create(dai.node.VideoEncoder)
            videoEnc.setDefaultProfilePreset(self.fps, dai.VideoEncoderProperties.Profile.MJPEG)
            videoEnc.setQuality(80)
            videoEnc.setNumFramesPool(self.oak_buffer_size + 1)
            out_node.link(videoEnc.input)
            self.q_camera = videoEnc.bitstream.createOutputQueue(maxSize=self.oak_buffer_size, blocking=False)
        else:
            self.q_camera = out_node.createOutputQueue(maxSize=self.oak_buffer_size, blocking=False)

    def start(self):
        self.pipeline.start()

    def stop(self):
        try:
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                self.pipeline.stop()
        except Exception:
            pass
        if self.device is not None:
            self.device.close()

    def get_intrinsics(self):
        """Returns camera matrix M and distortion coefficients D from factory calibration."""
        if self.device is not None:
            try:
                calib = self.device.readCalibration()
                M = np.array(calib.getCameraIntrinsics(self.board_socket, self.image_size[0], self.image_size[1]), dtype=np.float64)
                D = np.array(calib.getDistortionCoefficients(self.board_socket), dtype=np.float64)
                return M, D
            except Exception as e:
                print(f"Warning: could not read wrist camera factory calibration: {e}")
        return None, None

    def get_frame(self):
        """
        Blocks until a frame is received from the wrist camera.
        Returns: (frame_data_or_cv, timestamp, seq_num)
        """
        msg = self.q_camera.get()
        if msg is None:
            return None, None, None
        
        if self.compress:
            img = msg.getData()
        else:
            img = msg.getCvFrame()
            
        timestamp = msg.getTimestamp().total_seconds()
        seq_num = msg.getSequenceNum()
        return img, timestamp, seq_num


class HeadCameraPipeline:
    """
    Dedicated low-latency pipeline for the head-mounted Luxonis OAK-FFC 3P board.
    Captures ONLY the requested single fisheye camera (left: CAM_C or right: CAM_B) to eliminate
    unnecessary sensor capture, USB transfer, and ISP overhead.
    Maintains a high-frequency background ring buffer to allow closest-in-time synchronization.
    """
    def __init__(self, camera_name='left', device_id=None, fps=30, resolution_height=800, compress=True, oak_buffer_size=1):
        self.camera_name = camera_name.lower()
        if self.camera_name not in ['left', 'right']:
            raise ValueError(f"Unsupported head camera side: {self.camera_name}. Must be 'left' or 'right'.")
            
        # Left fisheye camera is connected to CAM_C; Right fisheye camera is connected to CAM_B
        self.board_socket = dai.CameraBoardSocket.CAM_C if self.camera_name == 'left' else dai.CameraBoardSocket.CAM_B
        self.model_name = "head_left" if self.camera_name == 'left' else "head_right"
        self.fps = fps
        self.resolution_height = resolution_height
        self.compress = compress
        self.oak_buffer_size = oak_buffer_size
        
        res_map = {
            400: (640, 400),
            600: (960, 600),
            800: (1280, 800),
            1200: (1920, 1200)
        }
        if self.resolution_height not in res_map:
            raise ValueError(f"Invalid head resolution height {self.resolution_height}. Supported: {list(res_map.keys())}")
        self.image_size = res_map[self.resolution_height]
        
        self.device_id = device_id
        if not self.device_id:
            self.device_id = get_device_port_by_product_name("OAK-FFC-3P", force_search=False)
            
        try:
            self.device = dai.Device(maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS, nameOrDeviceId=self.device_id)
        except Exception as e:
            print(f"Warning: Initial connection to head camera '{self.device_id}' failed ({e}). Performing active search...")
            self.device_id = get_device_port_by_product_name("OAK-FFC-3P", force_search=True)
            self.device = dai.Device(maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS, nameOrDeviceId=self.device_id)
            
        self.pipeline = dai.Pipeline(defaultDevice=self.device)
        self.pipeline.setXLinkChunkSize(0)
        
        cam_node = self.pipeline.create(dai.node.Camera)
        cam_node.setSensorType(dai.CameraSensorType.COLOR)
        cam_node.build(boardSocket=self.board_socket, sensorFps=self.fps)
        cam_node.setNumFramesPools(isp=self.oak_buffer_size + 1, raw=self.oak_buffer_size + 1, imgmanip=self.oak_buffer_size + 1)
        
        out_node = cam_node.requestOutput(
            size=self.image_size,
            type=dai.ImgFrame.Type.NV12,
            resizeMode=dai.ImgResizeMode.CROP,
            enableUndistortion=False,
        )
        
        if self.compress:
            videoEnc = self.pipeline.create(dai.node.VideoEncoder)
            videoEnc.setDefaultProfilePreset(self.fps, dai.VideoEncoderProperties.Profile.MJPEG)
            videoEnc.setQuality(80)
            videoEnc.setNumFramesPool(self.oak_buffer_size + 1)
            out_node.link(videoEnc.input)
            self.q_camera = videoEnc.bitstream.createOutputQueue(maxSize=self.oak_buffer_size, blocking=False)
        else:
            self.q_camera = out_node.createOutputQueue(maxSize=self.oak_buffer_size, blocking=False)
            
        self.history_size = max(100, self.fps * 2) # Buffer ~2 seconds worth of frames
        self.history_buffer = collections.deque(maxlen=self.history_size)
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        self.pipeline.start()
        self.running = True
        self.thread = threading.Thread(target=self._poll_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)
        try:
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                self.pipeline.stop()
        except Exception:
            pass
        if self.device is not None:
            self.device.close()

    def _poll_loop(self):
        """Background thread continuously draining non-blocking queue to prevent buffer bloat."""
        while self.running:
            msg = self.q_camera.tryGet()
            if msg is not None:
                if self.compress:
                    img = msg.getData()
                else:
                    img = msg.getCvFrame()
                    
                timestamp = msg.getTimestamp().total_seconds()
                seq_num = msg.getSequenceNum()
                with self.lock:
                    self.history_buffer.append((img, timestamp, seq_num))
            else:
                time.sleep(0.002)

    def get_closest_frame(self, target_timestamp):
        """Finds the frame in the ring buffer with timestamp closest to target_timestamp."""
        with self.lock:
            if not self.history_buffer:
                return None, None, None
            return min(self.history_buffer, key=lambda x: abs(x[1] - target_timestamp))

    def get_intrinsics(self):
        """Returns camera matrix M and distortion coefficients D (falling back to fleet calibration)."""
        M, D = None, None
        if self.device is not None:
            try:
                calib = self.device.readCalibration()
                M = np.array(calib.getCameraIntrinsics(self.board_socket, self.image_size[0], self.image_size[1]), dtype=np.float64)
                D = np.array(calib.getDistortionCoefficients(self.board_socket), dtype=np.float64)
                return M, D
            except Exception:
                pass
                
        try:
            from stretch4_body.subsystem.cameras.models.camera_calibration import RGBCameraCalibration
            from stretch4_body.subsystem.cameras import RGBCameras
            fleet_calib = RGBCameraCalibration.load_calibration_from_fleet_path(
                camera_type=RGBCameras[self.model_name], is_flip_width_and_height=False
            )
            if fleet_calib and fleet_calib.camera_matrix is not None:
                M = np.array(fleet_calib.camera_matrix, dtype=np.float64)
                D = np.array(fleet_calib.distortion_coefficients, dtype=np.float64)
                
                scale_x = self.image_size[0] / fleet_calib.width
                scale_y = self.image_size[1] / fleet_calib.height
                M[0, 0] *= scale_x
                M[1, 1] *= scale_y
                M[0, 2] *= scale_x
                M[1, 2] *= scale_y
                return M, D
        except Exception as ex:
            print(f"Warning: could not load fleet calibration for {self.model_name}: {ex}")
            
        return None, None


def main(use_remote_computer, wrist_device_id, head_device_id, wrist_camera_side, head_camera_side,
         wrist_image_size, head_res_height, compress, wrist_fps, head_fps, oak_buffer_size):
    print("Starting Robot Client...")
    robot = rc.RobotClient()
    robot.startup()
    
    if not robot.is_homed():
        print("WARNING: Robot is not homed. Joint values may be incorrect.")
        
    poller = RobotStatePoller(robot)

    wrist_camera = None
    head_camera = None
    try:
        print(f"Initializing Wrist Camera Pipeline (Camera: {wrist_camera_side}, Size: {wrist_image_size}, FPS: {wrist_fps}, Compress: {compress})...")
        wrist_camera = WristCameraPipeline(
            camera_name=wrist_camera_side,
            device_id=wrist_device_id,
            fps=wrist_fps,
            image_size=wrist_image_size,
            compress=compress,
            oak_buffer_size=oak_buffer_size
        )
        wrist_camera.start()

        print(f"Initializing Head Camera Pipeline (Camera: {head_camera_side}, Height: {head_res_height}p, FPS: {head_fps}, Compress: {compress})...")
        head_camera = HeadCameraPipeline(
            camera_name=head_camera_side,
            device_id=head_device_id,
            fps=head_fps,
            resolution_height=head_res_height,
            compress=compress,
            oak_buffer_size=oak_buffer_size
        )
        head_camera.start()

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

        wrist_M, wrist_D = wrist_camera.get_intrinsics()
        head_M, head_D = head_camera.get_intrinsics()

        # Sliding window for joint states: 1.0 second history (~500 samples at 500Hz)
        sliding_window = collections.deque(maxlen=500)
        robot_id = os.environ.get('HELLO_FLEET_ID')

        print(f"\nBroadcasting synchronized wrist ({wrist_camera_side}) + head ({head_camera_side}) frames and joint states...")
        print("Press Ctrl+C to stop.\n")

        while True:
            # Pacing driven by wrist camera frame arrival
            wrist_img, wrist_timestamp, wrist_seq_num = wrist_camera.get_frame()
            if wrist_img is None:
                continue

            if wrist_timestamp is None:
                wrist_timestamp = time.monotonic()

            system_boot_epoch = time.time() - time.monotonic()
            sys_timestamp = system_boot_epoch + wrist_timestamp

            # Retrieve closest frame from head camera ring buffer
            head_img, head_timestamp, head_seq_num = head_camera.get_closest_frame(wrist_timestamp)

            new_history = poller.get_and_clear_history()
            sliding_window.extend(new_history)

            closest_joint_state = None
            min_diff = float('inf')

            for state in sliding_window:
                time_relative = state['monotonic_timestamp'] - wrist_timestamp
                state['time_relative_to_image'] = time_relative
                if abs(time_relative) < min_diff:
                    min_diff = abs(time_relative)
                    closest_joint_state = copy.deepcopy(state)

            if closest_joint_state is not None:
                closest_joint_state['time_relative_to_image'] = closest_joint_state['monotonic_timestamp'] - wrist_timestamp

            for state in new_history:
                state['time_relative_to_image'] = state['monotonic_timestamp'] - wrist_timestamp

            output_dict = {
                'robot_id': robot_id,
                'wrist_camera_side': wrist_camera_side,
                'head_camera_side': head_camera_side,
                'image_number': wrist_seq_num,
                'camera_timestamp': wrist_timestamp,
                'system_timestamp': sys_timestamp,
                'joint_state_history': new_history,
                'closest_joint_state': closest_joint_state,
            }

            if wrist_M is not None:
                output_dict['wrist_camera_matrix'] = wrist_M
                output_dict['wrist_distortion_coefficients'] = wrist_D
                # Backward compatibility for existing telemetry receivers
                output_dict['camera_matrix'] = wrist_M
                output_dict['distortion_coefficients'] = wrist_D

            if head_M is not None:
                output_dict['head_camera_matrix'] = head_M
                output_dict['head_distortion_coefficients'] = head_D

            # Add wrist image
            if compress:
                output_dict['wrist_color_image_compressed'] = np.array(wrist_img)
                # Backward compatibility key
                output_dict['color_image_compressed'] = output_dict['wrist_color_image_compressed']
            else:
                output_dict['wrist_color_image'] = wrist_img
                # Backward compatibility key
                output_dict['color_image'] = wrist_img

            # Add head image
            if head_img is not None:
                output_dict['head_image_number'] = head_seq_num
                output_dict['head_camera_timestamp'] = head_timestamp
                output_dict['head_sync_offset_ms'] = (head_timestamp - wrist_timestamp) * 1000.0 if head_timestamp else 0.0

                if compress:
                    output_dict['head_color_image_compressed'] = np.array(head_img)
                else:
                    output_dict['head_color_image'] = head_img

            socket.send_pyobj(output_dict)

    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"Pipeline Error: {e}")
    finally:
        poller.stop()
        if wrist_camera is not None:
            wrist_camera.stop()
        if head_camera is not None:
            head_camera.stop()
        robot.stop()
        print("\nStopped transmitting.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Send Gripper and Head Images with Joint States',
        description='Broadcast synchronized wrist camera, head fisheye camera, and joint states via ZMQ for VLA control.'
    )
    parser.add_argument('-r', '--remote', action='store_true', help='Allow a remote computer to receive data. Configure gripper_networking.py first.')
    parser.add_argument('--wrist_camera', choices=['left', 'right'], default='right', help="Wrist OAK-D SR camera side to capture (left: CAM_B, right: CAM_C). Default: 'right'.")
    parser.add_argument('--head_camera', choices=['left', 'right'], default='left', help="Head OAK-FFC 3P fisheye camera to capture (left: CAM_C, right: CAM_B). Default: 'left'.")
    parser.add_argument('--wrist_device', type=str, default=None, help="Device port/ID for the wrist OAK-D SR camera. If None, automatically detected.")
    parser.add_argument('--head_device', type=str, default=None, help="Device port/ID for the head OAK-FFC-3P camera. If None, automatically detected.")
    parser.add_argument('--head_resolution', type=int, choices=[400, 600, 800, 1200], default=800, help="Vertical resolution for the head fisheye camera (default: 800 -> 1280x800).")
    parser.add_argument('--head_fps', type=int, default=30, help="Framerate for the head fisheye camera (default: 30).")
    add_camera_args(parser)
    args = parser.parse_args()

    wrist_image_size, auto_wrist_fps = process_camera_args(args)
    use_remote_computer = args.remote
    compress = not args.disable_compression

    main(
        use_remote_computer=use_remote_computer,
        wrist_device_id=args.wrist_device,
        head_device_id=args.head_device,
        wrist_camera_side=args.wrist_camera,
        head_camera_side=args.head_camera,
        wrist_image_size=wrist_image_size,
        head_res_height=args.head_resolution,
        compress=compress,
        wrist_fps=auto_wrist_fps,
        head_fps=args.head_fps,
        oak_buffer_size=args.oak_buffer_size
    )
