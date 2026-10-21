#!/usr/bin/env python3
import argparse
import os
import sys
import time
import threading
import collections
import zmq
import numpy as np
import copy
import depthai as dai
import yaml

import stretch4_body.robot.robot_client as rc
from stretch4_gripper_modeling_and_control.gripper_camera import get_device_port_by_product_name
from stretch4_gripper_modeling_and_control import gripper_networking as gn
from stretch4_gripper_modeling_and_control import calibration_utils as cu


# ==============================================================================
# CAMERA CONFIGURATION & VALID COMBINATIONS HELPERS
# ==============================================================================

def get_wrist_valid_combinations_text():
    return """
Valid Combinations of Resolution, Compression, and FPS for Wrist Camera:
  Uncompressed:
    - 400 (640x400)   -> 30 fps
    - 500 (800x500)   -> 15 fps
    - 600 (960x600)   -> 10 fps
    - 640 (1024x640)  -> 10 fps
    - 800 (1280x800)  -> 5 fps
  Compressed (MJPEG):
    - 400 (640x400)   -> 30 fps
    - 500 (800x500)   -> 30 fps
    - 600 (960x600)   -> 20 fps
    - 640 (1024x640)  -> 15 fps
    - 800 (1280x800)  -> 10 fps
"""

def get_head_valid_combinations_text():
    return """
Valid Combinations of Resolution, Compression, and FPS for Head Camera:
  Uncompressed:
    - 400 (640x400)   -> 30 fps
    - 600 (960x600)   -> 30 fps
    - 800 (1280x800)  -> 30 fps
    - 1200 (1920x1200)-> 30 fps
  Compressed (MJPEG):
    - 400 (640x400)   -> 30 fps
    - 600 (960x600)   -> 30 fps
    - 800 (1280x800)  -> 30 fps
    - 1200 (1920x1200)-> 30 fps
"""

WRIST_RES_MAP = {
    400: (640, 400),
    500: (800, 500),
    600: (960, 600),
    640: (1024, 640),
    800: (1280, 800)
}

HEAD_RES_MAP = {
    400: (640, 400),
    600: (960, 600),
    800: (1280, 800),
    1200: (1920, 1200)
}

def get_default_wrist_fps(resolution, compress):
    """Auto FPS limits guaranteeing low latency on USB 2.0 (under ~30 MB/s limit)."""
    if not compress:
        if resolution >= 800: return 5
        elif resolution >= 600: return 10
        elif resolution >= 500: return 15
        else: return 30
    else:
        if resolution >= 800: return 10
        elif resolution >= 640: return 15
        elif resolution >= 600: return 20
        else: return 30

def get_default_head_fps(resolution, compress):
    """Head camera runs on USB 3.0 (SUPER_PLUS) bus supporting up to 30 fps across all resolutions."""
    return 30


# ==============================================================================
# ROBOT STATE POLLER
# ==============================================================================

GRIPPER_JOINTS = ('stretch_gripper', 'parallel_gripper')
"""The end-of-arm keys a gripper can report under: SG4 and PG4."""

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
            # Whichever gripper the robot has on: the stretch gripper (SG4) reports
            # `pos_pct`, the parallel jaw gripper (PG4) `pos_mm`. Read off the
            # status rather than the params, because the status is what the robot
            # actually has fitted.
            gripper_joint = next((name for name in GRIPPER_JOINTS if name in eoa), None)
            gripper_st = eoa.get(gripper_joint, {}) if gripper_joint else {}

            # The gripper's timestamp is a good pulse check for new lower-level
            # samples. Without a gripper, wrist_yaw is on the same bus.
            pulse_st = gripper_st or eoa.get('wrist_yaw', {})
            curr_ts = pulse_st.get('timestamp_pc', 0.0)
            
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

            gripper_data = {
                'joint': gripper_joint,
                'effort': gripper_st.get('effort', 0.0)
            }
            if gripper_joint == 'parallel_gripper':
                gripper_data['pos_mm'] = gripper_st.get('pos_mm', 0.0)
            else:
                gripper_data['pos_pct'] = gripper_st.get('pos_pct', 0.0)

            data = {
                'gripper': gripper_data,
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


# ==============================================================================
# CAMERA PIPELINES
# ==============================================================================

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
    Captures ONLY the requested fisheye camera(s) (left: CAM_C, right: CAM_B) to eliminate
    unnecessary sensor capture, USB transfer, and ISP overhead. Both fisheye cameras can be
    captured at once from the single device by passing camera_names=['left', 'right'].
    Maintains a high-frequency background ring buffer per camera to allow closest-in-time synchronization.
    """
    def __init__(self, camera_names=('left',), device_id=None, fps=30, image_size=(1280, 800), compress=True, oak_buffer_size=1):
        if isinstance(camera_names, str):
            camera_names = [camera_names]
        self.camera_names = [name.lower() for name in camera_names]
        for name in self.camera_names:
            if name not in ['left', 'right']:
                raise ValueError(f"Unsupported head camera side: {name}. Must be 'left' or 'right'.")

        # Left fisheye camera is connected to CAM_C; Right fisheye camera is connected to CAM_B
        self.board_sockets = {
            name: dai.CameraBoardSocket.CAM_C if name == 'left' else dai.CameraBoardSocket.CAM_B
            for name in self.camera_names
        }
        self.model_names = {name: f"head_{name}" for name in self.camera_names}
        self.fps = fps
        self.image_size = image_size
        self.compress = compress
        self.oak_buffer_size = oak_buffer_size
        
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

        self.q_cameras = {}
        for name in self.camera_names:
            cam_node = self.pipeline.create(dai.node.Camera)
            cam_node.setSensorType(dai.CameraSensorType.COLOR)
            cam_node.build(boardSocket=self.board_sockets[name], sensorFps=self.fps)
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
                self.q_cameras[name] = videoEnc.bitstream.createOutputQueue(maxSize=self.oak_buffer_size, blocking=False)
            else:
                self.q_cameras[name] = out_node.createOutputQueue(maxSize=self.oak_buffer_size, blocking=False)
            
        self.history_size = max(100, int(self.fps * 2)) # Buffer ~2 seconds worth of frames
        self.history_buffers = {name: collections.deque(maxlen=self.history_size) for name in self.camera_names}
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
        """Background thread continuously draining non-blocking queues to prevent buffer bloat."""
        while self.running:
            got_frame = False
            for name, q_camera in self.q_cameras.items():
                msg = q_camera.tryGet()
                if msg is None:
                    continue
                got_frame = True
                if self.compress:
                    img = msg.getData()
                else:
                    img = msg.getCvFrame()
                    
                timestamp = msg.getTimestamp().total_seconds()
                seq_num = msg.getSequenceNum()
                with self.lock:
                    self.history_buffers[name].append((img, timestamp, seq_num))
            if not got_frame:
                time.sleep(0.002)

    def get_closest_frame(self, camera_name, target_timestamp):
        """Finds the frame in camera_name's ring buffer with timestamp closest to target_timestamp."""
        with self.lock:
            history_buffer = self.history_buffers[camera_name]
            if not history_buffer:
                return None, None, None
            return min(history_buffer, key=lambda x: abs(x[1] - target_timestamp))

    def get_intrinsics(self, camera_name):
        """Returns camera matrix M and distortion coefficients D (falling back to fleet calibration)."""
        board_socket = self.board_sockets[camera_name]
        model_name = self.model_names[camera_name]
        M, D = None, None
        if self.device is not None:
            try:
                calib = self.device.readCalibration()
                M = np.array(calib.getCameraIntrinsics(board_socket, self.image_size[0], self.image_size[1]), dtype=np.float64)
                D = np.array(calib.getDistortionCoefficients(board_socket), dtype=np.float64)
                return M, D
            except Exception:
                pass
                
        try:
            from stretch4_body.subsystem.cameras.models.camera_calibration import RGBCameraCalibration
            from stretch4_body.subsystem.cameras import RGBCameras
            fleet_calib = RGBCameraCalibration.load_calibration_from_fleet_path(
                camera_type=RGBCameras[model_name], is_flip_width_and_height=False
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
            print(f"Warning: could not load fleet calibration for {model_name}: {ex}")
            
        return None, None


# ==============================================================================
# MAIN BROADCAST LOOP
# ==============================================================================

def load_camera_calibration(path):
    """Loads the robot's camera calibration YAML, or returns None if it is unavailable."""
    if path is None:
        print("Warning: no camera calibration file found. Receivers will not get 'camera_calibration'.")
        return None
    try:
        with open(path, 'r') as f:
            calibration = yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: could not load camera calibration '{path}': {e}")
        return None
    print(f"Sending camera calibration from {path} (cameras: {', '.join(calibration)})")
    return calibration


def main(use_remote_computer, wrist_device_id, head_device_id, wrist_camera_side, head_camera_sides,
         wrist_image_size, head_image_size, wrist_compress, head_compress, wrist_fps, head_fps, oak_buffer_size,
         camera_calibration_path=None):
    print("Starting Robot Client...")
    robot = rc.RobotClient()
    robot.startup()
    
    if not robot.is_homed():
        print("WARNING: Robot is not homed. Joint values may be incorrect.")
        
    poller = RobotStatePoller(robot)

    wrist_camera = None
    head_camera = None
    try:
        print(f"Initializing Wrist Camera Pipeline (Camera: {wrist_camera_side}, Size: {wrist_image_size}, FPS: {wrist_fps}, Compress: {wrist_compress})...")
        wrist_camera = WristCameraPipeline(
            camera_name=wrist_camera_side,
            device_id=wrist_device_id,
            fps=wrist_fps,
            image_size=wrist_image_size,
            compress=wrist_compress,
            oak_buffer_size=oak_buffer_size
        )
        wrist_camera.start()

        head_cameras_text = ','.join(head_camera_sides)
        print(f"Initializing Head Camera Pipeline (Camera: {head_cameras_text}, Size: {head_image_size}, FPS: {head_fps}, Compress: {head_compress})...")
        head_camera = HeadCameraPipeline(
            camera_names=head_camera_sides,
            device_id=head_device_id,
            fps=head_fps,
            image_size=head_image_size,
            compress=head_compress,
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
        head_intrinsics = {side: head_camera.get_intrinsics(side) for side in head_camera_sides}
        # The first requested head camera also goes out under the unprefixed
        # head_* keys, so receivers expecting a single head camera keep working.
        primary_head_side = head_camera_sides[0]

        # Sliding window for joint states: 1.0 second history (~500 samples at 500Hz)
        sliding_window = collections.deque(maxlen=500)
        robot_id = os.environ.get('HELLO_FLEET_ID')

        # Sent with every message: receivers conflate to the latest message, so
        # one sent only at startup could be missed.
        camera_calibration = load_camera_calibration(camera_calibration_path)

        print(f"\nBroadcasting synchronized wrist ({wrist_camera_side}) + head ({head_cameras_text}) frames and joint states...")
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

            # Retrieve closest frame from each head camera's ring buffer
            head_frames = {side: head_camera.get_closest_frame(side, wrist_timestamp) for side in head_camera_sides}

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
                'head_camera_side': primary_head_side,
                'head_camera_sides': list(head_camera_sides),
                'image_number': wrist_seq_num,
                'camera_timestamp': wrist_timestamp,
                'system_timestamp': sys_timestamp,
                'joint_state_history': new_history,
                'closest_joint_state': closest_joint_state,
            }

            if camera_calibration is not None:
                output_dict['camera_calibration'] = camera_calibration
                output_dict['camera_calibration_path'] = camera_calibration_path

            if wrist_M is not None:
                output_dict['wrist_camera_matrix'] = wrist_M
                output_dict['wrist_distortion_coefficients'] = wrist_D
                # Backward compatibility for existing telemetry receivers
                output_dict['camera_matrix'] = wrist_M
                output_dict['distortion_coefficients'] = wrist_D

            for side, (head_M, head_D) in head_intrinsics.items():
                if head_M is None:
                    continue
                output_dict[f'head_{side}_camera_matrix'] = head_M
                output_dict[f'head_{side}_distortion_coefficients'] = head_D
                if side == primary_head_side:
                    output_dict['head_camera_matrix'] = head_M
                    output_dict['head_distortion_coefficients'] = head_D

            # Add wrist image
            if wrist_compress:
                output_dict['wrist_color_image_compressed'] = np.array(wrist_img)
                # Backward compatibility key
                output_dict['color_image_compressed'] = output_dict['wrist_color_image_compressed']
            else:
                output_dict['wrist_color_image'] = wrist_img
                # Backward compatibility key
                output_dict['color_image'] = wrist_img

            # Add head images
            for side, (head_img, head_timestamp, head_seq_num) in head_frames.items():
                if head_img is None:
                    continue
                head_fields = {
                    'image_number': head_seq_num,
                    'camera_timestamp': head_timestamp,
                    'sync_offset_ms': (head_timestamp - wrist_timestamp) * 1000.0 if head_timestamp else 0.0,
                }
                if head_compress:
                    head_fields['color_image_compressed'] = np.array(head_img)
                else:
                    head_fields['color_image'] = head_img

                for name, value in head_fields.items():
                    output_dict[f'head_{side}_{name}'] = value
                    if side == primary_head_side:
                        output_dict[f'head_{name}'] = value

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


# ==============================================================================
# CLI ARGUMENT PARSING
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Send Gripper and Head Images with Joint States',
        description='Broadcast synchronized wrist camera, head fisheye camera, and joint states via ZMQ for VLA control.',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('-r', '--remote', action='store_true', help='Allow a remote computer to receive data. Configure gripper_networking.py first.')
    parser.add_argument('--camera_calibration', type=str, default=None,
                        help='Camera calibration YAML to send to receivers under "camera_calibration". Default:\n'
                             '$HELLO_FLEET_PATH/$HELLO_FLEET_ID/calibration_cameras/calibration_rgb_head_camera.yaml')
    parser.add_argument('--oak_buffer_size', type=int, default=1, help='Size of the internal camera queue on both Luxonis devices. Default 1 to minimize latency.')

    # Wrist Camera Options
    wrist_group = parser.add_argument_group('Wrist Camera Options')
    wrist_group.add_argument('--wrist_camera', choices=['left', 'right'], default='right',
                             help="Wrist OAK-D SR camera side to capture (left: CAM_B, right: CAM_C). Default: 'right'.")
    wrist_group.add_argument('--wrist_device', type=str, default=None,
                             help="Device port/ID for the wrist OAK-D SR camera. If None, automatically detected.")
    wrist_group.add_argument('--wrist_resolution', '--resolution', dest='wrist_resolution', type=int, default=500,
                             help='Vertical resolution of the wrist camera image. Options: 400 (640x400), 500 (800x500), 600 (960x600), 640 (1024x640), 800 (1280x800).\n' + get_wrist_valid_combinations_text())
    wrist_group.add_argument('--wrist_fps', '--fps', dest='wrist_fps', type=int, default=None,
                             help='Framerate (FPS) for the wrist camera. If not specified, automatically selected based on resolution and compression.')
    wrist_group.add_argument('--disable_wrist_compression', '--disable_compression', dest='disable_wrist_compression', action='store_true',
                             help='Disable MJPEG compression for the wrist camera. Sends uncompressed NV12 frames.')

    # Head Camera Options
    head_group = parser.add_argument_group('Head Camera Options')
    head_group.add_argument('--head_camera', type=str, default='left',
                            help="Head OAK-FFC 3P fisheye camera(s) to capture (left: CAM_C, right: CAM_B).\n"
                                 "Use 'left,right' to send both. The first one listed is also sent under the\n"
                                 "unprefixed head_* keys for single-camera receivers. Default: 'left'.")
    head_group.add_argument('--head_device', type=str, default=None,
                            help="Device port/ID for the head OAK-FFC-3P camera. If None, automatically detected.")
    head_group.add_argument('--head_resolution', type=int, default=800,
                            help='Vertical resolution of the head fisheye camera. Options: 400 (640x400), 600 (960x600), 800 (1280x800), 1200 (1920x1200).\n' + get_head_valid_combinations_text())
    head_group.add_argument('--head_fps', type=int, default=None,
                            help='Framerate (FPS) for the head camera. If not specified, automatically selected based on resolution and compression (default: 30 fps).')
    head_group.add_argument('--disable_head_compression', action='store_true',
                            help='Disable MJPEG compression for the head camera. Sends uncompressed NV12 frames.')

    args = parser.parse_args()

    # Process Wrist Options
    if args.wrist_resolution not in WRIST_RES_MAP:
        if args.wrist_resolution == 480:
            print("Error: The 640x480 (480) resolution option is not supported because its cropped aspect ratio does not match the raw sensor images.")
            sys.exit(1)
        print(f"Error: Invalid wrist resolution '{args.wrist_resolution}'. Available options are:")
        for k, v in WRIST_RES_MAP.items():
            print(f"  {k} -> {v[0]}x{v[1]}")
        sys.exit(1)
    wrist_image_size = WRIST_RES_MAP[args.wrist_resolution]
    wrist_compress = not args.disable_wrist_compression
    wrist_fps = args.wrist_fps if args.wrist_fps is not None else get_default_wrist_fps(args.wrist_resolution, wrist_compress)

    # Process Head Options
    head_camera_sides = [side.strip().lower() for side in args.head_camera.split(',') if side.strip()]
    if not head_camera_sides or any(side not in ('left', 'right') for side in head_camera_sides) or len(set(head_camera_sides)) != len(head_camera_sides):
        print(f"Error: Invalid head camera '{args.head_camera}'. Use 'left', 'right', 'left,right', or 'right,left'.")
        sys.exit(1)
    if args.head_resolution not in HEAD_RES_MAP:
        print(f"Error: Invalid head resolution '{args.head_resolution}'. Available options are:")
        for k, v in HEAD_RES_MAP.items():
            print(f"  {k} -> {v[0]}x{v[1]}")
        sys.exit(1)
    head_image_size = HEAD_RES_MAP[args.head_resolution]
    head_compress = not args.disable_head_compression
    head_fps = args.head_fps if args.head_fps is not None else get_default_head_fps(args.head_resolution, head_compress)

    main(
        use_remote_computer=args.remote,
        wrist_device_id=args.wrist_device,
        head_device_id=args.head_device,
        wrist_camera_side=args.wrist_camera,
        head_camera_sides=head_camera_sides,
        wrist_image_size=wrist_image_size,
        head_image_size=head_image_size,
        wrist_compress=wrist_compress,
        head_compress=head_compress,
        wrist_fps=wrist_fps,
        head_fps=head_fps,
        oak_buffer_size=args.oak_buffer_size,
        camera_calibration_path=args.camera_calibration or cu.get_default_camera_calibration_path()
    )
