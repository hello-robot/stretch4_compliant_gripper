import os
import json
import time
import tempfile
import depthai as dai
import numpy as np
import datetime

CACHE_FILE_PATH = os.path.join(tempfile.gettempdir(), f"luxonis_device_cache_{os.getuid()}.json")
CACHE_EXPIRY_SECONDS = 24 * 60 * 60  # Invalidate after 24 hours

def _load_cache():
    try:
        if os.path.exists(CACHE_FILE_PATH):
            with open(CACHE_FILE_PATH, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load device cache: {e}")
    return {}

def _save_cache(cache_data):
    try:
        with open(CACHE_FILE_PATH, 'w') as f:
            json.dump(cache_data, f)
    except Exception as e:
        print(f"Warning: Failed to save device cache: {e}")

def clear_device_cache():
    try:
        if os.path.exists(CACHE_FILE_PATH):
            os.remove(CACHE_FILE_PATH)
    except Exception as e:
        print(f"Warning: Failed to clear device cache: {e}")

def get_device_port_by_product_name(product_name: str, force_search: bool = False):
    """
    Finds the USB port or device ID for a Luxonis camera matching product_name (e.g., 'OAK-D-SR', 'OAK-FFC-3P').
    First checks the local cache if force_search is False.
    If not found or cached lookup fails, actively queries connected devices and updates the cache.
    """
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        raise RuntimeError("No DepthAI devices found connected over USB.")

    cache_data = _load_cache()
    current_time = time.time()

    # 1. Check cache first if not forcing search
    if not force_search:
        for info in devices:
            cached_device = cache_data.get(info.deviceId)
            if cached_device:
                if current_time - cached_device.get("timestamp", 0) <= CACHE_EXPIRY_SECONDS:
                    if cached_device.get("product_name") == product_name:
                        return info.name

    # 2. Actively search connected devices
    for info in devices:
        if not force_search:
            cached_device = cache_data.get(info.deviceId)
            if cached_device and current_time - cached_device.get("timestamp", 0) <= CACHE_EXPIRY_SECONDS:
                continue

        try:
            with dai.Device(maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS, nameOrDeviceId=info.deviceId) as device:
                actual_product_name = device.getProductName()
                cache_data[info.deviceId] = {
                    "product_name": actual_product_name,
                    "timestamp": current_time,
                    "name": info.name
                }
                _save_cache(cache_data)

                if product_name == actual_product_name:
                    device.close()
                    time.sleep(1) # Sleep needed so device returns to ready state
                    return info.name
        except Exception as e:
            print(f"Warning: Could not query device {info.deviceId} ({info.name}): {e}")

    # If active search with cache-skip didn't find the product, do a full forced query before giving up
    if not force_search:
        return get_device_port_by_product_name(product_name, force_search=True)

    raise RuntimeError(f"Could not find any connected Luxonis device with product name '{product_name}'. Connected devices: {[d.name for d in devices]}")

def pixel_from_3d(xyz, camera_info):
    x_in, y_in, z_in = xyz
    camera_matrix = camera_info['camera_matrix']
    f_x = camera_matrix[0,0]
    c_x = camera_matrix[0,2]
    f_y = camera_matrix[1,1]
    c_y = camera_matrix[1,2]
    x_pix = ((f_x * x_in) / z_in) + c_x
    y_pix = ((f_y * y_in) / z_in) + c_y
    xy = np.array([x_pix, y_pix])
    return(xy)

def pixel_to_3d(xy_pix, z_in, camera_info):
    x_pix, y_pix = xy_pix
    camera_matrix = camera_info['camera_matrix']
    f_x = camera_matrix[0,0]
    c_x = camera_matrix[0,2]
    f_y = camera_matrix[1,1]
    c_y = camera_matrix[1,2]
    x_out = ((x_pix - c_x) * z_in) / f_x
    y_out = ((y_pix - c_y) * z_in) / f_y
    xyz_out = np.array([x_out, y_out, z_in])
    return(xyz_out)

class GripperCamera:
    """
    A minimal wrapper for the Stretch 4 Gripper Camera (OAK-D SR).
    Optionally initializes the center camera pipeline in addition.
    Yields left and right NV12 frames converted to BGR, along with depth and center frames.
    """
    def __init__(self, device_id=None, center_device_id=None, fps=30, image_size=(640, 400), use_gripper=True, use_center=False, compress=True, oak_buffer_size=1):
        self.device_id = device_id
        self.center_device_id = center_device_id
        self.fps = fps
        self.image_size = image_size
        self.compress = compress
        self.oak_buffer_size = oak_buffer_size
        
        if self.image_size == (640, 480):
            raise ValueError("The 640x480 resolution option is no longer supported because its cropped aspect ratio does not match the raw sensor images, creating complexities for factory camera calibration.")
        
        self.use_gripper = use_gripper
        self.use_center = use_center
        
        # Backward compatibility fallback
        if not self.use_gripper and not self.use_center:
            self.use_gripper = True
            
        self.gripper_pipeline = None
        self.center_pipeline = None
        
        self.q_sync = None
        self.q_center = None
        
        self.gripper_device = None
        self.center_device = None
        
        # 1. Gripper Camera Pipeline
        if self.use_gripper:
            gripper_port = self.device_id
            if not gripper_port:
                gripper_port = get_device_port_by_product_name("OAK-D-SR", force_search=False)

            try:
                self.gripper_device = dai.Device(maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS, nameOrDeviceId=gripper_port)
            except Exception as e:
                print(f"Warning: Initial connection to gripper camera '{gripper_port}' failed ({e}). Performing active device search...")
                gripper_port = get_device_port_by_product_name("OAK-D-SR", force_search=True)
                self.gripper_device = dai.Device(maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS, nameOrDeviceId=gripper_port)

            self.device_id = gripper_port
            self.gripper_pipeline = dai.Pipeline(defaultDevice=self.gripper_device)
            self.gripper_pipeline.setXLinkChunkSize(0)

            cam_left = self.gripper_pipeline.create(dai.node.Camera)
            cam_left.setSensorType(dai.CameraSensorType.COLOR)
            cam_left.build(boardSocket=dai.CameraBoardSocket.CAM_B, sensorFps=self.fps)
            # Internal left structure for Depth computation only - DO NOT request out over USB
            left_raw = cam_left.requestOutput(size=self.image_size, type=dai.ImgFrame.Type.NV12)

            cam_right = self.gripper_pipeline.create(dai.node.Camera)
            cam_right.setSensorType(dai.CameraSensorType.COLOR)
            cam_right.build(boardSocket=dai.CameraBoardSocket.CAM_C, sensorFps=self.fps)
            right_raw = cam_right.requestOutput(size=self.image_size, type=dai.ImgFrame.Type.NV12)
            
            if self.compress:
                videoEnc = self.gripper_pipeline.create(dai.node.VideoEncoder)
                videoEnc.setDefaultProfilePreset(self.fps, dai.VideoEncoderProperties.Profile.MJPEG)
                videoEnc.setQuality(80)
                
                # We still need the CROP resized shape to encode properly
                out_right = cam_right.requestOutput(
                    size=self.image_size,
                    type=dai.ImgFrame.Type.NV12,
                    resizeMode=dai.ImgResizeMode.CROP,
                    enableUndistortion=False,
                )
                out_right.link(videoEnc.input)
            else:
                out_right = cam_right.requestOutput(
                    size=self.image_size,
                    type=dai.ImgFrame.Type.NV12,
                    resizeMode=dai.ImgResizeMode.CROP,
                    enableUndistortion=False,
                )

            stereo = self.gripper_pipeline.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.ROBOTICS)
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_C)
            stereo.initialConfig.postProcessing.thresholdFilter.maxRange = 10000 
            
            left_raw.link(stereo.left)
            right_raw.link(stereo.right)

            sync = self.gripper_pipeline.create(dai.node.Sync)
            sync.setSyncThreshold(datetime.timedelta(milliseconds=15))

            if self.compress:
                videoEnc.bitstream.link(sync.inputs["right"])
            else:
                out_right.link(sync.inputs["right"])
                
            stereo.depth.link(sync.inputs["depth"])

            self.q_sync = sync.out.createOutputQueue(maxSize=self.oak_buffer_size, blocking=False)

        # 2. Center Camera Pipeline
        if self.use_center:
            center_port = self.center_device_id
            if not center_port:
                center_port = get_device_port_by_product_name("OAK-FFC-3P", force_search=False)

            try:
                self.center_device = dai.Device(maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS, nameOrDeviceId=center_port)
            except Exception as e:
                print(f"Warning: Initial connection to center camera '{center_port}' failed ({e}). Performing active device search...")
                center_port = get_device_port_by_product_name("OAK-FFC-3P", force_search=True)
                self.center_device = dai.Device(maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS, nameOrDeviceId=center_port)

            self.center_device_id = center_port
            self.center_pipeline = dai.Pipeline(defaultDevice=self.center_device)
            self.center_pipeline.setXLinkChunkSize(0)
                
            cam_center = self.center_pipeline.create(dai.node.Camera)
            cam_center.setSensorType(dai.CameraSensorType.COLOR)
            cam_center.build(boardSocket=dai.CameraBoardSocket.CAM_A, sensorFps=5)
            
            # Request 12MP for the center camera
            out_center = cam_center.requestOutput(
                size=(4056, 3040),
                fps=5,
                type=dai.ImgFrame.Type.NV12,
                resizeMode=dai.ImgResizeMode.CROP,
                enableUndistortion=False,
            )
            self.q_center = out_center.createOutputQueue(maxSize=self.oak_buffer_size, blocking=False)


    def start(self):
        """Starts the active pipelines."""
        if self.gripper_pipeline:
            self.gripper_pipeline.start()
        if self.center_pipeline:
            self.center_pipeline.start()

    def stop(self):
        """Stops the active pipelines."""
        if self.gripper_pipeline:
            self.gripper_pipeline.stop()
        if self.center_pipeline:
            self.center_pipeline.stop()

    def get_gripper_intrinsics(self):
        """Returns M_right and D_right from the hardware factory calibration if available."""
        if self.gripper_device is not None:
            try:
                calib = self.gripper_device.readCalibration()
                M_right = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, self.image_size[0], self.image_size[1]), dtype=np.float64)
                D_right = np.array(calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C), dtype=np.float64)
                return M_right, D_right
            except Exception as e:
                print(f"Warning: could not read calibration from OAK-D: {e}")
        return None, None

    def get_frames(self):
        """Blocks until frames are received from active cameras, then returns them."""
        img_left, img_right, depth_img, img_center, _, _ = self.get_frames_with_metadata()
        # Fallback return without center camera for backwards compatibility if needed, but signature changed
        return img_left, img_right, depth_img, img_center
        
    def get_frames_with_metadata(self):
        """Blocks until frames are received from active cameras, then returns (img_left, img_right, depth_img, img_center, timestamp, sequence_num)."""
        img_left, img_right, depth_img, img_center = None, None, None, None
        timestamp, sequence_num = None, None
        
        if self.use_gripper:
            msgGroup = self.q_sync.get()
            
            if msgGroup is not None:
                msgNames = msgGroup.getMessageNames()
                frame_right_msg = msgGroup["right"] if "right" in msgNames else None
                frame_depth_msg = msgGroup["depth"] if "depth" in msgNames else None
                
                if frame_right_msg and frame_depth_msg:
                    # We skip img_left entirely to save massive latency bandwidth
                    
                    if self.compress:
                        # Pass back raw MJPEG bytes instead of Mat object to ZMQ directly
                        img_right = frame_right_msg.getData()
                    else:
                        img_right = frame_right_msg.getCvFrame()
                        
                    depth_img = frame_depth_msg.getFrame()
                    
                    timestamp = frame_right_msg.getTimestamp().total_seconds()
                    sequence_num = frame_right_msg.getSequenceNum()
                
        if self.use_center:
            frame_center_msg = self.q_center.get()
            if frame_center_msg:
                img_center = frame_center_msg.getCvFrame()
                
                # Center cameras are sometimes mounted rotated depending on hardware, uncomment rotation if needed:
                img_center = np.rot90(img_center, k=-1)
                
                if not self.use_gripper:
                    timestamp = frame_center_msg.getTimestamp().total_seconds()
                    sequence_num = frame_center_msg.getSequenceNum()
                    
        return img_left, img_right, depth_img, img_center, timestamp, sequence_num

def get_valid_combinations_text():
    return """
Valid Combinations of Resolution, Compression, and FPS:
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

def add_camera_args(parser):
    import argparse
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.add_argument('--resolution', type=int, default=500, help='Vertical resolution of the gripper camera image. Options: 400 (640x400), 500 (800x500), 600 (960x600), 640 (1024x640), 800 (1280x800).\n' + get_valid_combinations_text())
    parser.add_argument('--disable_compression', action='store_true', help='Disable the use of MJPEG compression applied by the gripper camera to RGB images prior to transmission over USB. This will typically result in a lower frame rate.')
    parser.add_argument('--oak_buffer_size', type=int, default=1, help='Size of the OAK-D internal queue. Default 1 to minimize latency. Increase this value for applications that would benefit from fewer dropped frames at the cost of higher latency during USB hiccups.')

def process_camera_args(args):
    import sys
    res_map = {
        400: (640, 400),
        500: (800, 500),
        600: (960, 600),
        640: (1024, 640),
        800: (1280, 800)
    }
    if args.resolution not in res_map:
        if args.resolution == 480:
            print("Error: The 640x480 (480) resolution option is no longer supported because its cropped aspect ratio does not match the raw sensor images, creating complexities for factory camera calibration.")
            sys.exit(1)
            
        print(f"Error: Invalid resolution '{args.resolution}'. Available vertical resolution options are:")
        for k, v in res_map.items():
            print(f"  {k} -> {v[0]}x{v[1]}")
        sys.exit(1)
        
    image_size = res_map[args.resolution]
    
    # Auto FPS limits guaranteeing low latency on USB 2.0 (under ~30 MB/s limit)
    auto_fps = 30
    if args.disable_compression:
        if args.resolution >= 800: auto_fps = 5
        elif args.resolution >= 600: auto_fps = 10
        elif args.resolution >= 500: auto_fps = 15
        else: auto_fps = 30
    else:
        if args.resolution >= 800: auto_fps = 10
        elif args.resolution >= 640: auto_fps = 15
        elif args.resolution >= 600: auto_fps = 20
        else: auto_fps = 30
        
    return image_size, auto_fps
