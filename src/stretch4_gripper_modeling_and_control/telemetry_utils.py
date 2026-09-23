import collections
import cv2
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

# Coordinate transformation: Camera frame (RDF) -> Gripper frame (FLU)
# Camera: +X=Right, +Y=Down, +Z=Forward (optical axis)
# Gripper: +X=Forward, +Y=Left, +Z=Up
# v_grip = R_CAM_TO_GRIPPER @ v_cam
R_CAM_TO_GRIPPER = np.array([
    [ 0.0,  0.0,  1.0],
    [-1.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0]
], dtype=np.float64)

class JointStateHistory:
    """
    A helper class for receivers to reassemble and work with joint state histories
    transmitted over ZMQ.
    """
    def __init__(self, maxlen=400, warn_on_discontinuity=True):
        # 400 states is roughly 4 seconds at 100Hz
        self.history = collections.deque(maxlen=maxlen)
        self.last_state_number = None
        self.warn_on_discontinuity = warn_on_discontinuity

    def add_states(self, states_list):
        """
        Adds a list of joint state dictionaries to the history buffer.
        Validates state_number sequence if available to check for gaps.
        """
        for state in states_list:
            state_num = state.get('state_number')
            if state_num is not None:
                if self.last_state_number is not None:
                    # Duplicate check
                    if state_num <= self.last_state_number:
                        continue
                    # Gap check
                    if state_num > self.last_state_number + 1:
                        if self.warn_on_discontinuity:
                            print(f"Warning: Discontinuity detected in joint states. Expected state {self.last_state_number + 1}, got {state_num}.")
                self.last_state_number = state_num
            self.history.append(state)

    def get_closest_state(self, target_timestamp, timestamp_key='monotonic_timestamp'):
        """
        Linear search to find the state whose timestamp is closest to target_timestamp.
        """
        if not self.history:
            return None
            
        min_diff = float('inf')
        closest = None
        
        for state in self.history:
            ts = state.get(timestamp_key)
            if ts is not None:
                diff = abs(ts - target_timestamp)
                if diff < min_diff:
                    min_diff = diff
                    closest = state
                    
        return closest

    def get_window(self, start_time, end_time, timestamp_key='monotonic_timestamp'):
        """
        Returns all states that fall within the [start_time, end_time] interval.
        """
        return [s for s in self.history if s.get(timestamp_key) is not None and start_time <= s[timestamp_key] <= end_time]

    def get_history_list(self):
        """Returns the current assembled history as a standard python list."""
        return list(self.history)


def draw_history_graphs(history, width=800, height=200):
    """
    Renders continuous 2D line plots of the gripper's pos_pct and effort
    for visualization of the sliding history window.
    """
    graph_img = np.zeros((height, width, 3), dtype=np.uint8)
    if len(history) < 2:
        return graph_img
        
    times = [s.get('monotonic_timestamp') for s in history]
    # Filter out any states without a monotonic timestamp
    valid_indices = [i for i, t in enumerate(times) if t is not None]
    if len(valid_indices) < 2:
        return graph_img
        
    times = [times[i] for i in valid_indices]
    pos = [history[i]['gripper']['pos_pct'] for i in valid_indices]
    eff = [history[i]['gripper']['effort'] for i in valid_indices]
    
    t0, t1 = times[0], times[-1]
    if t1 == t0: return graph_img
    
    def get_x(t):
        return min(width-1, max(0, int(((t - t0) / (t1 - t0)) * width)))
        
    def draw_curve(values, color, y_min, y_max, y_offset, h_scale, label):
        pts = []
        val_range = y_max - y_min
        if val_range == 0: val_range = 1.0
        
        for i in range(len(values)):
            x = get_x(times[i])
            normalized = (values[i] - y_min) / val_range
            y = min(y_offset + h_scale - 1, max(y_offset, int(y_offset + h_scale - (normalized * h_scale))))
            pts.append((x, y))
            
        pts = np.array(pts, np.int32)
        cv2.polylines(graph_img, [pts], False, color, 1, cv2.LINE_AA)
        
        # Draw background rect for text to make it readable
        text = f"{label} ({values[-1]:.1f})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(graph_img, (5, y_offset + 5), (5 + tw + 10, y_offset + 5 + th + 10), (0,0,0), -1)
        cv2.putText(graph_img, text, (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # pos_pct usually 0 to 100
    pos_min, pos_max = 0.0, 100.0
    if pos:
        pm = max(pos)
        if pm > 100.0: pos_max = pm
    
    draw_curve(pos, (0, 255, 255), pos_min, pos_max, 0, height // 2, "pos_pct")
    
    # effort scale
    eff_min = min(-10, min(eff))
    eff_max = max(10, max(eff))
    draw_curve(eff, (255, 0, 255), eff_min, eff_max, height // 2, height // 2, "effort")
    
    zero_y = int(height//2 + (height//2) - ((0 - eff_min) / (eff_max - eff_min)) * (height//2))
    zero_y = min(height - 1, max(height // 2, zero_y))
    cv2.line(graph_img, (0, zero_y), (width, zero_y), (100, 100, 100), 1)
    
    return graph_img


class IMUHistory:
    """
    A helper class for receivers to reassemble and work with IMU measurement histories
    transmitted over ZMQ.
    """
    def __init__(self, maxlen=400, warn_on_discontinuity=False):
        # 400 samples is 4 seconds at 100Hz
        self.history = collections.deque(maxlen=maxlen)
        self.last_sequence_number = None
        self.warn_on_discontinuity = warn_on_discontinuity

    def add_measurements(self, measurements_list):
        """
        Adds a list of IMU measurement dictionaries to the history buffer.
        Validates sequence_number to check for gaps and duplicates.
        """
        for meas in measurements_list:
            seq_num = meas.get('sequence_number')
            if seq_num is not None:
                if self.last_sequence_number is not None:
                    # Duplicate check
                    if seq_num <= self.last_sequence_number:
                        continue
                    # Gap check
                    if seq_num > self.last_sequence_number + 1 and self.warn_on_discontinuity:
                        print(f"Warning: Discontinuity detected in IMU measurements. Expected {self.last_sequence_number + 1}, got {seq_num}.")
                self.last_sequence_number = seq_num
            self.history.append(meas)

    def get_closest_measurement(self, target_timestamp, timestamp_key='timestamp'):
        """
        Linear search to find the measurement whose timestamp is closest to target_timestamp.
        """
        if not self.history:
            return None
        min_diff = float('inf')
        closest = None
        for meas in self.history:
            ts = meas.get(timestamp_key)
            if ts is not None:
                diff = abs(ts - target_timestamp)
                if diff < min_diff:
                    min_diff = diff
                    closest = meas
        return closest

    def get_window(self, start_time, end_time, timestamp_key='timestamp'):
        """
        Returns all measurements that fall within the [start_time, end_time] interval.
        """
        return [m for m in self.history if m.get(timestamp_key) is not None and start_time <= m[timestamp_key] <= end_time]

    def get_history_list(self):
        """Returns the current assembled history as a standard python list."""
        return list(self.history)


def convert_imu_packet_to_dict(pkt, reference_cam_timestamp=None):
    """
    Converts a DepthAI IMUPacket into a clean, serializable dictionary with
    explicit Gripper Frame (FLU) and Camera Frame (RDF) fields, timestamps,
    and accuracy ratings.
    """
    # Timestamps
    ts = None
    dev_ts = None
    seq = None
    
    # Priority for timestamp: acceleroMeter -> gyroscope -> rotationVector -> magneticField
    for sensor in [pkt.acceleroMeter, pkt.gyroscope, pkt.rotationVector, pkt.magneticField]:
        if sensor is not None:
            ts = sensor.getTimestamp().total_seconds()
            dev_ts = sensor.getTimestampDevice().total_seconds()
            seq = sensor.sequence
            break
            
    if ts is None:
        ts = time.monotonic()
        dev_ts = 0.0
        seq = -1

    # Raw Camera Frame Vectors (RDF: +X=Right, +Y=Down, +Z=Forward)
    a_c = np.array([pkt.acceleroMeter.x, pkt.acceleroMeter.y, pkt.acceleroMeter.z], dtype=float) if pkt.acceleroMeter else None
    g_c = np.array([pkt.gyroscope.x, pkt.gyroscope.y, pkt.gyroscope.z], dtype=float) if pkt.gyroscope else None
    m_c = np.array([pkt.magneticField.x, pkt.magneticField.y, pkt.magneticField.z], dtype=float) if pkt.magneticField else None
    
    # Transform to Gripper Frame (FLU: +X=Forward, +Y=Left, +Z=Up)
    a_g = (R_CAM_TO_GRIPPER @ a_c) if a_c is not None else None
    g_g = (R_CAM_TO_GRIPPER @ g_c) if g_c is not None else None
    m_g = (R_CAM_TO_GRIPPER @ m_c) if m_c is not None else None
    
    # Rotation and derived gravity
    grav_c = None
    grav_g = None
    q_c = None
    q_g = None
    euler_g = None
    rot_accuracy = 0.0
    
    if pkt.rotationVector:
        rot = pkt.rotationVector
        q_c = [rot.i, rot.j, rot.k, rot.real] # i, j, k, real
        rot_accuracy = getattr(rot, 'rotationVectorAccuracy', 0.0)
        try:
            r_c = R.from_quat(q_c)
            # Mathematically compute gravity vector in camera frame from fused rotation:
            # Gravity reaction in world ENU is [0, 0, 9.80665]
            grav_c = r_c.inv().apply([0.0, 0.0, 9.80665])
            grav_g = R_CAM_TO_GRIPPER @ grav_c
            
            # Orientation of gripper frame in world ENU:
            # R_grip = R_cam @ R_CAM_TO_GRIPPER.T
            r_g = R.from_matrix(r_c.as_matrix() @ R_CAM_TO_GRIPPER.T)
            q_g = r_g.as_quat().tolist() # [x, y, z, w]
            euler_g_arr = r_g.as_euler('xyz', degrees=True)
            euler_g = {'roll': float(euler_g_arr[0]), 'pitch': float(euler_g_arr[1]), 'yaw': float(euler_g_arr[2])}
        except Exception:
            pass

    time_rel = (ts - reference_cam_timestamp) if reference_cam_timestamp is not None else None

    return {
        'timestamp': float(ts),
        'device_timestamp': float(dev_ts),
        'system_timestamp': float(time.time()),
        'time_relative_to_image': float(time_rel) if time_rel is not None else None,
        'sequence_number': int(seq),
        'gripper_frame': {
            'linear_acceleration': {'x': float(a_g[0]), 'y': float(a_g[1]), 'z': float(a_g[2])} if a_g is not None else None,
            'gravity': {'x': float(grav_g[0]), 'y': float(grav_g[1]), 'z': float(grav_g[2])} if grav_g is not None else None,
            'gyroscope': {'x': float(g_g[0]), 'y': float(g_g[1]), 'z': float(g_g[2])} if g_g is not None else None,
            'magnetic_field': {'x': float(m_g[0]), 'y': float(m_g[1]), 'z': float(m_g[2])} if m_g is not None else None,
            'rotation': {
                'quaternion_xyzw': q_g,
                'euler_deg': euler_g,
            } if q_g is not None else None,
        },
        'camera_frame': {
            'linear_acceleration': {'x': float(a_c[0]), 'y': float(a_c[1]), 'z': float(a_c[2])} if a_c is not None else None,
            'gravity': {'x': float(gx_c) if 'gx_c' in locals() and gx_c is not None else (float(grav_c[0]) if grav_c is not None else None),
                        'y': float(gy_c) if 'gy_c' in locals() and gy_c is not None else (float(grav_c[1]) if grav_c is not None else None),
                        'z': float(gz_c) if 'gz_c' in locals() and gz_c is not None else (float(grav_c[2]) if grav_c is not None else None)},
            'gyroscope': {'x': float(g_c[0]), 'y': float(g_c[1]), 'z': float(g_c[2])} if g_c is not None else None,
            'magnetic_field': {'x': float(m_c[0]), 'y': float(m_c[1]), 'z': float(m_c[2])} if m_c is not None else None,
            'rotation': {
                'quaternion_ijkr': q_c,
            } if q_c is not None else None,
        },
        'accuracy': {
            'linear_acceleration': int(pkt.acceleroMeter.accuracy) if pkt.acceleroMeter else 0,
            'gyroscope': int(pkt.gyroscope.accuracy) if pkt.gyroscope else 0,
            'magnetic_field': int(pkt.magneticField.accuracy) if pkt.magneticField else 0,
            'rotation_rad': float(rot_accuracy),
        }
    }


def cross_align_streams(joint_history, imu_history, cam_timestamp, image_seq=None):
    """
    Performs bidirectional temporal cross-alignment between joint states,
    IMU measurements, and camera frame timestamp.
    Returns (closest_joint, closest_imu, sync_metadata).
    """
    closest_joint = None
    min_joint_diff = float('inf')
    for state in joint_history:
        ts = state.get('monotonic_timestamp')
        if ts is not None:
            diff = abs(ts - cam_timestamp)
            if diff < min_joint_diff:
                min_joint_diff = diff
                closest_joint = state

    closest_imu = None
    min_imu_diff = float('inf')
    for meas in imu_history:
        ts = meas.get('timestamp')
        if ts is not None:
            diff = abs(ts - cam_timestamp)
            if diff < min_imu_diff:
                min_imu_diff = diff
                closest_imu = meas

    # Cross-align every IMU sample with its closest joint state
    if joint_history:
        for meas in imu_history:
            imu_ts = meas.get('timestamp')
            if imu_ts is not None:
                c_joint = min(joint_history, key=lambda s: abs(s.get('monotonic_timestamp', 0.0) - imu_ts))
                meas['closest_joint_state_number'] = c_joint.get('state_number')
                meas['time_relative_to_joint'] = float(imu_ts - c_joint.get('monotonic_timestamp', 0.0))
                meas['time_relative_to_image'] = float(imu_ts - cam_timestamp)

    # Cross-align every joint state with its closest IMU measurement
    if imu_history:
        for state in joint_history:
            joint_ts = state.get('monotonic_timestamp')
            if joint_ts is not None:
                c_imu = min(imu_history, key=lambda m: abs(m.get('timestamp', 0.0) - joint_ts))
                state['closest_imu_sequence'] = c_imu.get('sequence_number')
                state['time_relative_to_imu'] = float(joint_ts - c_imu.get('timestamp', 0.0))
                state['time_relative_to_image'] = float(joint_ts - cam_timestamp)

    sync_metadata = {
        'clock_domain': 'Linux CLOCK_MONOTONIC (shared across host RobotStatePoller and DepthAI XLink time-sync)',
        'image': {
            'sequence_number': image_seq,
            'timestamp': float(cam_timestamp),
        },
        'closest_joint_state': {
            'state_number': closest_joint.get('state_number') if closest_joint else None,
            'timestamp': float(closest_joint.get('monotonic_timestamp')) if closest_joint else None,
            'offset_to_image_ms': float((closest_joint['monotonic_timestamp'] - cam_timestamp) * 1000.0) if closest_joint else None,
        },
        'closest_imu_measurement': {
            'sequence_number': closest_imu.get('sequence_number') if closest_imu else None,
            'timestamp': float(closest_imu.get('timestamp')) if closest_imu else None,
            'offset_to_image_ms': float((closest_imu['timestamp'] - cam_timestamp) * 1000.0) if closest_imu else None,
        },
        'joint_to_imu_offset_ms': float((closest_joint['monotonic_timestamp'] - closest_imu['timestamp']) * 1000.0) if (closest_joint and closest_imu) else None,
    }

    return closest_joint, closest_imu, sync_metadata


class BumpDetector:
    """
    Detects taps and bumps on an object held by the Stretch 4 gripper.
    In the gripper frame:
    - +X is Forward (normal to the camera front face and gripper forward direction).
    - An impact pushing into the front surface produces an acceleration along -X.
    
    Detection criteria:
    1. ax < -threshold (pointing into the front face).
    2. Alignment: (-ax) / norm(a) >= cos(max_angle_deg) (approximately normal).
    3. Cooldown debounce: suppresses ringing and secondary bounces.
    """
    def __init__(self, threshold=2.5, max_angle_deg=45.0, cooldown_seconds=0.3):
        self.threshold = threshold
        self.max_angle_deg = max_angle_deg
        self.cos_max_angle = float(np.cos(np.radians(max_angle_deg)))
        self.cooldown_seconds = cooldown_seconds
        
        self.last_bump_time = -1.0
        self.last_bump_event = None
        self.bump_events = []
        self.max_events = 50

    def update(self, imu_sample):
        """
        Evaluates a single IMU sample (dict).
        Returns a bump_event dict if a new bump is detected, else None.
        """
        if not imu_sample:
            return None
            
        grip_acc = imu_sample.get('gripper_frame', {}).get('linear_acceleration', None)
        if grip_acc is None:
            return None
            
        ax = grip_acc.get('x', 0.0)
        ay = grip_acc.get('y', 0.0)
        az = grip_acc.get('z', 0.0)
        ts = imu_sample.get('timestamp', 0.0)
        
        # 1. Threshold check: pushing into the camera front face (negative X in gripper frame)
        if ax >= -self.threshold:
            return None
            
        # 2. Alignment check: is the vector approximately normal to the front surface?
        mag = float(np.sqrt(ax * ax + ay * ay + az * az))
        if mag < 1e-4:
            return None
            
        cos_angle = (-ax) / mag
        if cos_angle < self.cos_max_angle:
            return None
            
        # 3. Cooldown / debounce check
        if ts - self.last_bump_time < self.cooldown_seconds:
            return None
            
        self.last_bump_time = ts
        event = {
            'timestamp': float(ts),
            'peak_ax': float(ax),
            'magnitude': float(mag),
            'alignment_cos': float(cos_angle),
            'angle_deg': float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))),
            'ay': float(ay),
            'az': float(az),
            'sequence_number': imu_sample.get('sequence_number', None)
        }
        self.last_bump_event = event
        self.bump_events.append(event)
        if len(self.bump_events) > self.max_events:
            self.bump_events.pop(0)
            
        return event

    def update_history(self, imu_history_list):
        """
        Evaluates an entire list of IMU samples, returning any newly detected bump events.
        """
        new_events = []
        for sample in imu_history_list:
            ev = self.update(sample)
            if ev is not None:
                new_events.append(ev)
        return new_events

    def is_alert_active(self, current_time, alert_duration=0.5):
        """Returns True if a bump occurred within the last alert_duration seconds."""
        return (current_time - self.last_bump_time) < alert_duration


def draw_imu_history_graphs(history, width=800, height=150, threshold=2.5, bump_events=None):
    """
    Renders continuous 2D line plots of linear acceleration in the Gripper frame:
    ax (Forward/Backward normal): Red/Orange
    ay (Left/Right transverse): Green
    az (Up/Down transverse): Blue
    Also renders the tap threshold line at -threshold, zero line, and bump markers.
    """
    graph_img = np.zeros((height, width, 3), dtype=np.uint8)
    if len(history) < 2:
        return graph_img
        
    times = [s.get('timestamp') for s in history]
    valid_indices = [i for i, t in enumerate(times) if t is not None and history[i].get('gripper_frame', {}).get('linear_acceleration') is not None]
    if len(valid_indices) < 2:
        return graph_img
        
    times = [times[i] for i in valid_indices]
    ax = [history[i]['gripper_frame']['linear_acceleration']['x'] for i in valid_indices]
    ay = [history[i]['gripper_frame']['linear_acceleration']['y'] for i in valid_indices]
    az = [history[i]['gripper_frame']['linear_acceleration']['z'] for i in valid_indices]
    
    t0, t1 = times[0], times[-1]
    if t1 == t0:
        return graph_img
        
    def get_x(t):
        return min(width - 1, max(0, int(((t - t0) / (t1 - t0)) * width)))

    # Determine dynamic y-range centered around zero
    all_vals = ax + ay + az + [-threshold * 1.5, threshold * 1.5]
    v_min = min(all_vals)
    v_max = max(all_vals)
    span = max(abs(v_min), abs(v_max))
    y_min, y_max = -span, span
    val_range = y_max - y_min
    if val_range == 0:
        val_range = 1.0

    def val_to_y(v):
        normalized = (v - y_min) / val_range
        return min(height - 1, max(0, int(height - (normalized * height))))

    # Draw Zero reference line
    y_zero = val_to_y(0.0)
    cv2.line(graph_img, (0, y_zero), (width, y_zero), (70, 70, 70), 1)

    # Draw Tap Threshold line at -threshold (red dashed / dotted)
    y_thresh = val_to_y(-threshold)
    for x_dash in range(0, width, 12):
        cv2.line(graph_img, (x_dash, y_thresh), (min(width - 1, x_dash + 6), y_thresh), (0, 0, 180), 1)

    # Draw curves
    def draw_curve(values, color):
        pts = [(get_x(times[i]), val_to_y(values[i])) for i in range(len(values))]
        pts = np.array(pts, np.int32)
        cv2.polylines(graph_img, [pts], False, color, 1, cv2.LINE_AA)

    draw_curve(ay, (0, 200, 0))     # ay: Green (Left)
    draw_curve(az, (255, 120, 0))   # az: Blue (Up)
    draw_curve(ax, (0, 100, 255))   # ax: Orange/Red (Forward/Normal)

    # Draw bump event vertical indicators
    if bump_events:
        for ev in bump_events:
            ev_t = ev.get('timestamp')
            if ev_t is not None and t0 <= ev_t <= t1:
                ev_x = get_x(ev_t)
                cv2.line(graph_img, (ev_x, 0), (ev_x, height), (0, 0, 255), 2)
                cv2.circle(graph_img, (ev_x, val_to_y(ev.get('peak_ax', -threshold))), 4, (0, 255, 255), -1)

    # Legend & Live values
    latest_ax = ax[-1]
    latest_ay = ay[-1]
    latest_az = az[-1]
    legend_text = f"IMU Linear Accel (m/s^2) [FLU] | Fwd ax:{latest_ax:+.2f}  Left ay:{latest_ay:+.2f}  Up az:{latest_az:+.2f}"
    (tw, th), _ = cv2.getTextSize(legend_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(graph_img, (5, 5), (5 + tw + 10, 5 + th + 10), (0, 0, 0), -1)
    cv2.putText(graph_img, legend_text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    thresh_label = f"Tap Thresh (-{threshold:.1f} m/s^2)"
    cv2.putText(graph_img, thresh_label, (width - 180, min(height - 5, max(15, y_thresh - 4))), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1, cv2.LINE_AA)

    return graph_img

