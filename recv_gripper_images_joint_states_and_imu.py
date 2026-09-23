#!/usr/bin/env python3
"""
Receive and display synchronized gripper camera images, robot joint states, and 9-axis IMU measurements
transmitted from the robot via ZeroMQ.

Features:
- Live display of RGB imagery with fingertip kinematics overlay.
- Real-time overlay of tri-modal sync offsets (Camera <-> Joint State <-> IMU).
- Visual tap/bump detection alert banner upon contact normal to the OAK-D-SR front face.
- Stacked real-time history plots:
  - Top plot: Gripper position and effort curves.
  - Bottom plot: IMU linear acceleration curves (ax, ay, az in FLU Gripper frame) with tap threshold and bump markers.
- Optional --low-latency-mode connecting to Tier 1 telemetry stream on Port 4410 for ultra-low-latency closed-loop control reflexes.
"""

import argparse
import collections
import cv2
import zmq
import numpy as np
import time
import os

from stretch4_gripper_modeling_and_control import gripper_networking as gn
from stretch4_gripper_modeling_and_control import visualization_utils as vu
from stretch4_gripper_modeling_and_control import telemetry_utils as tu
from stretch4_gripper_modeling_and_control import calibration_utils as cu
from visualize_fingertip_model import FingertipVisualizer
from stretch4_gripper_modeling_and_control import aruco_to_fingertips as af


def run_low_latency_mode(use_remote_computer, tap_threshold=2.5):
    """
    Direct subscriber to Tier 1 fast telemetry channel (Port 4410).
    Receives ~100 Hz IMU and joint states with ~4 ms latency for closed-loop control.
    """
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1)
    
    host_ip = gn.robot_ip if use_remote_computer else '127.0.0.1'
    address = f'tcp://{host_ip}:{gn.gripper_telemetry_port}'
    print(f"[Low-Latency Mode] Connecting to Tier 1 Telemetry stream at {address}...")
    socket.connect(address)
    
    last_print = time.time()
    count = 0
    detector = tu.BumpDetector(threshold=tap_threshold, max_angle_deg=45.0, cooldown_seconds=0.3)
    
    print("[Low-Latency Mode] Running fast reflex loop... Press Ctrl+C to stop.")
    try:
        while True:
            msg = socket.recv_pyobj()
            count += 1
            now_mono = time.monotonic()
            
            # Latency metric
            ts = msg.get('timestamp', now_mono)
            latency_ms = (now_mono - ts) * 1000.0
            
            # IMU linear acceleration in Gripper frame
            imu_data = msg.get('imu', {})
            lin_acc = imu_data.get('linear_acceleration', {})
            ax = lin_acc.get('x', 0.0) if lin_acc else 0.0
            ay = lin_acc.get('y', 0.0) if lin_acc else 0.0
            az = lin_acc.get('z', 0.0) if lin_acc else 0.0
            
            # Check for bump event
            bump_event = msg.get('bump_event')
            if bump_event is None and lin_acc:
                # Local check if sender did not attach event
                bump_event = detector.update({
                    'timestamp': ts,
                    'sequence_number': msg.get('sequence_number'),
                    'gripper_frame': imu_data
                })
                
            if bump_event:
                peak = bump_event.get('peak_ax', ax)
                mag = bump_event.get('magnitude', 0.0)
                deg = bump_event.get('angle_deg', 0.0)
                print(f"\n>>> [TAP DETECTED!] Peak ax: {peak:.2f} m/s^2 | Mag: {mag:.2f} m/s^2 | Angle: {deg:.1f} deg | E2E Latency: {latency_ms:.2f} ms")
                
            now = time.time()
            if now - last_print >= 2.0:
                hz = count / (now - last_print)
                js = msg.get('joint_state', {})
                grip_pos = js.get('gripper', {}).get('pos_pct', 0.0) if js else 0.0
                print(f"[Telemetry] Rate: {hz:.1f} Hz | Latency: {latency_ms:.2f} ms | ax: {ax:+.2f} ay: {ay:+.2f} az: {az:+.2f} | GripPos: {grip_pos:.1f}%")
                count = 0
                last_print = now
                
    except KeyboardInterrupt:
        pass
    finally:
        socket.close()
        context.term()
        print("\nExited low-latency telemetry receiver.")


def main(use_remote_computer, display_scale, model_path=None, disable_rate_print=False,
         tap_threshold=2.5, low_latency_mode=False, no_gui=False):
    
    if low_latency_mode:
        run_low_latency_mode(use_remote_computer, tap_threshold=tap_threshold)
        return

    # Check GUI display availability
    has_display = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')) and not no_gui

    visualizer = None
    aruco_to_tips = None
    if model_path is not None:
        print(f"Loading Fingertip Visualizer with model: {model_path}")
        try:
            visualizer = FingertipVisualizer(model_path)
            aruco_to_tips = af.ArucoToFingertips()
        except Exception as e:
            print(f"Warning: Could not initialize FingertipVisualizer: {e}")

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1)
    
    host_ip = gn.robot_ip if use_remote_computer else '127.0.0.1'
    address = f'tcp://{host_ip}:{gn.gripper_and_joints_port}'
    print(f"Connecting ZMQ Subscriber to {address}")
    socket.connect(address)

    # History reconstructors
    joint_history_mgr = tu.JointStateHistory(maxlen=400)
    imu_history_mgr = tu.IMUHistory(maxlen=400)
    bump_detector = tu.BumpDetector(threshold=tap_threshold, max_angle_deg=45.0, cooldown_seconds=0.3)
    
    last_print_time = time.time()
    messages_received = 0
    last_seq_num = None
    dropped_messages = 0

    print("Receiving synced imagery, joint states, and IMU measurements... Press 'q' or 'Esc' to quit.")
    try:
        while True:
            output_dict = socket.recv_pyobj()
            
            if 'color_image_compressed' in output_dict:
                color_image = cv2.imdecode(np.frombuffer(output_dict['color_image_compressed'], np.uint8), cv2.IMREAD_COLOR)
            else:
                color_image = output_dict.get('color_image')
                
            depth_image = output_dict.get('depth_image', None)
            joint_history = output_dict.get('joint_state_history', [])
            imu_history = output_dict.get('imu_history', [])
            closest_joint = output_dict.get('closest_joint_state', None)
            closest_imu = output_dict.get('closest_imu_measurement', None)
            sync_metadata = output_dict.get('sync_metadata', {})
            img_seq = output_dict.get('image_number', 'N/A')
            cam_ts = output_dict.get('camera_timestamp', time.monotonic())
            
            messages_received += 1
            if img_seq != 'N/A':
                if last_seq_num is not None:
                    dropped = img_seq - last_seq_num - 1
                    if dropped > 0:
                        dropped_messages += dropped
                last_seq_num = img_seq
                
            current_time = time.time()
            elapsed_time = current_time - last_print_time
            if elapsed_time >= 5.0:
                if not disable_rate_print:
                    hz = messages_received / elapsed_time
                    j_offset = sync_metadata.get('closest_joint_state', {}).get('offset_to_image_ms')
                    i_offset = sync_metadata.get('closest_imu_measurement', {}).get('offset_to_image_ms')
                    ji_offset = sync_metadata.get('joint_to_imu_offset_ms')
                    j_str = f"{j_offset:+.2f} ms" if j_offset is not None else "N/A"
                    i_str = f"{i_offset:+.2f} ms" if i_offset is not None else "N/A"
                    ji_str = f"{ji_offset:+.2f} ms" if ji_offset is not None else "N/A"
                    print(f"Rate: {hz:.2f} Hz | Dropped: {dropped_messages} | Offsets: Cam-Joint: {j_str}, Cam-IMU: {i_str}, Joint-IMU: {ji_str}")
                messages_received = 0
                dropped_messages = 0
                last_print_time = current_time

            # Update histories
            joint_history_mgr.add_states(joint_history)
            imu_history_mgr.add_measurements(imu_history)
            
            # Detect bumps / taps in newly received IMU measurements
            new_bumps = bump_detector.update_history(imu_history)
            for b in new_bumps:
                print(f">>> [TAP DETECTED] Peak ax: {b['peak_ax']:.2f} m/s^2 | Mag: {b['magnitude']:.2f} m/s^2 | Angle: {b['angle_deg']:.1f} deg")

            if not has_display or color_image is None:
                continue

            rgb_camera_info = {
                'camera_matrix': output_dict.get('camera_matrix', np.eye(3)),
                'distortion_coefficients': output_dict.get('distortion_coefficients', np.zeros(5))
            }
            
            display_img = color_image.copy()
            display_img, scaled_camera_info = vu.apply_display_scale(display_img, display_scale, camera_info=rgb_camera_info)

            # Visualize fingertip kinematics based on pos_pct
            if visualizer is not None and closest_joint is not None:
                matched_pct = closest_joint['gripper']['pos_pct']
                full_history_list = joint_history_mgr.get_history_list()
                
                direction = 'closing'
                if len(full_history_list) > 10:
                    recent = full_history_list[-10:]
                    diff = recent[-1]['gripper']['pos_pct'] - recent[0]['gripper']['pos_pct']
                    direction = 'opening' if diff > 0 else 'closing'
                    
                predicted_fingertips = {}
                for side in ['left', 'right']:
                    pos_pred, rot_pred = visualizer.predict(side, matched_pct, direction)
                    if pos_pred is not None and rot_pred is not None:
                        predicted_fingertips[side] = {
                            'pos': pos_pred,
                            'x_axis': rot_pred[:, 0],
                            'y_axis': rot_pred[:, 1],
                            'z_axis': rot_pred[:, 2]
                        }
                        
                vu.draw_predicted_frames(predicted_fingertips, display_img, scaled_camera_info)
                if aruco_to_tips is not None:
                    aruco_to_tips.draw_fingertip_suction_cups(predicted_fingertips, display_img, scaled_camera_info)

            # Annotate Closest Joint State and IMU Measurement
            text_lines = [f"Img: #{img_seq}"]
            
            if closest_joint is not None:
                pos = closest_joint['gripper']['pos_pct']
                eff = closest_joint['gripper']['effort']
                joint_seq = closest_joint.get('state_number', 'N/A')
                j_off = (closest_joint.get('monotonic_timestamp', cam_ts) - cam_ts) * 1000.0
                text_lines.append(f"Joint #{joint_seq} | Offset: {j_off:+.1f} ms | Pos: {pos:.1f}% | Eff: {eff:.1f}")
                
            if closest_imu is not None:
                imu_seq = closest_imu.get('sequence_number', 'N/A')
                i_off = (closest_imu.get('timestamp', cam_ts) - cam_ts) * 1000.0
                g_frame = closest_imu.get('gripper_frame', {})
                lin_acc = g_frame.get('linear_acceleration', {})
                ax = lin_acc.get('x', 0.0) if lin_acc else 0.0
                ay = lin_acc.get('y', 0.0) if lin_acc else 0.0
                az = lin_acc.get('z', 0.0) if lin_acc else 0.0
                text_lines.append(f"IMU #{imu_seq} | Offset: {i_off:+.1f} ms | ax:{ax:+.2f} ay:{ay:+.2f} az:{az:+.2f} m/s^2")

            y_offset = 25
            for line in text_lines:
                cv2.putText(display_img, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(display_img, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                y_offset += 22

            # Visual Tap Detection Alert Banner
            if bump_detector.is_alert_active(cam_ts, alert_duration=0.5):
                last_ev = bump_detector.last_bump_event
                banner_text = f"*** TAP DETECTED! *** (ax: {last_ev['peak_ax']:.2f} m/s^2)" if last_ev else "*** TAP DETECTED! ***"
                (bw, bh), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
                bx = max(10, (display_img.shape[1] - bw) // 2)
                by = 60
                # Red glowing box
                cv2.rectangle(display_img, (bx - 15, by - bh - 10), (bx + bw + 15, by + 10), (0, 0, 220), -1)
                cv2.rectangle(display_img, (bx - 15, by - bh - 10), (bx + bw + 15, by + 10), (255, 255, 255), 2)
                cv2.putText(display_img, banner_text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

            # Stacked Telemetry Graphs
            plot_width = display_img.shape[1]
            joint_graph_img = tu.draw_history_graphs(joint_history_mgr.get_history_list(), width=plot_width, height=120)
            imu_graph_img = tu.draw_imu_history_graphs(
                imu_history_mgr.get_history_list(),
                width=plot_width,
                height=130,
                threshold=tap_threshold,
                bump_events=bump_detector.bump_events
            )
            
            combined_display = np.vstack((display_img, joint_graph_img, imu_graph_img))
            
            cv2.namedWindow("Synced Gripper Imagery, Kinematics & IMU Telemetry", cv2.WINDOW_NORMAL)
            cv2.imshow("Synced Gripper Imagery, Kinematics & IMU Telemetry", combined_display)
            
            # Show depth if available
            if depth_image is not None:
                depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                depth_vis = vu.apply_display_scale(depth_vis, display_scale)
                cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)
                cv2.imshow("Depth Image", depth_vis)
            
            key = cv2.waitKey(1)
            if key in (27, ord('q')):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("\nStopped receiving.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Receive Synced Gripper Images, Joint States, and IMU',
        description='Display synchronized gripper images, joint kinematics, and 9-axis IMU measurements.'
    )
    parser.add_argument('-r', '--remote', action='store_true',
                        help='Use this argument when running the code on a remote computer. Configure gripper_networking.py first.')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to the model planar YAML file. Activates kinematic visualization. If not provided, defaults to latest fleet calibration.')
    parser.add_argument('--disable-rate-print', action='store_true',
                        help='Disable printing of the receiving rate and dropped messages.')
    parser.add_argument('--tap-threshold', type=float, default=2.5,
                        help='Acceleration magnitude threshold in m/s^2 for bump/tap detection normal to OAK-D-SR (default: 2.5).')
    parser.add_argument('--low-latency-mode', action='store_true',
                        help='Connect directly to Tier 1 fast telemetry stream (Port 4410) for ultra-low-latency closed-loop control.')
    parser.add_argument('--no-gui', action='store_true',
                        help='Disable OpenCV GUI windows for headless environments or automated benchmarking.')
    vu.add_display_scale_argument(parser)

    args = parser.parse_args()

    if args.model is None and not args.low_latency_mode:
        args.model = cu.get_default_model_path()
        if args.model:
            print(f"No model path provided. Defaulting to fleet calibration: {args.model}")

    main(
        use_remote_computer=args.remote,
        display_scale=args.display_scale,
        model_path=args.model,
        disable_rate_print=args.disable_rate_print,
        tap_threshold=args.tap_threshold,
        low_latency_mode=args.low_latency_mode,
        no_gui=args.no_gui
    )
