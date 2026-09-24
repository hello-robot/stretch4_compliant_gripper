#!/usr/bin/env python3
"""
IMU Bump Detection & Audio Feedback Demo for the Stretch 4 Gripper.

Based on recv_gripper_images_joint_states_and_imu.py.
Receives synchronized gripper images, robot joint states, and wrist camera IMU measurements.
Plays 'point_score.wav' (or customizable sound) each time a tap/bump is detected normal
to the front face of the wrist-mounted Luxonis OAK-D-SR stereo camera.

Features:
- Instantaneous, non-blocking audio playback upon impact detection.
- Live video stream with fingertip kinematics model overlay.
- Visual alert banner with tap telemetry and point score tally.
- Real-time stacked history plots of joint states and 3-axis linear acceleration.
- Optional --low-latency-mode connecting to Tier 1 fast telemetry (Port 4410)
  for sub-10 ms audio reflex from physical impact.
"""

import argparse
import collections
import cv2
import zmq
import numpy as np
import time
import os
import threading
import subprocess

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from stretch4_gripper_modeling_and_control import gripper_networking as gn
from stretch4_gripper_modeling_and_control import visualization_utils as vu
from stretch4_gripper_modeling_and_control import telemetry_utils as tu
from stretch4_gripper_modeling_and_control import calibration_utils as cu
from visualize_fingertip_model import FingertipVisualizer
from stretch4_gripper_modeling_and_control import aruco_to_fingertips as af


class SoundPlayer:
    """
    Ultra-low-latency sound player using pygame.mixer with automatic fallback
    to system audio utilities (pw-play / aplay).
    """
    def __init__(self, sound_path=None, volume=1.0):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if sound_path is None:
            sound_path = os.path.join(base_dir, 'sounds', 'point_score.wav')
        elif not os.path.isabs(sound_path) and not os.path.exists(sound_path):
            candidate = os.path.join(base_dir, 'sounds', sound_path)
            if os.path.exists(candidate):
                sound_path = candidate
            elif os.path.exists(candidate + '.wav'):
                sound_path = candidate + '.wav'
                
        self.sound_path = sound_path
        self.volume = max(0.0, min(1.0, volume))
        self.has_pygame = False
        self.sound = None
        self.play_count = 0
        
        if not os.path.isfile(self.sound_path):
            print(f"Warning: Sound file '{self.sound_path}' not found.")
            return

        if PYGAME_AVAILABLE:
            try:
                # 512-sample low-latency audio buffer
                pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
                pygame.mixer.init()
                self.sound = pygame.mixer.Sound(self.sound_path)
                self.sound.set_volume(self.volume)
                self.has_pygame = True
                print(f"Audio Feedback: Loaded '{os.path.basename(self.sound_path)}' via pygame.mixer (low-latency).")
            except Exception as e:
                print(f"Audio Feedback: pygame.mixer init warning ({e}), falling back to system audio player.")
        else:
            print("Audio Feedback: pygame not installed, using system audio player.")

    def play(self):
        """Dispatches audio playback asynchronously with sub-millisecond overhead."""
        self.play_count += 1
        if self.has_pygame and self.sound is not None:
            try:
                self.sound.play()
                return
            except Exception:
                pass
                
        # Non-blocking fallback for systems without pygame audio
        if os.path.isfile(self.sound_path):
            threading.Thread(target=self._fallback_play, daemon=True).start()

    def _fallback_play(self):
        try:
            subprocess.run(['pw-play', self.sound_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        except Exception:
            try:
                subprocess.run(['aplay', self.sound_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            except Exception:
                pass

    def close(self):
        if self.has_pygame:
            try:
                pygame.mixer.quit()
            except Exception:
                pass


def run_low_latency_mode(use_remote_computer, sound_player, tap_threshold=2.5):
    """
    Direct subscriber to Tier 1 fast telemetry channel (Port 4410).
    Evaluates bumps at ~100 Hz and triggers the sound reflex within ~4 ms of impact.
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
    
    print("[Low-Latency Mode] Running fast audio reflex loop... Press Ctrl+C to stop.")
    try:
        while True:
            msg = socket.recv_pyobj()
            count += 1
            now_mono = time.monotonic()
            
            ts = msg.get('timestamp', now_mono)
            latency_ms = (now_mono - ts) * 1000.0
            
            imu_data = msg.get('imu', {})
            lin_acc = imu_data.get('linear_acceleration', {})
            ax = lin_acc.get('x', 0.0) if lin_acc else 0.0
            ay = lin_acc.get('y', 0.0) if lin_acc else 0.0
            az = lin_acc.get('z', 0.0) if lin_acc else 0.0
            
            bump_event = msg.get('bump_event')
            if bump_event is None and lin_acc:
                bump_event = detector.update({
                    'timestamp': ts,
                    'sequence_number': msg.get('sequence_number'),
                    'gripper_frame': imu_data
                })
                
            if bump_event:
                sound_player.play()
                peak = bump_event.get('peak_ax', ax)
                mag = bump_event.get('magnitude', 0.0)
                deg = bump_event.get('angle_deg', 0.0)
                print(f"\n>>> [TAP DETECTED! POINT SCORED!] Bump #{sound_player.play_count} | Peak ax: {peak:.2f} m/s^2 | Mag: {mag:.2f} m/s^2 | Angle: {deg:.1f} deg | E2E Latency: {latency_ms:.2f} ms")
                
            now = time.time()
            if now - last_print >= 2.0:
                hz = count / (now - last_print)
                js = msg.get('joint_state', {})
                grip_pos = js.get('gripper', {}).get('pos_pct', 0.0) if js else 0.0
                print(f"[Telemetry] Rate: {hz:.1f} Hz | Latency: {latency_ms:.2f} ms | Score: {sound_player.play_count} | ax: {ax:+.2f} ay: {ay:+.2f} az: {az:+.2f} | GripPos: {grip_pos:.1f}%")
                count = 0
                last_print = now
                
    except KeyboardInterrupt:
        pass
    finally:
        socket.close()
        context.term()
        sound_player.close()
        print("\nExited low-latency sound receiver.")


def main(use_remote_computer, display_scale, model_path=None, disable_rate_print=False,
         tap_threshold=2.5, low_latency_mode=False, no_gui=False, sound_path=None, volume=1.0):
    
    sound_player = SoundPlayer(sound_path=sound_path, volume=volume)

    if low_latency_mode:
        run_low_latency_mode(use_remote_computer, sound_player, tap_threshold=tap_threshold)
        return

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

    joint_history_mgr = tu.JointStateHistory(maxlen=400)
    imu_history_mgr = tu.IMUHistory(maxlen=400)
    bump_detector = tu.BumpDetector(threshold=tap_threshold, max_angle_deg=45.0, cooldown_seconds=0.3)
    
    last_print_time = time.time()
    messages_received = 0
    last_seq_num = None
    dropped_messages = 0

    print(f"IMU Bump & Sound Demo active. Sound: '{os.path.basename(sound_player.sound_path)}'.")
    print("Tap an object held by the gripper normal to the camera face to score points! Press 'q' or 'Esc' to quit.")
    
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
                    print(f"Rate: {hz:.2f} Hz | Score: {sound_player.play_count} | Dropped: {dropped_messages} | Offsets: Cam-Joint: {j_str}, Cam-IMU: {i_str}, Joint-IMU: {ji_str}")
                messages_received = 0
                dropped_messages = 0
                last_print_time = current_time

            # Update histories
            joint_history_mgr.add_states(joint_history)
            imu_history_mgr.add_measurements(imu_history)
            
            # Detect bumps / taps in newly received IMU measurements and play sound
            new_bumps = bump_detector.update_history(imu_history)
            for b in new_bumps:
                sound_player.play()
                print(f"\n>>> [TAP DETECTED! POINT SCORED!] Bump #{sound_player.play_count} | Peak ax: {b['peak_ax']:.2f} m/s^2 | Mag: {b['magnitude']:.2f} m/s^2 | Angle: {b['angle_deg']:.1f} deg")

            if not has_display or color_image is None:
                continue

            rgb_camera_info = {
                'camera_matrix': output_dict.get('camera_matrix', np.eye(3)),
                'distortion_coefficients': output_dict.get('distortion_coefficients', np.zeros(5))
            }
            
            display_img = color_image.copy()
            display_img, scaled_camera_info = vu.apply_display_scale(display_img, display_scale, camera_info=rgb_camera_info)

            # Visualize fingertip kinematics based on pos_pct
            if False:
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
            text_lines = [f"Img: #{img_seq} | Point Score: {sound_player.play_count}"]
            
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

            # Visual Tap Detection Alert Banner with Score
            if bump_detector.is_alert_active(cam_ts, alert_duration=0.6):
                last_ev = bump_detector.last_bump_event
                peak_str = f"ax: {last_ev['peak_ax']:.2f} m/s^2" if last_ev else ""
                banner_text = f"*** POINT SCORED! ({sound_player.play_count}) *** {peak_str}"
                (bw, bh), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                bx = max(10, (display_img.shape[1] - bw) // 2)
                by = 60
                # Glowing Gold/Orange Box
                cv2.rectangle(display_img, (bx - 15, by - bh - 10), (bx + bw + 15, by + 10), (0, 140, 255), -1)
                cv2.rectangle(display_img, (bx - 15, by - bh - 10), (bx + bw + 15, by + 10), (255, 255, 255), 2)
                cv2.putText(display_img, banner_text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

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
            
            window_name = "Stretch 4 Gripper - IMU Bump Audio Demo"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1800, 1800)
            cv2.imshow(window_name, combined_display)

            if False: 
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
        sound_player.close()
        print("\nStopped receiving.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='IMU Bump Audio Demo',
        description='Play a sound (point_score.wav) each time an IMU bump is detected on the Stretch 4 gripper.'
    )
    parser.add_argument('-r', '--remote', action='store_true',
                        help='Use this argument when running the code on a remote computer. Configure gripper_networking.py first.')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to the model planar YAML file. Activates kinematic visualization. If not provided, defaults to latest fleet calibration.')
    parser.add_argument('--disable-rate-print', action='store_true',
                        help='Disable printing of the receiving rate and dropped messages.')
    parser.add_argument('--tap-threshold', type=float, default=6.0,
                        help='Acceleration magnitude threshold in m/s^2 for bump/tap detection normal to OAK-D-SR (default: 2.5).')
    parser.add_argument('--low-latency-mode', action='store_true',
                        help='Connect directly to Tier 1 fast telemetry stream (Port 4410) for ultra-low-latency sound reflex.')
    parser.add_argument('--no-gui', action='store_true',
                        help='Disable OpenCV GUI windows for headless environments or automated benchmarking.')
    parser.add_argument('--sound', type=str, default=None,
                        help="Path to WAV audio file to play on bump detection (default: 'sounds/point_score.wav').")
    parser.add_argument('--volume', type=float, default=1.0,
                        help="Audio playback volume between 0.0 and 1.0 (default: 1.0).")
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
        no_gui=args.no_gui,
        sound_path=args.sound,
        volume=args.volume
    )
