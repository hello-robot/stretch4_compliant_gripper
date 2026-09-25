#!/usr/bin/env python3

import argparse
import collections
import cv2
import zmq
import numpy as np
import time

from stretch4_gripper_modeling_and_control import gripper_networking as gn
from stretch4_gripper_modeling_and_control import visualization_utils as vu
from stretch4_gripper_modeling_and_control import telemetry_utils as tu
from stretch4_gripper_modeling_and_control import calibration_utils as cu
from visualize_fingertip_model import FingertipVisualizer
from stretch4_gripper_modeling_and_control import aruco_to_fingertips as af


def rotate_head_image_to_upright(img, head_camera_side):
    """
    Stretch 4 head fisheye cameras are mounted with landscape sensors.
    To display them upright:
      - Left fisheye camera is rotated 90 degrees CCW (cv2.ROTATE_90_COUNTERCLOCKWISE).
      - Right fisheye camera is rotated 90 degrees CW (cv2.ROTATE_90_CLOCKWISE).
    """
    if head_camera_side == 'left':
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif head_camera_side == 'right':
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def main(use_remote_computer, display_scale, model_path=None, disable_rate_print=False, no_rotate_head=False):
    visualizer = None
    aruco_to_tips = None
    if model_path is not None:
        print(f"Loading Fingertip Visualizer with model: {model_path}")
        visualizer = FingertipVisualizer(model_path)
        aruco_to_tips = af.ArucoToFingertips()

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

    reconstructor = tu.JointStateHistory(maxlen=400) # about 4 seconds
    
    last_print_time = time.time()
    messages_received = 0
    last_seq_num = None
    dropped_messages = 0

    print("Receiving synced wrist + head frames and joint states... Press 'q' or 'Esc' to quit.")
    try:
        while True:
            output_dict = socket.recv_pyobj()
            
            # 1. Decode wrist image
            wrist_img = None
            if 'wrist_color_image_compressed' in output_dict:
                wrist_img = cv2.imdecode(np.frombuffer(output_dict['wrist_color_image_compressed'], np.uint8), cv2.IMREAD_COLOR)
            elif 'wrist_color_image' in output_dict:
                wrist_img = output_dict['wrist_color_image']
            elif 'color_image_compressed' in output_dict:
                wrist_img = cv2.imdecode(np.frombuffer(output_dict['color_image_compressed'], np.uint8), cv2.IMREAD_COLOR)
            elif 'color_image' in output_dict:
                wrist_img = output_dict['color_image']

            # 2. Decode head image
            head_img = None
            if 'head_color_image_compressed' in output_dict:
                head_img = cv2.imdecode(np.frombuffer(output_dict['head_color_image_compressed'], np.uint8), cv2.IMREAD_COLOR)
            elif 'head_color_image' in output_dict:
                head_img = output_dict['head_color_image']

            wrist_side = output_dict.get('wrist_camera_side', 'wrist')
            head_side = output_dict.get('head_camera_side', 'head')
            img_seq = output_dict.get('image_number', 'N/A')
            head_seq = output_dict.get('head_image_number', 'N/A')
            head_offset_ms = output_dict.get('head_sync_offset_ms', 0.0)
            
            joint_history = output_dict.get('joint_state_history', [])
            closest = output_dict.get('closest_joint_state', None)
            
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
                    print(f"Rate: {hz:.2f} Hz | Head-Wrist sync offset: {head_offset_ms:+.1f} ms | Est. dropped messages in last {elapsed_time:.1f}s: {dropped_messages}")
                messages_received = 0
                dropped_messages = 0
                last_print_time = current_time

            reconstructor.add_states(joint_history)

            wrist_camera_info = {
                'camera_matrix': output_dict.get('wrist_camera_matrix', output_dict.get('camera_matrix', np.eye(3))),
                'distortion_coefficients': output_dict.get('wrist_distortion_coefficients', output_dict.get('distortion_coefficients', np.zeros(5)))
            }
            
            display_wrist = wrist_img.copy() if wrist_img is not None else np.zeros((400, 640, 3), dtype=np.uint8)

            # Apply display scale to wrist image
            display_wrist, scaled_camera_info = vu.apply_display_scale(display_wrist, display_scale, camera_info=wrist_camera_info)

            # Visualize kinematic model based on pos_pct if requested
            if visualizer is not None and closest is not None:
                matched_pct = closest['gripper']['pos_pct']
                full_history_list = reconstructor.get_history_list()
                
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
                        
                vu.draw_predicted_frames(predicted_fingertips, display_wrist, scaled_camera_info)
                if aruco_to_tips is not None:
                    aruco_to_tips.draw_fingertip_suction_cups(predicted_fingertips, display_wrist, scaled_camera_info)

            # Annotate Closest Joint State on Wrist Image
            if closest is not None:
                pos = closest['gripper']['pos_pct']
                eff = closest['gripper']['effort']
                offset = closest.get('time_relative_to_image', 0) * 1000.0 # to ms
                state_seq = closest.get('state_number', 'N/A')
                
                text_lines = [
                    f"Wrist ({wrist_side}): #{img_seq} | State: #{state_seq}",
                    f"Sync Offset: {offset:+.1f} ms",
                    f"Gripper Pos: {pos:.1f}",
                    f"Effort:      {eff:.1f}"
                ]
                
                y_offset = 30
                for line in text_lines:
                    cv2.putText(display_wrist, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(display_wrist, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
                    y_offset += 25

            # Prepare and annotate Head Image
            if head_img is not None:
                if not no_rotate_head:
                    head_disp = rotate_head_image_to_upright(head_img, head_side)
                else:
                    head_disp = head_img.copy()

                head_disp = vu.apply_display_scale(head_disp, display_scale)
                
                head_text_lines = [
                    f"Head ({head_side}): #{head_seq}",
                    f"Sync Offset: {head_offset_ms:+.1f} ms",
                ]
                hy_offset = 30
                for line in head_text_lines:
                    cv2.putText(head_disp, line, (10, hy_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(head_disp, line, (10, hy_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
                    hy_offset += 25
            else:
                head_disp = np.zeros_like(display_wrist)
                cv2.putText(head_disp, "No Head Frame", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Match display heights for side-by-side presentation
            target_h = max(display_wrist.shape[0], head_disp.shape[0])
            if display_wrist.shape[0] != target_h:
                scale_w = target_h / display_wrist.shape[0]
                display_wrist = cv2.resize(display_wrist, (int(display_wrist.shape[1] * scale_w), target_h))
            if head_disp.shape[0] != target_h:
                scale_h = target_h / head_disp.shape[0]
                head_disp = cv2.resize(head_disp, (int(head_disp.shape[1] * scale_h), target_h))

            # Combine head (left) + wrist (right) horizontally
            combined_images = np.hstack((head_disp, display_wrist))

            # Render joint state telemetry graphs below the combined images
            graph_img = tu.draw_history_graphs(reconstructor.get_history_list(), width=combined_images.shape[1], height=150)
            combined_display = np.vstack((combined_images, graph_img))

            cv2.namedWindow("Synced Gripper + Head Telemetry", cv2.WINDOW_NORMAL)
            cv2.imshow("Synced Gripper + Head Telemetry", combined_display)
            
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
        prog='Receive Gripper and Head Images with Joint States',
        description='Display synchronized wrist camera, head fisheye camera, and joint state telemetry.'
    )
    parser.add_argument('-r', '--remote', action='store_true', help='Use this argument when running on a remote computer. Configure gripper_networking.py first.')
    parser.add_argument('--model', type=str, default=None, help='Path to the model planar YAML file. Activates kinematic visualization. If not provided, defaults to the latest fleet calibration model.')
    parser.add_argument('--disable-rate-print', action='store_true', help='Disable printing of the receiving rate and dropped messages.')
    parser.add_argument('--no-rotate-head', action='store_true', help='Disable automatic 90-degree upright rotation of the head fisheye image.')
    vu.add_display_scale_argument(parser)

    args = parser.parse_args()

    if args.model is None:
        args.model = cu.get_default_model_path()
        if args.model:
            print(f"No model path provided. Defaulting to fleet calibration: {args.model}")
            
    main(args.remote, args.display_scale, args.model, args.disable_rate_print, args.no_rotate_head)
