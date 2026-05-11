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

def main(use_remote_computer, display_scale, model_path=None, disable_rate_print=False):
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

    # Use the new utility to correctly align timestamps
    reconstructor = tu.JointStateHistory(maxlen=400) # about 4 seconds
    
    last_print_time = time.time()
    messages_received = 0
    last_seq_num = None
    dropped_messages = 0

    print("Receiving synced frames and joint states... Press 'q' or 'Esc' to quit.")
    try:
        while True:
            output_dict = socket.recv_pyobj()
            
            if 'color_image_compressed' in output_dict:
                color_image = cv2.imdecode(np.frombuffer(output_dict['color_image_compressed'], np.uint8), cv2.IMREAD_COLOR)
            else:
                color_image = output_dict['color_image']
                
            depth_image = output_dict.get('depth_image', None)
            joint_history = output_dict.get('joint_state_history', [])
            closest = output_dict.get('closest_joint_state', None)
            img_seq = output_dict.get('image_number', 'N/A')
            
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
                    print(f"Rate: {hz:.2f} Hz | Estimated dropped messages in last {elapsed_time:.1f}s: {dropped_messages}")
                messages_received = 0
                dropped_messages = 0
                last_print_time = current_time

            reconstructor.add_states(joint_history)

            rgb_camera_info = {
                'camera_matrix': output_dict.get('camera_matrix', np.eye(3)),
                'distortion_coefficients': output_dict.get('distortion_coefficients', np.zeros(5))
            }
            
            display_img = color_image.copy()

            # Apply display scale here so everything drawn correctly fits
            display_img, scaled_camera_info = vu.apply_display_scale(display_img, display_scale, camera_info=rgb_camera_info)

            # Visualize kinematic model based on pos_pct
            if visualizer is not None and closest is not None:
                matched_pct = closest['gripper']['pos_pct']
                
                # Retrieve the assembled python array from the reconstructor
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
                        
                vu.draw_predicted_frames(predicted_fingertips, display_img, scaled_camera_info)
                if aruco_to_tips is not None:
                    aruco_to_tips.draw_fingertip_suction_cups(predicted_fingertips, display_img, scaled_camera_info)
            
            # Annotate Closest Joint State on Image
            if closest is not None:
                pos = closest['gripper']['pos_pct']
                eff = closest['gripper']['effort']
                offset = closest.get('time_relative_to_image', 0) * 1000.0 # to ms
                state_seq = closest.get('state_number', 'N/A')
                
                text_lines = [
                    f"Img: #{img_seq} | State: #{state_seq}",
                    f"Sync Offset: {offset:+.1f} ms",
                    f"Gripper Pos: {pos:.1f}",
                    f"Effort:      {eff:.1f}"
                ]
                
                y_offset = 30
                for line in text_lines:
                    cv2.putText(display_img, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(display_img, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                    y_offset += 25
            
            # Plot graphs
            graph_img = tu.draw_history_graphs(reconstructor.get_history_list(), width=display_img.shape[1], height=150)
            
            combined_display = np.vstack((display_img, graph_img))
            
            cv2.namedWindow("Synced Gripper Telemetry", cv2.WINDOW_NORMAL)
            cv2.imshow("Synced Gripper Telemetry", combined_display)
            
            # Show depth
            if depth_image is not None:
                # Basic depth visualization
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
        prog='Receive Synced Gripper Images and Joint States',
        description='Display images combined with perfectly synchronized joint state telemetry.'
    )
    parser.add_argument('-r', '--remote', action='store_true', help='Use this argument when running the code on a remote computer. Configure gripper_networking.py first.')
    parser.add_argument('--model', type=str, default=None, help='Path to the model planar YAML file. Activates kinematic visualization. If not provided, defaults to the latest fleet calibration model.')
    parser.add_argument('--disable-rate-print', action='store_true', help='Disable printing of the receiving rate and dropped messages.')
    vu.add_display_scale_argument(parser)

    args = parser.parse_args()

    if args.model is None:
        args.model = cu.get_default_model_path()
        if args.model:
            print(f"No model path provided. Defaulting to fleet calibration: {args.model}")
    main(args.remote, args.display_scale, args.model, args.disable_rate_print)
