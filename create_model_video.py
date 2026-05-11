#!/usr/bin/env python3

import os
import sys
import glob
import yaml
import argparse
import numpy as np
import cv2
import subprocess

from visualize_fingertip_model import FingertipVisualizer
from recv_and_detect_fingertips import FingertipDetector, add_fingertip_detector_args, process_fingertip_detector_args
from stretch4_gripper_modeling_and_control import visualization_utils as vu
from stretch4_gripper_modeling_and_control import calibration_utils as cu

def main():
    parser = argparse.ArgumentParser(description='Generate a video visualizing model predictions on raw calibration images.')
    parser.add_argument('model_path', type=str, nargs='?', help='Path to the model YAML file')
    vu.add_suction_cup_argument(parser)
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for output video (default: 30)')
    add_fingertip_detector_args(parser)
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = cu.get_default_model_path()
        if args.model_path:
            print(f"No model path provided. Defaulting to fleet calibration: {args.model_path}")
        else:
            print("Error: No model path provided and could not locate latest_model_planar.yaml in fleet directory.")
            sys.exit(1)

    model_path = os.path.abspath(args.model_path)
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        sys.exit(1)

    model = FingertipVisualizer(model_path)
    
    with open(model_path, 'r') as f:
        data = yaml.safe_load(f)
        calib_dir = data.get('metadata', {}).get('source_directory')
        if not calib_dir or not os.path.exists(calib_dir):
            calib_dir = os.path.dirname(model_path)

    yaml_files = glob.glob(os.path.join(calib_dir, "*_gripper_data_*.yaml"))
    
    camera_matrix = None
    dist_coeffs = None
    if yaml_files:
        with open(yaml_files[0], 'r') as f:
            calib_data = yaml.safe_load(f)
            if 'metadata' in calib_data and 'camera_matrix' in calib_data['metadata']:
                camera_matrix = np.array(calib_data['metadata']['camera_matrix'])
                dist_coeffs = np.array(calib_data['metadata']['distortion_coefficients'])

    if camera_matrix is None:
        print("Warning: Camera intrinsics not found. Cannot project 3D frames correctly.")
        sys.exit(1)

    rgb_camera_info = {
        'camera_matrix': camera_matrix,
        'distortion_coefficients': dist_coeffs
    }
    
    detector = process_fingertip_detector_args(args)
    generated_videos = []
    
    for subdir in ['closing_images', 'opening_images']:
        dir_path = os.path.join(calib_dir, subdir)
        if not os.path.exists(dir_path): continue
        
        yaml_name = 'closing_gripper_data' if subdir == 'closing_images' else 'opening_gripper_data'
        phase_yamls = glob.glob(os.path.join(calib_dir, f"{yaml_name}_*.yaml"))
        if not phase_yamls: continue
        
        with open(phase_yamls[0], 'r') as f:
            phase_data = yaml.safe_load(f)
            
        dataset = phase_data.get('data', [])
        img_num_to_pos = {}
        for row in dataset:
            pos_pct_before = row.get('gripper_status_before', {}).get('pos_pct')
            pos_pct_after = row.get('gripper_status_after', {}).get('pos_pct')
            if pos_pct_before is not None and pos_pct_after is not None:
                img_num_to_pos[row['image_number']] = (pos_pct_before + pos_pct_after) / 2.0
                
        raw_images = sorted(glob.glob(os.path.join(dir_path, "raw_image_*.png")))
        if not raw_images: continue
        
        print(f"Rendering model predictions for {subdir}...")
        for raw_img_path in raw_images:
            filename = os.path.basename(raw_img_path)
            try:
                img_num = int(filename.replace('raw_image_', '').replace('.png', ''))
            except:
                continue
                
            if img_num not in img_num_to_pos:
                continue
                
            pos_pct = img_num_to_pos[img_num]
            img = cv2.imread(raw_img_path)
            if img is None: continue
            
            pred_frames = {}
            for side in ['left', 'right']:
                dir_str = 'closing' if 'closing' in subdir else 'opening'
                pos, F_rot = model.predict(side, pos_pct, direction=dir_str)
                if pos is not None:
                    pred_frames[side] = {
                        'pos': pos,
                        'x_axis': F_rot[:,0],
                        'y_axis': F_rot[:,1],
                        'z_axis': F_rot[:,2]
                    }
                    
            if not args.disable_suction_cups:
                detector.aruco_to_fingertips.draw_fingertip_suction_cups(pred_frames, img, rgb_camera_info)
                
            model_img = detector.draw_fingertip_frames(pred_frames, img.copy(), rgb_camera_info)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(model_img, f"MODEL PRED | pos_pct: {pos_pct:.2f}", (10, 30), font, 0.7, (255, 100, 0), 2, cv2.LINE_AA)
            if 'left' in pred_frames and 'right' in pred_frames:
                dist = np.linalg.norm(pred_frames['left']['pos'] - pred_frames['right']['pos']) * 1000.0
                cv2.putText(model_img, f"Pred Dist: {dist:.0f} mm", (10, 60), font, 0.7, (255, 100, 0), 2, cv2.LINE_AA)
                
            out_path = os.path.join(dir_path, f"model_image_{img_num:05d}.png")
            cv2.imwrite(out_path, model_img)
            
        print(f"Creating video for {subdir}...")
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        model_video_path = os.path.join(calib_dir, f"{model_name}_video_{subdir}.mp4")
        if os.path.exists(model_video_path):
            os.remove(model_video_path)
            
        cmd = [
            'ffmpeg', '-y', '-framerate', str(args.fps),
            '-pattern_type', 'glob',
            '-i', f'{dir_path}/model_image_*.png',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            model_video_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Clean up temporary frames
        for f in glob.glob(os.path.join(dir_path, "model_image_*.png")):
            os.remove(f)
            
        print(f"[{subdir}] Model video saved to {model_video_path}")
        generated_videos.append(model_video_path)
        
    if len(generated_videos) > 0:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        combined_video_path = os.path.join(calib_dir, f"{model_name}_video_{args.fps}fps.mp4")
        concat_file = os.path.join(calib_dir, "ffmpeg_concat_model_tmp.txt")
        with open(concat_file, "w") as f:
            for vid in generated_videos:
                f.write(f"file '{vid}'\n")
        
        print(f"Combining videos into {combined_video_path}...")
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', 
            '-i', concat_file, '-c', 'copy', combined_video_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists(concat_file):
            os.remove(concat_file)
            
        print(f"Combined model video saved to {combined_video_path}")

if __name__ == '__main__':
    main()
