#!/usr/bin/env python3

import os
import sys
import glob
import yaml
import argparse
import numpy as np
import cv2
import subprocess

from recv_and_detect_fingertips import FingertipDetector, add_fingertip_detector_args, process_fingertip_detector_args
from stretch4_gripper_modeling_and_control import visualization_utils as vu

def main():
    parser = argparse.ArgumentParser(description='Create a calibration video from raw images and YAML data.')
    parser.add_argument('calib_dir', type=str, help='Path to the timestamped gripper calibration directory')
    parser.add_argument('--fps', type=int, default=5, help='Frames per second for output video (default: 5)')
    vu.add_suction_cup_argument(parser)
    add_fingertip_detector_args(parser)
    args = parser.parse_args()

    calib_dir = os.path.abspath(args.calib_dir)
    
    if not os.path.isdir(calib_dir):
        print(f"Error: {calib_dir} is not a valid directory.")
        sys.exit(1)

    closing_dir = os.path.join(calib_dir, "closing_images")
    opening_dir = os.path.join(calib_dir, "opening_images")
    if not os.path.isdir(closing_dir) or not os.path.isdir(opening_dir):
        print(f"Error: Expecting {closing_dir} and {opening_dir} to exist.")
        sys.exit(1)

    detector = process_fingertip_detector_args(args)

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

    generated_videos = []

    for motion_direction in ['closing', 'opening']:
        dir_path = os.path.join(calib_dir, f"{motion_direction}_images")
        yaml_name = f"{motion_direction}_gripper_data"
        phase_yamls = glob.glob(os.path.join(calib_dir, f"{yaml_name}_*.yaml"))
        if not phase_yamls: continue
        
        with open(phase_yamls[0], 'r') as f:
            phase_data = yaml.safe_load(f)
            
        dataset = phase_data.get('data', [])
        if not dataset: continue
        
        start_time = dataset[0].get('image_timestamp', 0)
        
        print(f"Rendering frames for {motion_direction}...")
        for row in dataset:
            img_num = row['image_number']
            raw_img_path = os.path.join(dir_path, f"raw_image_{img_num:05d}.png")
            if not os.path.exists(raw_img_path): continue
            
            img = cv2.imread(raw_img_path)
            if img is None: continue

            # Reconstruct fingertips dict structure from row elements
            fingertips = {}
            for side in ['left', 'right']:
                side_key = f"{side}_frame"
                if side_key in row:
                    f_data = row[side_key]
                    fingertips[side] = {
                        'pos': np.array(f_data['pos']),
                        'x_axis': np.array(f_data['x_axis']),
                        'y_axis': np.array(f_data['y_axis']),
                        'z_axis': np.array(f_data['z_axis'])
                    }
                    if 'alt' in f_data:
                        fingertips[side]['alt'] = {
                            'pos': np.array(f_data['alt']['pos']),
                            'x_axis': np.array(f_data['alt']['x_axis']),
                            'y_axis': np.array(f_data['alt']['y_axis']),
                            'z_axis': np.array(f_data['alt']['z_axis'])
                        }
            
            if not args.disable_suction_cups:
                detector.aruco_to_fingertips.draw_fingertip_suction_cups(fingertips, img, rgb_camera_info)

            annotated_img = detector.draw_fingertip_frames(fingertips, img.copy(), rgb_camera_info)

            # Re-draw the context text
            img_ts = row.get('image_timestamp', start_time)
            closest_before = row.get('gripper_status_before', {})
            closest_after = row.get('gripper_status_after', {})
            
            time_since_motion = img_ts - start_time
            motion_str = f"Motion: {motion_direction}"
            time_str = f"Time since start: {time_since_motion:.2f} s"
            pos_str = f"pos_pct before: {closest_before.get('pos_pct',0):.2f}, after: {closest_after.get('pos_pct',0):.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(annotated_img, motion_str, (10, 30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_img, time_str, (10, 60), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_img, pos_str, (10, 90), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            if 'left' in fingertips and 'right' in fingertips:
                dist = np.linalg.norm(fingertips['left']['pos'] - fingertips['right']['pos'])
                aperture_before = closest_before.get('gripper_conversion', {}).get('aperture_m', 0.0)
                aperture_after = closest_after.get('gripper_conversion', {}).get('aperture_m', 0.0)
                dist_mm = dist * 1000.0
                ap_before_mm = aperture_before * 1000.0
                ap_after_mm = aperture_after * 1000.0
                dist_str = f"Dist: {dist_mm:.0f} mm, Aperture (before/after): {ap_before_mm:.0f} mm / {ap_after_mm:.0f} mm"
                
                error_before_mm = (dist - aperture_before) * 1000.0
                error_after_mm = (dist - aperture_after) * 1000.0
                error_str = f"Error (before/after): {error_before_mm:.0f} mm / {error_after_mm:.0f} mm"
                
                cv2.putText(annotated_img, dist_str, (10, 120), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(annotated_img, error_str, (10, 150), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
            out_path = os.path.join(dir_path, f"annotated_image_{img_num:05d}.png")
            cv2.imwrite(out_path, annotated_img)

        print(f"Creating video for {motion_direction}...")
        video_path = os.path.join(calib_dir, f"calibration_video_{motion_direction}.mp4")
        if os.path.exists(video_path):
            os.remove(video_path)
            
        cmd = [
            'ffmpeg', '-y', '-framerate', str(args.fps),
            '-pattern_type', 'glob',
            '-i', f'{dir_path}/annotated_image_*.png',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            video_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Clean up temporary frames
        for f in glob.glob(os.path.join(dir_path, "annotated_image_*.png")):
            os.remove(f)
            
        print(f"[{motion_direction}] Video saved to {video_path}")
        generated_videos.append(video_path)
        
    if len(generated_videos) > 0:
        combined_video_path = os.path.join(calib_dir, f"calibration_video_{args.fps}fps.mp4")
        concat_file = os.path.join(calib_dir, "ffmpeg_concat_calib_tmp.txt")
        with open(concat_file, "w") as f:
            for vid in generated_videos:
                f.write(f"file '{os.path.basename(vid)}'\n")
        
        print(f"Combining videos into {combined_video_path}...")
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', 
            '-i', concat_file, '-c', 'copy', combined_video_path
        ], cwd=calib_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists(concat_file):
            os.remove(concat_file)
            
        print(f"Combined calibration video saved to {combined_video_path}")

if __name__ == '__main__':
    main()
