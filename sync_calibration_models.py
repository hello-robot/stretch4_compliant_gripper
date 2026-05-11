#!/usr/bin/env python3
import os
import subprocess
import argparse
import tempfile
import shutil
import yaml
from stretch4_gripper_modeling_and_control.gripper_networking import robot_ip

def main():
    parser = argparse.ArgumentParser(description="Sync gripper calibration models from the robot to the local desktop.")
    parser.add_argument('--ip', type=str, default=robot_ip, help="The IP address of the robot")
    parser.add_argument('--user', type=str, default='hello-robot', help="The SSH user for the robot")
    args = parser.parse_args()

    local_fleet_path = os.environ.get('HELLO_FLEET_PATH', os.path.expanduser('~/stretch_user'))
    
    print(f"Connecting to {args.user}@{args.ip} to fetch calibration models...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # We use a single scp command with a wildcard to copy both latest_model_planar.yaml and latest_mirror_transforms.yaml
        # The wildcard is expanded on the remote side, avoiding the need to know the robot_id upfront.
        remote_pattern = f'/home/{args.user}/stretch_user/*/calibration_gripper/latest_*.yaml'
        
        command = [
            'scp', 
            f'{args.user}@{args.ip}:{remote_pattern}', 
            temp_dir
        ]
        
        try:
            subprocess.run(command, check=True)
            print("Successfully downloaded calibration files to a temporary directory.")
        except subprocess.CalledProcessError as e:
            print(f"Error syncing calibration files: {e}")
            return
            
        model_planar_path = os.path.join(temp_dir, 'latest_model_planar.yaml')
        if not os.path.exists(model_planar_path):
            print("Error: Could not find latest_model_planar.yaml in the downloaded files.")
            return
            
        with open(model_planar_path, 'r') as f:
            try:
                data = yaml.safe_load(f)
                robot_id = data.get('metadata', {}).get('robot_id')
            except Exception as e:
                print(f"Error parsing {model_planar_path}: {e}")
                return
                
        if not robot_id:
            print("Error: Could not extract robot_id from the downloaded latest_model_planar.yaml.")
            return
            
        print(f"Extracted robot_id: {robot_id}")
            
        local_dir = os.path.join(local_fleet_path, robot_id, 'calibration_gripper')
        os.makedirs(local_dir, exist_ok=True)
        
        # Move files to the correct local directory
        shutil.move(model_planar_path, os.path.join(local_dir, 'latest_model_planar.yaml'))
        
        mirror_transforms_path = os.path.join(temp_dir, 'latest_mirror_transforms.yaml')
        if os.path.exists(mirror_transforms_path):
            shutil.move(mirror_transforms_path, os.path.join(local_dir, 'latest_mirror_transforms.yaml'))
            print(f"Successfully synced both calibration models to {local_dir}")
        else:
            print(f"Successfully synced latest_model_planar.yaml to {local_dir}")
            print(f"Note: latest_mirror_transforms.yaml was not found.")

if __name__ == "__main__":
    main()
