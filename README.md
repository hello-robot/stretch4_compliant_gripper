# Stretch 4 Gripper Modeling and Control

This repository provides tools and utilities for modeling, calibrating, and controlling the standard compliant gripper on the Stretch 4 robot. It includes capabilities for kinematic modeling of the fingertips, teleoperation, and visually estimating fingertip configurations via the wrist-mounted gripper camera.

## Table of Contents
1. [Installation](#installation)
2. [Calibrate the Kinematic Model](#calibrate-the-kinematic-model)
3. [Visualizing Fingertips](#visualizing-fingertips)
4. [Estimating Fingertip Loads](#estimating-fingertip-loads)
5. [Teleoperation](#teleoperation)
6. [Synergy with Grasping Demo](#synergy-with-grasping-demo)

---

## Installation

This repository has been structured as a pip-installable Python package. Collecting gripper calibration data, fitting the calibrated gripper kinematic model, and running teleoperation scripts can be done with or without a remote desktop connection.

For integration with `stretch4_grasping_demo`, you should install this repository on **both** the Stretch 4 robot's onboard NUC computer (Ubuntu 24.04) and your offboard desktop machine (Ubuntu 24.04) running the GPU-accelerated code.

### Prerequisites
Both machines must be running Ubuntu 24.04 and have Python 3.10+ installed. Ensure you have the `stretch_body` ecosystem installed and configured, particularly `HELLO_FLEET_PATH` and `HELLO_FLEET_ID` environment variables. 

### Other Dependencies

Please clone and install the following dependencies in the virtual environment associated with this package:

- `stretch4_flying_gripper`: https://github.com/hello-robot/stretch4_flying_gripper

### Installation Steps

1. Clone the repository to your machine:
   ```bash
   git clone <repository_url>
   cd stretch4_gripper_modeling_and_control
   ```

2. Activate the appropriate Python virtual environment:
   - **On the desktop:** You should install this into the existing `stretch4_grasping_demo` virtual environment:
     ```bash
     source ~/repos/stretch4_grasping_demo/.venv/bin/activate
     ```
   - **On the robot:** If you are using a virtual environment for your Stretch software, activate it. Otherwise, if creating a new one:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```
   *(Note: This automatically installs all required dependencies like `numpy`, `opencv-python`, `scipy`, `pyyaml`, and `pyzmq`.)*

### Network Configuration

If you plan to use this code with a remote machine connected to the robot, you need to configure the network settings in `src/stretch4_gripper_modeling_and_control/gripper_networking.py`. `robot_ip` should be set to the robot's IP address and `remote_computer_ip` should be set to the remote computer's IP address.

The two variables to change are at the top of the file, as shown in the following excerpt:

```bash
# Set these values for your network
robot_ip = '100.90.83.97'
remote_computer_ip = '100.69.89.24'  # if using a remote desktop computer
```


## Calibrate the Kinematic Model

Because the flexible fingers on the gripper deform non-linearly, this repository utilizes a data-driven approach to fit a 3D kinematic B-Spline model.

Prior to running the scripts below, the robot needs to be homed. You can home the robot by running the following command in a terminal:

```bash
stretch_robot_home
```

### 1. Data Collection
On the robot, first run the following code to send high-resolution images and synchronized joint states:
```bash
python3 send_gripper_images_and_joint_states.py --resolution 800
```
Then, in a separate terminal, run the data collection script:
```bash
python3 capture_gripper_data.py
```
This automatically exercises the gripper through its full range of motion while recording synchronized joint states and ArUco visual estimations. It generates a timestamped `calib_dir` containing the raw YAML data.

### 2. Fit the Kinematic Model
Once data is collected, fit the kinematic model to the data that you just collected:
```bash
python3 fit_fingertip_model.py <calib_dir>
```
*Motivation:* This script uses robust RANSAC to fit a generic plane to the fingertip trajectories, then uses univariate splines to smoothly map the actuator's kinemaic state (i.e., `pos_pct`) to kinematic predictions of the fingertips' frames of reference. It saves the resulting model as `latest_model_planar.yaml` in your robot's `$HELLO_FLEET_PATH/$HELLO_FLEET_ID/calibration_gripper/` directory.

### 3. Fit Mirror Transforms
To robustly predict one fingertip from the other (in case one is occluded), the following code estimates mirror transforms that predict the right fingertip's frame from the left's and vice versa.
```bash
python3 fit_mirror_transforms.py
```
*Motivation:* This computes the SE(3) mirror symmetries of the gripper mechanism. It saves `latest_mirror_transforms.yaml` to the same calibration directory.

### 4. Inspect the Results
At this point, it is a good idea to visually inspect the results of the kinematic calibration. 

First, use ReRun to visualize the 3D kinematic model via the following command. The visualization shows the camera's frame of reference, the plane in which the fingers move, and the suction cup fingertips with their frames of reference.  

```bash
python3 visualize_fingertip_model.py
```

Second, use the following command to generate a video overlaying the kinematic model estimates onto the gripper images that were used for calibration. The rendered suction cups in blue should closely match the appearance of the actual suction cups in the images. The resulting video can be found within the calibration directory.

```bash
python3 create_model_video.py
```

If the calibration results do not appear to be accurate, you can use the following command to generate a video that visualizes the visually-estimated fingertip frames of reference used for calibration. The resulting video can be found within the calibration directory. If the suction cups in this video do not appear to be correctly positioned, you should re-run the calibration procedure. For example, poor lighting can result in poor fingertip estimation based on the ArUco markers, leading to a poor kinematic model.  

```bash
python3 create_calibration_video.py <calib_dir>
```

### 5. Syncing to a Desktop
To use these calibration files on your offboard desktop, simply run the provided sync tool on your **desktop**:

```bash
python3 sync_calibration_models.py
```

This tool automatically connects to the robot using the IP specified in `gripper_networking.py`, downloads the calibration files in a single `scp` command (minimizing password prompts), and extracts the correct `HELLO_FLEET_ID` directly from the downloaded data. It then automatically organizes the models into your desktop's local `~/stretch_user/<robot_id>/calibration_gripper` directory, keeping both machines in sync.

---

## Visualizing Fingertips

To see the real-time visual estimates of the fingertips and the rendered suction cups overlaid on the camera feed, you can use the following scripts.

1. **`recv_and_detect_fingertips.py`**:
   This script receives live images from the robot, runs the ArUco marker detection, and draws the visually-estimated fingertip frames of reference on the images.

2. **`visualize_fingertip_depth_range.py`**:
   This script can visualize the visually-estimated fingertip frames, the kinematically-estimated fingertip frames, and the swept volume of the fingertips associated with closing the gripper. While running this visualization, move an object into the gripper as though the gripper were about to grasp it. The pixels on the object's surface that fall within the swept volume of the fingertips should be highlighted in the video.

---

## Estimating Fingertip Loads

In one terminal, the following command should be run:

```bash
python3 send_gripper_images_and_joint_states.py
```

Then in another terminal, the user can run one of the following commands:

```bash
python3 estimate_fingertip_loads.py
```

or:

```bash
python3 estimate_fingertip_loads.py --mode cross_hud
```

Both of these commands visualize how the kinematically predicted fingertip poses differ from the visually estimated fingertip poses. Because the fingers are compliant, these differences correspond with the loads applied to the gripper fingers. Running the script without any command line arguments visualizes differences related to the applied grip force. Running the script with `--mode cross_hud` visualizes quantities associated with other types of loads applied to the finger tips, such as when the gripper makes contact with horizontal or vertical surfaces. 

---

## Teleoperation

You can teleoperate the robot's gripper using an Xbox-style gamepad connected to your remote desktop. This relies on the control logic provided by the `stretch4_flying_gripper_control` package.

### Local Teleoperation

1. In **Terminal 1** (on the robot):
   Start the command receiver to listen for incoming UDP/ZMQ joint targets:
   ```bash
   python3 recv_and_execute_gripper_commands.py
   ```

2. In **Terminal 2** (on the robot):
   Run the gamepad mapping script to broadcast your controller inputs:
   ```bash
   python3 send_gripper_commands.py
   ```

### Remote Teleoperation

1. On the **Robot**:
   Start the command receiver to listen for incoming UDP/ZMQ joint targets:
   ```bash
   python3 recv_and_execute_gripper_commands.py --remote
   ```

2. On the **Desktop**:
   Run the gamepad mapping script to broadcast your controller inputs:
   ```bash
   python3 send_gripper_commands.py --remote
   ```
   *Motivation:* This enables you to test that the desktop can control the gripper with acceptable latency and fidelity.

---

## Synergy with Grasping Demo

This repository is designed to run in conjunction with code in the `stretch4_grasping_demo` repository. The grasping demo relies on the network streams and calibration models generated here. The two repositories are packaged in such a way that you can execute a full remote grasping pipeline by running the following commands across three terminal windows.

**ON YOUR DESKTOP MACHINE**
*(Ensure you are in the `stretch4_gripper_modeling_and_control` repository)*

```bash
python3 sync_calibration_models.py
```   

**ON THE ROBOT: TERMINAL #1**
*(Starts listening for gripper commands)*
```bash
python3 recv_and_execute_gripper_commands.py --remote
```

**ON THE ROBOT: TERMINAL #2**
*(Starts broadcasting the camera feed and joint telemetry)*
```bash
python3 send_gripper_images_and_joint_states.py --remote
```

**ON THE DESKTOP IN THE stretch4_grasping_demo REPOSITORY**
*(Sync the latest calibration models from the robot first!)*
```bash
# On your desktop, within this repository:
python3 sync_calibration_models.py

# Then, switch to your grasping demo repository:
source ~/repos/stretch4_grasping_demo/.venv/bin/activate
python3 visually_servo_gripper.py --model ~/stretch_user/stretch-se4-4010/calibration_gripper/latest_model_planar.yaml --grasp_estimator ellipsoid_2d --remote OBJECT_DESCRIPTION
```
*(Note: Adjust the model path to match your specific `HELLO_FLEET_ID` as synced).*
