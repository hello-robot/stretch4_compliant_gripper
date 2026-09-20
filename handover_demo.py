#!/usr/bin/env python3
"""
handover_demo.py

Finite State Machine (FSM) demo for Stretch 4 robot object handover:
1. INITIALIZE: Connects ZMQ command publisher and telemetry subscriber.
2. OPEN_GRIPPER: Commands gripper to open wide (pos_pct) and pauses for user to place object.
3. GRASP_OBJECT: User places object in open gripper; robot closes gripper until
   Ap Diff (mm) < -20.00 mm (matching estimate_fingertip_loads.py --mode magnitude).
4. RETRACT_ARM: Fully retracts the robot's telescoping arm.
5. EXTEND_ARM: Extends the telescoping arm by 30cm (0.30 m).
6. MONITOR_LOAD: Monitors fingertip load using ArUco markers and calibrated model.
   When both left and right Z+ upward loads exceed 3.0 mm, triggers release.
7. RELEASE_OBJECT: Opens the gripper to release the object to the person.
8. COMPLETED: Handover completed; robot holds stationary.

Parameters and tuning:
All default thresholds and parameters are defined in hand_over_demo_config.py.
"""

import argparse
import collections
import enum
import os
import sys
import time
import cv2
import numpy as np
import zmq

import hand_over_demo_config as cfg
from stretch4_gripper_modeling_and_control import gripper_networking as gn
from stretch4_gripper_modeling_and_control import gripper_camera as gc
from stretch4_gripper_modeling_and_control import calibration_utils as cu
from stretch4_gripper_modeling_and_control import visualization_utils as vu
from stretch4_gripper_modeling_and_control.fingertip_visualizer import FingertipVisualizer
from recv_and_detect_fingertips import add_fingertip_detector_args, process_fingertip_detector_args


class HandoverState(enum.Enum):
    INITIALIZE = "INITIALIZE"
    OPEN_GRIPPER = "OPEN_GRIPPER"
    GRASP_OBJECT = "GRASP_OBJECT"
    RETRACT_ARM = "RETRACT_ARM"
    EXTEND_ARM = "EXTEND_ARM"
    MONITOR_LOAD = "MONITOR_LOAD"
    RELEASE_OBJECT = "RELEASE_OBJECT"
    COMPLETED = "COMPLETED"


def compute_fingertip_displacements(predicted_fingertips, vis_fingertips, visualizer):
    """
    Computes fingertip displacements along finger plane coordinates (X, Y, Z normal)
    matching cross_hud mode, and aperture difference (Ap Diff) matching magnitude mode
    in estimate_fingertip_loads.py.
    """
    normal = visualizer.normal.copy()
    centroid = visualizer.centroid.copy()
    if np.dot(-centroid, normal) < 0:
        normal = -normal
    z_axis = normal

    cam_z = np.array([0.0, 0.0, 1.0])
    x_axis_unnorm = cam_z - np.dot(cam_z, z_axis) * z_axis
    if np.linalg.norm(x_axis_unnorm) > 1e-6:
        x_axis = x_axis_unnorm / np.linalg.norm(x_axis_unnorm)
    else:
        cam_y = np.array([0.0, 1.0, 0.0])
        y_proj = cam_y - np.dot(cam_y, z_axis) * z_axis
        x_axis = y_proj / np.linalg.norm(y_proj)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    loads = {
        'left': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'detected': False},
        'right': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'detected': False}
    }

    for side in ['left', 'right']:
        if side in predicted_fingertips and side in vis_fingertips:
            f_pred = predicted_fingertips[side]
            f_vis = vis_fingertips[side]
            E = (f_vis['pos'] - f_pred['pos']) * 1000.0  # mm displacement

            dx = float(np.dot(E, x_axis))
            dy = float(np.dot(E, -y_axis))
            dz = float(np.dot(E, z_axis))

            # Rotation around normal (z_axis)
            R_pred = np.column_stack((f_pred['x_axis'], f_pred['y_axis'], f_pred['z_axis']))
            R_vis = np.column_stack((f_vis['x_axis'], f_vis['y_axis'], f_vis['z_axis']))
            R_diff = R_vis @ R_pred.T
            rvec, _ = cv2.Rodrigues(R_diff)
            twist_z_rad = np.dot(rvec.flatten(), z_axis)
            twist_z_deg = float(np.degrees(twist_z_rad))

            rot_val = twist_z_deg if side == 'left' else -twist_z_deg
            loads[side] = {
                'x': rot_val,  # RotZ (degrees)
                'y': dy,       # Y displacement (mm)
                'z': dz,       # Z displacement (mm) - upward load
                'detected': True
            }

    # Aperture Difference (mm): ap_pred - ap_vis
    aperture_diff = None
    if ('left' in predicted_fingertips and 'right' in predicted_fingertips and
        'left' in vis_fingertips and 'right' in vis_fingertips):
        ap_pred = float(np.linalg.norm(predicted_fingertips['left']['pos'] - predicted_fingertips['right']['pos']) * 1000.0)
        ap_vis = float(np.linalg.norm(vis_fingertips['left']['pos'] - vis_fingertips['right']['pos']) * 1000.0)
        aperture_diff = ap_pred - ap_vis

    return loads, z_axis, x_axis, y_axis, aperture_diff


def draw_hud(image, loads, aperture_diff, fsm, current_gripper_pct,
             load_threshold, grasp_threshold, vis_scale=1.0):
    """
    Renders cross HUD gauges, Aperture Diff bar, and FSM status banner on the OpenCV window.
    """
    h, w, _ = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    w_bar = 20
    gap = 15
    max_len = 65

    ppu_z = (max_len / 10.0) * vis_scale
    ppu_y = (max_len / 20.0) * vis_scale
    ppu_rot = (max_len / 5.0) * vis_scale

    state = fsm.state
    arm_pos = fsm.last_arm_pos
    target_pos = fsm.target_arm_pos

    # Top-side HUD background panels for gauges
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (320, 250), (0, 0, 0), -1)
    cv2.rectangle(overlay, (w - 320, 0), (w, 250), (0, 0, 0), -1)

    # Top-center FSM Banner
    cv2.rectangle(overlay, (325, 0), (w - 325, 125), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

    def draw_hud_cross(val_z, val_y, val_rot, cx, cy, x_fwd_offset, label_prefix, detected):
        if not detected:
            cv2.putText(image, f"{label_prefix}: Not Detected", (cx - 50, cy), font, font_scale, (100, 100, 100), 1, cv2.LINE_AA)
            return

        # Z (up/down) Vertical
        bz = min(int(abs(val_z) * ppu_z), max_len)
        cv2.rectangle(image, (cx - w_bar // 2, cy - gap - max_len), (cx + w_bar // 2, cy - gap), (100, 100, 100), -1)
        cv2.rectangle(image, (cx - w_bar // 2, cy + gap), (cx + w_bar // 2, cy + gap + max_len), (100, 100, 100), -1)
        
        # Color: green if >= threshold, magenta otherwise
        z_color = (0, 255, 0) if val_z >= load_threshold else (255, 0, 255)
        if val_z >= 0:
            cv2.rectangle(image, (cx - w_bar // 2, cy - gap - bz), (cx + w_bar // 2, cy - gap), z_color, -1)
        else:
            cv2.rectangle(image, (cx - w_bar // 2, cy + gap), (cx + w_bar // 2, cy + gap + bz), z_color, -1)

        # Y (right/left) Horizontal
        by = min(int(abs(val_y) * ppu_y), max_len)
        cv2.rectangle(image, (cx + gap, cy - w_bar // 2), (cx + gap + max_len, cy + w_bar // 2), (100, 100, 100), -1)
        cv2.rectangle(image, (cx - gap - max_len, cy - w_bar // 2), (cx - gap, cy + w_bar // 2), (100, 100, 100), -1)
        if val_y >= 0:
            cv2.rectangle(image, (cx + gap, cy - w_bar // 2), (cx + gap + by, cy + w_bar // 2), (0, 255, 0), -1)
        else:
            cv2.rectangle(image, (cx - gap - by, cy - w_bar // 2), (cx - gap, cy + w_bar // 2), (0, 255, 0), -1)

        # Rot Z (inward/outward) Vertical at xf
        xf = cx + x_fwd_offset
        bx = min(int(abs(val_rot) * ppu_rot), max_len)
        cv2.rectangle(image, (xf - w_bar // 2, cy - max_len), (xf + w_bar // 2, cy + max_len), (100, 100, 100), -1)
        if val_rot >= 0:
            cv2.rectangle(image, (xf - w_bar // 2, cy - bx), (xf + w_bar // 2, cy), (0, 255, 255), -1)
        else:
            cv2.rectangle(image, (xf - w_bar // 2, cy), (xf + w_bar // 2, cy + bx), (0, 255, 255), -1)

        # Center outline
        cv2.rectangle(image, (cx - gap, cy - gap), (cx + gap, cy + gap), (200, 200, 200), 1)
        cv2.line(image, (xf - w_bar // 2 - 5, cy), (xf + w_bar // 2 + 5, cy), (200, 200, 200), 1)

        # Labels
        cv2.putText(image, f"{label_prefix} Finger", (cx - 30, cy - gap - max_len - 15), font, font_scale + 0.1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f"Z: {val_z:+.1f}", (cx - 20, cy + gap + max_len + 15), font, font_scale, z_color, 1, cv2.LINE_AA)

        val_y_str = f"Y: {val_y:+.1f}"
        tw = cv2.getTextSize(val_y_str, font, font_scale, 1)[0][0]
        if x_fwd_offset < 0:
            cv2.putText(image, val_y_str, (cx + gap + max_len + 5, cy + 4), font, font_scale, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(image, val_y_str, (cx - gap - max_len - tw - 5, cy + 4), font, font_scale, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(image, f"RotZ:{val_rot:+.1f}dg", (xf - 35, cy + max_len + 15), font, font_scale, (0, 255, 255), 1, cv2.LINE_AA)

    # Cross Displays at Top Left/Right Corners
    draw_hud_cross(loads['left']['z'], loads['left']['y'], loads['left']['x'],
                   cx=150, cy=105, x_fwd_offset=-110, label_prefix="Left", detected=loads['left']['detected'])
    draw_hud_cross(loads['right']['z'], loads['right']['y'], loads['right']['x'],
                   cx=w - 150, cy=105, x_fwd_offset=110, label_prefix="Right", detected=loads['right']['detected'])

    # FSM State Banner Info
    state_colors = {
        HandoverState.INITIALIZE: (0, 165, 255),      # Orange
        HandoverState.OPEN_GRIPPER: (255, 191, 0),    # Amber
        HandoverState.GRASP_OBJECT: (0, 255, 255),    # Yellow
        HandoverState.RETRACT_ARM: (255, 200, 0),     # Cyan-Blue
        HandoverState.EXTEND_ARM: (255, 255, 0),      # Cyan
        HandoverState.MONITOR_LOAD: (255, 0, 255),    # Magenta
        HandoverState.RELEASE_OBJECT: (0, 255, 0),    # Green
        HandoverState.COMPLETED: (0, 255, 0)          # Green
    }
    state_color = state_colors.get(state, (255, 255, 255))

    banner_cx = w // 2
    cv2.putText(image, f"FSM: {state.value}", (banner_cx - 150, 26), font, 0.65, state_color, 2, cv2.LINE_AA)

    # Arm Extension text
    arm_str = f"Arm Ext: {arm_pos * 100.0:4.1f} cm" if arm_pos is not None else "Arm Ext: N/A"
    if target_pos is not None:
        arm_str += f" -> Target: {target_pos * 100.0:4.1f} cm"
    cv2.putText(image, arm_str, (banner_cx - 150, 52), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    # State specific text
    if state == HandoverState.OPEN_GRIPPER:
        if not fsm.gripper_opened:
            cur_pct_str = f"{current_gripper_pct:.0f}%" if current_gripper_pct is not None else "..."
            open_text = f"Opening gripper to {fsm.args.open_gripper_target_pct:.0f}% (current: {cur_pct_str})..."
            cv2.putText(image, open_text, (banner_cx - 150, 78), font, 0.45, (255, 191, 0), 1, cv2.LINE_AA)
        else:
            rem = max(0.0, fsm.args.open_gripper_wait - (time.time() - fsm.open_reached_time))
            open_text = f"Place object now! Closing in: {rem:3.1f}s"
            cv2.putText(image, open_text, (banner_cx - 150, 78), font, 0.45, (0, 255, 0), 2, cv2.LINE_AA)
    elif aperture_diff is not None:
        ap_grasped = (aperture_diff < grasp_threshold)
        ap_color = (0, 255, 0) if ap_grasped else (0, 165, 255)
        ap_text = f"Ap Diff: {aperture_diff:+.1f} mm (Target < {grasp_threshold:.1f} mm)"
        cv2.putText(image, ap_text, (banner_cx - 150, 78), font, 0.45, ap_color, 1 if not ap_grasped else 2, cv2.LINE_AA)
    else:
        cv2.putText(image, "Ap Diff: Detecting markers...", (banner_cx - 150, 78), font, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

    # Load readings display (for MONITOR_LOAD)
    l_z = loads['left']['z']
    r_z = loads['right']['z']
    load_cond = (loads['left']['detected'] and loads['right']['detected'] and l_z > load_threshold and r_z > load_threshold)
    load_color = (0, 255, 0) if load_cond else (200, 200, 200)
    load_text = f"Load (Z+): L={l_z:+.1f}, R={r_z:+.1f} (Threshold > {load_threshold:.1f})"
    cv2.putText(image, load_text, (banner_cx - 150, 104), font, 0.45, load_color, 1 if not load_cond else 2, cv2.LINE_AA)


class HandoverFSM:
    def __init__(self, cmd_socket, args):
        self.cmd_socket = cmd_socket
        self.args = args
        self.state = HandoverState.INITIALIZE
        self.state_start_time = time.time()

        # Open gripper state variables
        self.gripper_opened = False
        self.open_reached_time = None

        # Arm state variables
        self.retracted_arm_pos = None
        self.target_arm_pos = None
        self.last_arm_pos = None
        self.stall_counter = 0

        # Grasp state variables
        self.consecutive_grasp_triggers = 0
        self.locked_gripper_pos = None
        self.grasp_active = False

        # Load / release variables
        self.consecutive_load_triggers = 0
        self.release_start_time = None

        print(f"\n[FSM] Initialized. Starting in state: {self.state.value}")

    def transition_to(self, new_state):
        print(f"\n[FSM] Transitioning: {self.state.value} -> {new_state.value}")
        self.state = new_state
        self.state_start_time = time.time()
        self.gripper_opened = False
        self.open_reached_time = None
        self.stall_counter = 0
        self.consecutive_grasp_triggers = 0
        self.consecutive_load_triggers = 0
        if new_state == HandoverState.RELEASE_OBJECT:
            self.release_start_time = time.time()

    def step(self, current_arm_pos, current_gripper_pct, loads, aperture_diff):
        """
        Executes one FSM update cycle.
        Returns: (arm_velocity_cmd, grip_cmd)
        """
        arm_vel_cmd = 0.0
        grip_cmd = None

        # ---------------------------------------------------------------------
        # 1. INITIALIZE: Verify robot telemetry connection
        # ---------------------------------------------------------------------
        if self.state == HandoverState.INITIALIZE:
            arm_vel_cmd = 0.0
            grip_cmd = None
            if current_arm_pos is not None and current_gripper_pct is not None:
                print(f"[FSM] Telemetry verified. Arm extension: {current_arm_pos * 100:.2f} cm, Gripper: {current_gripper_pct:.1f}%")
                self.transition_to(HandoverState.OPEN_GRIPPER)

        # ---------------------------------------------------------------------
        # 2. OPEN_GRIPPER: Open gripper and wait to give user time to position object
        # ---------------------------------------------------------------------
        elif self.state == HandoverState.OPEN_GRIPPER:
            arm_vel_cmd = 0.0
            grip_cmd = {
                'pos_pct': self.args.open_gripper_target_pct,
                'speed': self.args.open_gripper_speed,
                'accel': self.args.open_gripper_accel
            }

            now = time.time()
            elapsed_total = now - self.state_start_time

            # Check if gripper has reached the target open position
            target_reached = False
            if current_gripper_pct is not None:
                if current_gripper_pct >= (self.args.open_gripper_target_pct - self.args.open_gripper_tolerance):
                    target_reached = True

            # If target reached or safety fallback timeout exceeded, begin countdown
            if target_reached or (elapsed_total >= self.args.open_gripper_timeout):
                if not self.gripper_opened:
                    self.gripper_opened = True
                    self.open_reached_time = now
                    pct_info = f"{current_gripper_pct:.1f}%" if current_gripper_pct is not None else "timeout"
                    print(f"[FSM] Gripper reached open position ({pct_info}). "
                          f"Waiting {self.args.open_gripper_wait:.1f}s for user to place object...")

            if self.gripper_opened:
                wait_elapsed = now - self.open_reached_time
                if wait_elapsed >= self.args.open_gripper_wait:
                    print(f"[FSM] Open wait complete. Ready to grasp object!")
                    self.transition_to(HandoverState.GRASP_OBJECT)

        # ---------------------------------------------------------------------
        # 3. GRASP_OBJECT: Close gripper until Ap Diff (mm) < threshold (-20 mm)
        # ---------------------------------------------------------------------
        elif self.state == HandoverState.GRASP_OBJECT:
            arm_vel_cmd = 0.0

            # Optional pre-wait before closing
            elapsed = time.time() - self.state_start_time
            if elapsed < self.args.grasp_pre_wait:
                grip_cmd = None
                return arm_vel_cmd, grip_cmd

            # Command gripper closing towards target pos
            grip_cmd = {
                'pos_pct': self.args.grasp_target_pct,
                'speed': self.args.grasp_speed,
                'accel': self.args.grasp_accel
            }

            # Check if Ap Diff (mm) < threshold (e.g. -20.00 mm)
            if aperture_diff is not None and aperture_diff < self.args.grasp_threshold:
                self.consecutive_grasp_triggers += 1
                if self.consecutive_grasp_triggers >= self.args.grasp_consecutive_frames:
                    print(f"\n[FSM] Object grasped firmly! Ap Diff = {aperture_diff:.2f} mm (< {self.args.grasp_threshold:.2f} mm)")
                    self.locked_gripper_pos = current_gripper_pct if current_gripper_pct is not None else self.args.grasp_target_pct
                    # Lock gripper in place using relative displacement on the robot's live position.
                    # This prevents the gripper from opening backwards due to telemetry latency,
                    # and maintains preloaded contact squeeze.
                    grip_cmd = {
                        'pos_pct_disp': float(self.args.grasp_hold_disp_pct),
                        'speed': float(self.args.grasp_hold_speed),
                        'accel': float(self.args.grasp_hold_accel)
                    }
                    self.transition_to(HandoverState.RETRACT_ARM)
            else:
                self.consecutive_grasp_triggers = 0

        # ---------------------------------------------------------------------
        # 4. RETRACT_ARM: Retract arm fully into robot body
        # ---------------------------------------------------------------------
        elif self.state == HandoverState.RETRACT_ARM:
            grip_cmd = None
            if current_arm_pos is None:
                arm_vel_cmd = 0.0
                return arm_vel_cmd, grip_cmd

            # Command arm retraction
            arm_vel_cmd = -abs(self.args.retract_speed)

            # Check stall against inner mechanical stop
            if self.last_arm_pos is not None and abs(current_arm_pos - self.last_arm_pos) < self.args.stall_motion_thresh:
                self.stall_counter += 1
            else:
                self.stall_counter = 0

            # Fully retracted condition
            if current_arm_pos <= self.args.retract_tolerance or (self.stall_counter >= self.args.stall_frames and current_arm_pos <= 0.03):
                self.retracted_arm_pos = current_arm_pos
                self.target_arm_pos = self.retracted_arm_pos + self.args.extension
                print(f"[FSM] Arm fully retracted at {self.retracted_arm_pos * 100:.2f} cm. "
                      f"Target extension: {self.target_arm_pos * 100:.2f} cm")
                arm_vel_cmd = 0.0
                self.transition_to(HandoverState.EXTEND_ARM)

        # ---------------------------------------------------------------------
        # 5. EXTEND_ARM: Extend arm forward by +30 cm (0.30 m)
        # ---------------------------------------------------------------------
        elif self.state == HandoverState.EXTEND_ARM:
            grip_cmd = None
            if current_arm_pos is None or self.target_arm_pos is None:
                arm_vel_cmd = 0.0
                return arm_vel_cmd, grip_cmd

            dist_remaining = self.target_arm_pos - current_arm_pos

            if dist_remaining <= self.args.extend_tolerance:
                print(f"[FSM] Arm extended to target: {current_arm_pos * 100:.2f} cm (+{self.args.extension * 100:.1f} cm)")
                arm_vel_cmd = 0.0
                self.transition_to(HandoverState.MONITOR_LOAD)
            else:
                # Proportional velocity scaling
                arm_vel_cmd = float(np.clip(self.args.extend_kp * dist_remaining,
                                            self.args.extend_min_speed,
                                            abs(self.args.extend_speed)))

        # ---------------------------------------------------------------------
        # 6. MONITOR_LOAD: Wait for user upward pull (Z+ > 3.0 on both fingers)
        # ---------------------------------------------------------------------
        elif self.state == HandoverState.MONITOR_LOAD:
            arm_vel_cmd = 0.0
            grip_cmd = None

            left_detected = loads['left']['detected']
            right_detected = loads['right']['detected']
            left_z = loads['left']['z']
            right_z = loads['right']['z']

            if left_detected and right_detected:
                if left_z > self.args.left_z_threshold and right_z > self.args.right_z_threshold:
                    self.consecutive_load_triggers += 1
                    if self.consecutive_load_triggers >= self.args.load_consecutive_frames:
                        print(f"\n[FSM] Upward load detected! L_Z={left_z:.2f}, R_Z={right_z:.2f} "
                              f"(Thresholds > {self.args.left_z_threshold:.1f}, {self.args.right_z_threshold:.1f}). Releasing object!")
                        self.transition_to(HandoverState.RELEASE_OBJECT)
                        self.release_start_time = time.time()
                else:
                    self.consecutive_load_triggers = 0
            else:
                self.consecutive_load_triggers = 0

        # ---------------------------------------------------------------------
        # 7. RELEASE_OBJECT: Open gripper to release object to person
        # ---------------------------------------------------------------------
        elif self.state == HandoverState.RELEASE_OBJECT:
            arm_vel_cmd = 0.0
            if time.time() - self.release_start_time > self.args.release_duration:
                print("[FSM] Gripper opened. Handover completed successfully!")
                self.transition_to(HandoverState.COMPLETED)
                grip_cmd = None
            else:
                grip_cmd = {
                    'pos_pct': self.args.release_target_pct,
                    'speed': self.args.release_speed,
                    'accel': self.args.release_accel
                }

        # ---------------------------------------------------------------------
        # 8. COMPLETED: Handover finished
        # ---------------------------------------------------------------------
        elif self.state == HandoverState.COMPLETED:
            arm_vel_cmd = 0.0
            grip_cmd = None

        self.last_arm_pos = current_arm_pos
        return arm_vel_cmd, grip_cmd


def send_robot_command(socket, arm_vel, grip_cmd):
    """Sends control command dictionary to recv_and_execute_gripper_commands.py via ZMQ PUB."""
    output_dict = {
        'control_mode': 3,
        'joint_velocity_commands': {
            'arm': arm_vel,
            'base_x': 0.0,
            'base_y': 0.0,
            'base_theta': 0.0,
            'lift': 0.0,
            'wrist_yaw': 0.0,
            'wrist_pitch': 0.0,
            'wrist_roll': 0.0
        },
        'grip': grip_cmd
    }
    socket.send_pyobj(output_dict)


def main():
    parser = argparse.ArgumentParser(
        description='Stretch 4 Object Handover Demo: open gripper, grasp object (Ap Diff < -20 mm), retract arm, extend 30cm, monitor load (Z+ > 3.0), and release.'
    )
    # Model & Networking
    parser.add_argument('--model', type=str, default=cfg.DEFAULT_MODEL_PATH,
                        help='Path to the fingertip kinematic model YAML file. If not provided, defaults to latest fleet calibration model.')
    parser.add_argument('-r', '--remote', action='store_true',
                        help='Run remotely: connect to robot on port 4409 and bind publisher on port 4407.')

    # Open Gripper parameters
    parser.add_argument('--open_gripper_target_pct', type=float, default=cfg.OPEN_GRIPPER_TARGET_POS_PCT,
                        help=f'Target pos_pct to open gripper in OPEN_GRIPPER state (default: {cfg.OPEN_GRIPPER_TARGET_POS_PCT}%%).')
    parser.add_argument('--open_gripper_speed', type=float, default=cfg.OPEN_GRIPPER_SPEED,
                        help=f'Gripper opening speed in %%/s (default: {cfg.OPEN_GRIPPER_SPEED}%%/s).')
    parser.add_argument('--open_gripper_accel', type=float, default=cfg.OPEN_GRIPPER_ACCEL,
                        help=f'Gripper opening accel in %%/s^2 (default: {cfg.OPEN_GRIPPER_ACCEL}%%/s^2).')
    parser.add_argument('--open_gripper_tolerance', type=float, default=cfg.OPEN_GRIPPER_TOLERANCE_PCT,
                        help=f'Arrival tolerance pos_pct for open position (default: {cfg.OPEN_GRIPPER_TOLERANCE_PCT}%%).')
    parser.add_argument('--open_gripper_wait', type=float, default=cfg.OPEN_GRIPPER_WAIT_SECONDS,
                        help=f'Wait duration in seconds in OPEN_GRIPPER state after opening (default: {cfg.OPEN_GRIPPER_WAIT_SECONDS} s).')
    parser.add_argument('--open_gripper_timeout', type=float, default=cfg.OPEN_GRIPPER_MAX_TIMEOUT_SECONDS,
                        help=f'Maximum fallback timeout in seconds in OPEN_GRIPPER (default: {cfg.OPEN_GRIPPER_MAX_TIMEOUT_SECONDS} s).')

    # Grasping parameters
    parser.add_argument('--grasp_threshold', type=float, default=cfg.GRASP_APERTURE_DIFF_THRESHOLD,
                        help=f'Aperture difference (Ap Diff) threshold in mm to detect object contact (default: {cfg.GRASP_APERTURE_DIFF_THRESHOLD}).')
    parser.add_argument('--grasp_speed', type=float, default=cfg.GRASP_CLOSE_SPEED,
                        help=f'Gripper closing speed in %%/s (default: {cfg.GRASP_CLOSE_SPEED}).')
    parser.add_argument('--grasp_accel', type=float, default=cfg.GRASP_CLOSE_ACCEL,
                        help=f'Gripper closing accel in %%/s^2 (default: {cfg.GRASP_CLOSE_ACCEL}).')
    parser.add_argument('--grasp_target_pct', type=float, default=cfg.GRASP_TARGET_POS_PCT,
                        help=f'Target closing pos_pct (default: {cfg.GRASP_TARGET_POS_PCT}).')
    parser.add_argument('--grasp_consecutive_frames', type=int, default=cfg.GRASP_CONSECUTIVE_FRAMES,
                        help=f'Consecutive frames required for grasp confirmation (default: {cfg.GRASP_CONSECUTIVE_FRAMES}).')
    parser.add_argument('--grasp_pre_wait', type=float, default=cfg.GRASP_PRE_WAIT_SECONDS,
                        help=f'Pre-grasp wait time in seconds (default: {cfg.GRASP_PRE_WAIT_SECONDS}).')
    parser.add_argument('--grasp_hold_disp_pct', type=float, default=cfg.GRASP_HOLD_DISPLACEMENT_PCT,
                        help=f'Gripper hold displacement (pos_pct_disp) upon grasp (default: {cfg.GRASP_HOLD_DISPLACEMENT_PCT}%%).')
    parser.add_argument('--grasp_hold_speed', type=float, default=cfg.GRASP_HOLD_SPEED,
                        help=f'Gripper hold speed in %%/s (default: {cfg.GRASP_HOLD_SPEED}).')
    parser.add_argument('--grasp_hold_accel', type=float, default=cfg.GRASP_HOLD_ACCEL,
                        help=f'Gripper hold accel in %%/s^2 (default: {cfg.GRASP_HOLD_ACCEL}).')

    # Arm Retraction & Extension
    parser.add_argument('--retract_speed', type=float, default=cfg.ARM_RETRACT_SPEED,
                        help=f'Arm retraction velocity command in [0, 1] (default: {cfg.ARM_RETRACT_SPEED}).')
    parser.add_argument('--retract_tolerance', type=float, default=cfg.ARM_RETRACT_TOLERANCE_M,
                        help=f'Arm retraction arrival tolerance in meters (default: {cfg.ARM_RETRACT_TOLERANCE_M}).')
    parser.add_argument('--stall_frames', type=int, default=cfg.ARM_STALL_CONSECUTIVE_FRAMES,
                        help=f'Stall detection consecutive frames (default: {cfg.ARM_STALL_CONSECUTIVE_FRAMES}).')
    parser.add_argument('--stall_motion_thresh', type=float, default=cfg.ARM_STALL_MOTION_THRESHOLD_M,
                        help=f'Stall position motion threshold in meters (default: {cfg.ARM_STALL_MOTION_THRESHOLD_M}).')

    parser.add_argument('--extension', type=float, default=cfg.ARM_EXTENSION_DISTANCE_M,
                        help=f'Arm extension distance in meters (default: {cfg.ARM_EXTENSION_DISTANCE_M} m = 30 cm).')
    parser.add_argument('--extend_speed', type=float, default=cfg.ARM_EXTEND_SPEED,
                        help=f'Arm extension velocity command in [0, 1] (default: {cfg.ARM_EXTEND_SPEED}).')
    parser.add_argument('--extend_min_speed', type=float, default=cfg.ARM_EXTEND_MIN_SPEED,
                        help=f'Arm extension minimum speed (default: {cfg.ARM_EXTEND_MIN_SPEED}).')
    parser.add_argument('--extend_kp', type=float, default=cfg.ARM_EXTEND_KP,
                        help=f'Arm extension proportional gain kp (default: {cfg.ARM_EXTEND_KP}).')
    parser.add_argument('--extend_tolerance', type=float, default=cfg.ARM_EXTEND_TOLERANCE_M,
                        help=f'Arm extension arrival tolerance in meters (default: {cfg.ARM_EXTEND_TOLERANCE_M}).')

    # Load Monitoring & Handover Release
    parser.add_argument('--load_threshold', type=float, default=cfg.LOAD_THRESHOLD_Z,
                        help=f'Upward Z+ load threshold in mm to trigger release (default: {cfg.LOAD_THRESHOLD_Z}).')
    parser.add_argument('--left_z_threshold', type=float, default=cfg.LOAD_LEFT_Z_THRESHOLD,
                        help=f'Left fingertip Z+ threshold in mm (default: {cfg.LOAD_LEFT_Z_THRESHOLD}).')
    parser.add_argument('--right_z_threshold', type=float, default=cfg.LOAD_RIGHT_Z_THRESHOLD,
                        help=f'Right fingertip Z+ threshold in mm (default: {cfg.LOAD_RIGHT_Z_THRESHOLD}).')
    parser.add_argument('--load_consecutive_frames', type=int, default=cfg.LOAD_CONSECUTIVE_FRAMES,
                        help=f'Consecutive frames required for load release confirmation (default: {cfg.LOAD_CONSECUTIVE_FRAMES}).')

    # Release & Exit
    parser.add_argument('--release_target_pct', type=float, default=cfg.RELEASE_TARGET_POS_PCT,
                        help=f'Target pos_pct to release object (default: {cfg.RELEASE_TARGET_POS_PCT}%%).')
    parser.add_argument('--release_speed', type=float, default=cfg.RELEASE_SPEED,
                        help=f'Gripper release speed in %%/s (default: {cfg.RELEASE_SPEED}%%/s).')
    parser.add_argument('--release_accel', type=float, default=cfg.RELEASE_ACCEL,
                        help=f'Gripper release accel in %%/s^2 (default: {cfg.RELEASE_ACCEL}%%/s^2).')
    parser.add_argument('--release_duration', type=float, default=cfg.RELEASE_DURATION_S,
                        help=f'Time in seconds allowed for gripper to open (default: {cfg.RELEASE_DURATION_S} s).')
    parser.add_argument('--auto_exit', action='store_true', default=cfg.COMPLETED_AUTO_EXIT,
                        help='Automatically exit after handover completion.')
    parser.add_argument('--auto_exit_delay', type=float, default=cfg.COMPLETED_AUTO_EXIT_DELAY_S,
                        help=f'Delay in seconds before auto-exiting (default: {cfg.COMPLETED_AUTO_EXIT_DELAY_S} s).')

    # Vision & Display
    parser.add_argument('--interpolate', action='store_true',
                        help='Use linear interpolation for pos_pct instead of nearest neighbor.')
    parser.add_argument('--no_display', action='store_true',
                        help='Disable OpenCV display window.')

    add_fingertip_detector_args(parser)
    vu.add_suction_cup_argument(parser)
    vu.add_display_scale_argument(parser)

    args = parser.parse_args()

    # Model resolution
    if args.model is not None:
        args.model = os.path.expanduser(args.model)
    else:
        args.model = cu.get_default_model_path()
        if args.model:
            print(f"No model path provided. Defaulting to fleet calibration: {args.model}")
        else:
            print("Error: No model path provided and could not locate latest_model_planar.yaml in fleet directory.")
            sys.exit(1)

    if not os.path.exists(args.model):
        print(f"Error: Model file does not exist: {args.model}")
        sys.exit(1)

    print("Loading Fingertip Kinematic Model...")
    visualizer = FingertipVisualizer(args.model)

    print(f"Initializing Fingertip Detector... (smooth={getattr(args, 'smooth', cfg.SMOOTH_ESTIMATES)})")
    detector = process_fingertip_detector_args(args)

    # Setup ZMQ Context
    context = zmq.Context()

    # ZMQ Publisher for Commands -> recv_and_execute_gripper_commands.py
    cmd_socket = context.socket(zmq.PUB)
    cmd_socket.setsockopt(zmq.SNDHWM, 1)
    cmd_socket.setsockopt(zmq.RCVHWM, 1)

    if args.remote:
        cmd_address = f"tcp://*:{cfg.GRIPPER_CMD_PORT}"
    else:
        cmd_address = f"tcp://127.0.0.1:{cfg.GRIPPER_CMD_PORT}"
    print(f"Binding ZMQ Command Publisher to {cmd_address}")
    cmd_socket.bind(cmd_address)

    # ZMQ Subscriber for Robot Images & Joint States -> send_gripper_images_and_joint_states.py
    sub_socket = context.socket(zmq.SUB)
    sub_socket.setsockopt(zmq.SUBSCRIBE, b'')
    sub_socket.setsockopt(zmq.SNDHWM, 1)
    sub_socket.setsockopt(zmq.RCVHWM, 1)
    sub_socket.setsockopt(zmq.CONFLATE, 1)

    sub_address = f"tcp://{gn.robot_ip if args.remote else '127.0.0.1'}:{cfg.GRIPPER_TELEMETRY_PORT}"
    print(f"Connecting ZMQ Telemetry Subscriber to {sub_address}")
    sub_socket.connect(sub_address)

    gn.print_network_info()

    fsm = HandoverFSM(cmd_socket, args)

    hz = cfg.CONTROL_HZ
    dt = 1.0 / hz
    rate = time.time()

    rgb_camera_info = {
        'camera_matrix': np.array([
            [425.0, 0.0, 320.0],
            [0.0, 425.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64),
        'distortion_coefficients': np.zeros(5, dtype=np.float64)
    }

    status_history = collections.deque(maxlen=1000)

    current_arm_pos = None
    current_gripper_pct = None
    loads = {
        'left': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'detected': False},
        'right': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'detected': False}
    }
    aperture_diff = None

    window_name = "Stretch 4 Object Handover Demo"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("================================================================")
    print("Stretch 4 Object Handover Demo Running")
    print(f"Open Gripper Target: {args.open_gripper_target_pct:.1f}% (Wait after open: {args.open_gripper_wait:.1f} s)")
    print(f"Grasp Threshold: Ap Diff < {args.grasp_threshold:.2f} mm")
    print(f"Extension Distance: {args.extension * 100:.1f} cm")
    print(f"Upward Release Threshold: Z+ > {args.left_z_threshold:.1f} (Left), {args.right_z_threshold:.1f} (Right)")
    print("Press 'q' or 'Esc' to quit.")
    print("================================================================")

    try:
        while True:
            # Poll for new image & joint state packet (non-blocking)
            try:
                output_dict = sub_socket.recv_pyobj(flags=zmq.NOBLOCK)

                if 'camera_matrix' in output_dict and 'distortion_coefficients' in output_dict:
                    rgb_camera_info['camera_matrix'] = output_dict['camera_matrix']
                    rgb_camera_info['distortion_coefficients'] = output_dict['distortion_coefficients']

                if 'color_image_compressed' in output_dict:
                    color_image = cv2.imdecode(
                        np.frombuffer(output_dict['color_image_compressed'], np.uint8),
                        cv2.IMREAD_COLOR
                    )
                else:
                    color_image = output_dict.get('color_image')

                # Update status history
                for st in output_dict.get('joint_state_history', []):
                    status_history.append({'ts': st['timestamp'], 'pct': st['gripper']['pos_pct']})

                closest_joint_state = output_dict.get('closest_joint_state')
                if closest_joint_state is not None:
                    matched_pct = closest_joint_state['gripper']['pos_pct']
                    current_gripper_pct = matched_pct
                    if 'arm' in closest_joint_state and 'extension' in closest_joint_state['arm']:
                        current_arm_pos = closest_joint_state['arm']['extension']

                    if len(status_history) >= 10:
                        pcts = np.array([s['pct'] for s in status_history])
                        direction = 'opening' if (pcts[-1] - pcts[0]) > 0 else 'closing'
                    else:
                        direction = 'closing'
                else:
                    matched_pct = None
                    direction = 'closing'

                # Prefer latest arm/gripper position from joint_state_history for control & HUD
                if output_dict.get('joint_state_history'):
                    latest_state = output_dict['joint_state_history'][-1]
                    if 'arm' in latest_state and 'extension' in latest_state['arm']:
                        current_arm_pos = latest_state['arm']['extension']
                    if 'gripper' in latest_state and 'pos_pct' in latest_state['gripper']:
                        current_gripper_pct = latest_state['gripper']['pos_pct']

                # Kinematic frame prediction
                predicted_fingertips = {}
                if matched_pct is not None:
                    for side in ['left', 'right']:
                        pos_pred, rot_pred = visualizer.predict(side, matched_pct, direction)
                        if pos_pred is not None and rot_pred is not None:
                            predicted_fingertips[side] = {
                                'pos': pos_pred,
                                'x_axis': rot_pred[:, 0],
                                'y_axis': rot_pred[:, 1],
                                'z_axis': rot_pred[:, 2]
                            }

                # Vision-based fingertip tracking
                if color_image is not None:
                    vis_fingertips = detector.process_image(color_image, rgb_camera_info, pos_pct=matched_pct)

                    # Compute loads and aperture difference
                    loads, _, _, _, aperture_diff = compute_fingertip_displacements(
                        predicted_fingertips, vis_fingertips, visualizer
                    )

                    # UI Visualization
                    if not args.no_display:
                        display_img = color_image.copy()
                        display_img, scaled_camera_info = vu.apply_display_scale(
                            display_img, args.display_scale, camera_info=rgb_camera_info
                        )

                        # Draw kinematically predicted frames
                        if predicted_fingertips:
                            vu.draw_predicted_frames(predicted_fingertips, display_img, scaled_camera_info)
                            if not args.disable_suction_cups:
                                detector.aruco_to_fingertips.draw_fingertip_suction_cups(
                                    predicted_fingertips, display_img, scaled_camera_info,
                                    color=(128, 0, 0), alpha=0.4
                                )

                        # Draw visually estimated frames
                        detector.aruco_to_fingertips.draw_fingertip_frames(
                            vis_fingertips, display_img, scaled_camera_info,
                            axis_length_in_m=0.02, draw_origins=True, write_coordinates=False
                        )
                        if not args.disable_suction_cups and vis_fingertips:
                            detector.aruco_to_fingertips.draw_fingertip_suction_cups(
                                vis_fingertips, display_img, scaled_camera_info,
                                color=(255, 0, 0), alpha=0.4
                            )

                        # Draw cross HUD, Ap Diff bar, and status banner
                        draw_hud(display_img, loads, aperture_diff, fsm, current_gripper_pct,
                                 load_threshold=args.load_threshold,
                                 grasp_threshold=args.grasp_threshold,
                                 vis_scale=1.0)

                        cv2.imshow(window_name, display_img)
                        key = cv2.waitKey(1)
                        if key in (27, ord('q')):
                            print("\nUser requested exit.")
                            break

            except zmq.Again:
                pass

            # FSM Step
            arm_vel, grip_cmd = fsm.step(current_arm_pos, current_gripper_pct, loads, aperture_diff)

            # Send command to robot
            send_robot_command(cmd_socket, arm_vel, grip_cmd)

            # Auto exit check if enabled
            if fsm.state == HandoverState.COMPLETED and args.auto_exit:
                if time.time() - fsm.state_start_time > args.auto_exit_delay:
                    print("Auto-exit triggered.")
                    break

            # Sleep to maintain 30 Hz loop rate
            sleep_time = rate + dt - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            rate = time.time()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("Stopping robot and cleaning up...")
        # Send zero-velocity stop commands
        send_robot_command(cmd_socket, 0.0, None)
        time.sleep(0.05)
        send_robot_command(cmd_socket, 0.0, None)

        if not args.no_display:
            cv2.destroyAllWindows()
        cmd_socket.close()
        sub_socket.close()
        context.term()
        print("Done.")


if __name__ == '__main__':
    main()
