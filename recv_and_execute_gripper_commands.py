#!/usr/bin/env python3
import time
import argparse
import sys
import numpy as np
import zmq

import stretch4_body.robot.robot_client as rc
import stretch4_body.robot.robot as rb
from stretch4_body.core.robot_params import RobotParams

from stretch4_flying_gripper.kinematic_controller import KinematicController
from stretch4_gripper_modeling_and_control import gripper_networking as gn
from stretch4_flying_gripper import teleop_config

def main():
    parser = teleop_config.get_base_parser('Receive and Execute Gripper Commands for Stretch')
    parser.add_argument('-r', '--remote', action='store_true', help='Use this argument when running the code on a remote computer. Configure gripper_networking.py first.') 
    parser.add_argument('--disable_flipped_wrist', action='store_true', help='Disable the flipped wrist configuration.')
    args = parser.parse_args()

    robot, ikin, accel_vel_dict = teleop_config.initialize_teleop_hardware(args)

    accel_base_xy = accel_vel_dict['accel_base_xy']
    accel_base_w = accel_vel_dict['accel_base_w']
    accel_lift = accel_vel_dict['accel_lift']
    accel_arm = accel_vel_dict['accel_arm']
    accel_yaw = accel_vel_dict['accel_yaw']
    accel_pitch = accel_vel_dict['accel_pitch']
    accel_roll = accel_vel_dict['accel_roll']
    
    vel_yaw = accel_vel_dict['vel_yaw']
    vel_pitch = accel_vel_dict['vel_pitch']
    vel_roll = accel_vel_dict['vel_roll']

    vel_grip = accel_vel_dict['vel_grip']
    acc_grip = accel_vel_dict['acc_grip']
    
    gamepad_speed_trans = accel_vel_dict['gamepad_speed_trans']
    gamepad_speed_rot = accel_vel_dict['gamepad_speed_rot']
        
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1)
    
    if args.remote:
        address = 'tcp://' + gn.remote_computer_ip + ':' + str(gn.gripper_cmd_port)
    else:
        address = 'tcp://127.0.0.1:' + str(gn.gripper_cmd_port)
        
    print(f"Connecting ZMQ Subscriber to {address}")
    socket.connect(address)
    
    gn.print_network_info()
    
    control_mode = 1
    hz = 30.0
    dt = 1.0 / hz
    rate = time.time()
    
    gripper_close_pct = -60.0
    gripper_open_pct = 60.0
    
    # Store time to implement safety timeout
    last_cmd_time = time.time()
    timeout_duration = 2.0 # Halt robot if no commands for 2.0s
    
    print("====================================")
    print("Receiver Started. Mode: Gripper Frame Relative")
    print("Ctrl+C to Quit")
    print("====================================")

    try:
        import pinocchio as pin
        cmd = None
        
        last_move_to_targets = {
            'lift': None,
            'wrist_roll': None,
            'stretch_gripper': None
        }

        while True:
            # Poll for commands
            try:
                # Use zmq.NOBLOCK, if not available raises zmq.error.Again
                while True:
                    new_cmd = socket.recv_pyobj(flags=zmq.NOBLOCK)
                    cmd = new_cmd
                    last_cmd_time = time.time()
            except zmq.error.Again:
                pass
                
            # Safety timeout check
            if time.time() - last_cmd_time > timeout_duration:
                # Zero out velocities
                cmd = {
                    'v_desired': [0.0, 0.0, 0.0],
                    'rot_change': [0.0, 0.0, 0.0],
                    'grip': None
                }
                last_move_to_targets = {
                    'lift': None,
                    'wrist_roll': None,
                    'stretch_gripper': None
                }
            
            if not args.direct:
                robot.pull_status()
                
            # # Sync IK configuration with real robot
            # pitch_sign_mult = 1.0 if args.disable_flipped_wrist else -1.0
            # roll_sign_mult = 1.0 if args.disable_flipped_wrist else -1.0
            pitch_sign_mult = 1.0
            roll_sign_mult = -1.0
            
            ikin.q[0] = robot.base.status['x']
            ikin.q[1] = robot.base.status['y']
            ikin.q[2] = np.cos(robot.base.status['theta'])
            ikin.q[3] = np.sin(robot.base.status['theta'])
            ikin.q[4] = robot.lift.status['pos']
            ikin.q[5] = robot.arm.status['pos']
            ikin.q[6] = robot.end_of_arm.status['wrist_yaw']['pos']
            ikin.q[7] = robot.end_of_arm.status['wrist_pitch']['pos'] * pitch_sign_mult
            ikin.q[8] = robot.end_of_arm.status['wrist_roll']['pos'] * roll_sign_mult
            
            pin.forwardKinematics(ikin.model, ikin.data, ikin.q)
            pin.updateFramePlacements(ikin.model, ikin.data)
            
            if cmd is None:
                robot.base.set_velocity(0, 0, 0, accel_base_xy*2, accel_base_w*2)
                robot.lift.set_velocity(0, a_m=accel_lift)
                robot.arm.set_velocity(0, a_m=accel_arm)
                robot.end_of_arm.move_by('wrist_yaw', 0)
                robot.end_of_arm.move_by('wrist_pitch', 0)
                robot.end_of_arm.move_by('wrist_roll', 0)
                robot.push_command()
                
                sleep_time = rate + dt - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                rate = time.time()
                continue
            
            grip_val = cmd.get('grip')
            if isinstance(grip_val, dict):
                g_disp = grip_val.get('pos_pct_disp')
                g_pos = grip_val.get('pos_pct')
                g_action = grip_val.get('action')
                g_vel = grip_val.get('speed', vel_grip)
                g_acc = grip_val.get('accel', acc_grip)
                
                if g_disp is not None:
                    robot.end_of_arm.move_by('stretch_gripper', g_disp, g_vel, g_acc)
                    last_move_to_targets['stretch_gripper'] = None
                elif g_pos is not None:
                    if last_move_to_targets['stretch_gripper'] != g_pos:
                        robot.end_of_arm.move_to('stretch_gripper', g_pos, g_vel, g_acc)
                        last_move_to_targets['stretch_gripper'] = g_pos
                elif g_action == "OPEN":
                    robot.end_of_arm.move_by('stretch_gripper', gripper_open_pct, g_vel, g_acc)
                    last_move_to_targets['stretch_gripper'] = None
                elif g_action == "CLOSE":
                    robot.end_of_arm.move_by('stretch_gripper', gripper_close_pct, g_vel, g_acc)
                    last_move_to_targets['stretch_gripper'] = None
                cmd['grip'] = None
            elif grip_val == "OPEN":
                robot.end_of_arm.move_by('stretch_gripper', gripper_open_pct, vel_grip, acc_grip)
                last_move_to_targets['stretch_gripper'] = None
                cmd['grip'] = None
            elif grip_val == "CLOSE":
                robot.end_of_arm.move_by('stretch_gripper', gripper_close_pct, vel_grip, acc_grip)
                last_move_to_targets['stretch_gripper'] = None
                cmd['grip'] = None

            active_pos_cmds = set()
            j_pos_cmds = cmd.get('joint_position_commands')
            if j_pos_cmds is not None:
                if 'lift' in j_pos_cmds:
                    target = j_pos_cmds['lift']
                    active_pos_cmds.add('lift')
                    if last_move_to_targets['lift'] != target:
                        robot.lift.move_to(target)
                        last_move_to_targets['lift'] = target
                if 'wrist_roll' in j_pos_cmds:
                    target = j_pos_cmds['wrist_roll']
                    active_pos_cmds.add('wrist_roll')
                    if last_move_to_targets['wrist_roll'] != target:
                        robot.end_of_arm.move_to('wrist_roll', target)
                        last_move_to_targets['wrist_roll'] = target

            new_mode = cmd.get('control_mode', control_mode)
            if new_mode != control_mode:
                control_mode = new_mode
                ikin.retract_state['is_retracting'] = False
                mode_names = {
                    1: "Gripper Frame Relative (IK)",
                    2: "Projected Base Frame Relative (IK)",
                    3: "Joint-Space Direct Control",
                    4: "Base Frame Translation (IK)"
                }
                print(f"--> Receiver Switched to Mode {control_mode}: {mode_names.get(control_mode, 'Unknown')}")

            if control_mode == 3:
                if not args.use_system_speeds:
                    # Direct joint space mapping ignoring kinematics solver
                    j_cmds = cmd.get('joint_velocity_commands', {})
                    v_vel = np.zeros(8)
                    v_vel[0] = j_cmds.get('base_x', 0.0) * gamepad_speed_trans
                    v_vel[1] = j_cmds.get('base_y', 0.0) * gamepad_speed_trans
                    v_vel[2] = j_cmds.get('base_theta', 0.0) * gamepad_speed_rot
                    v_vel[3] = j_cmds.get('lift', 0.0) * gamepad_speed_trans
                    v_vel[4] = j_cmds.get('arm', 0.0) * gamepad_speed_trans
                    v_vel[5] = j_cmds.get('wrist_yaw', 0.0) * gamepad_speed_rot
                    v_vel[6] = j_cmds.get('wrist_pitch', 0.0) * gamepad_speed_rot
                    v_vel[7] = j_cmds.get('wrist_roll', 0.0) * gamepad_speed_rot
                    v = v_vel * dt
                else: 
                    j_cmds = cmd.get('joint_velocity_commands', {})
                    v_vel = np.zeros(8)
                    # Bound the commands to be in the range [-1, 1] and use this to scale the robot's joint speed.
                    # For each joint, this will limit the speed to the maximum speed for that joint.
                    # NOTE: the 
                    v_vel[0] = np.clip(j_cmds.get('base_x', 0.0), -1.0, 1.0) * accel_vel_dict['vel_base_xy']
                    v_vel[1] = np.clip(j_cmds.get('base_y', 0.0), -1.0, 1.0) * accel_vel_dict['vel_base_xy']
                    v_vel[2] = np.clip(j_cmds.get('base_theta', 0.0), -1.0, 1.0) * accel_vel_dict['vel_base_w']
                    v_vel[3] = np.clip(j_cmds.get('lift', 0.0), -1.0, 1.0) * accel_vel_dict['vel_lift']
                    v_vel[4] = np.clip(j_cmds.get('arm', 0.0), -1.0, 1.0) * accel_vel_dict['vel_arm']
                    v_vel[5] = np.clip(j_cmds.get('wrist_yaw', 0.0), -1.0, 1.0) * accel_vel_dict['vel_yaw']
                    v_vel[6] = np.clip(j_cmds.get('wrist_pitch', 0.0), -1.0, 1.0) * accel_vel_dict['vel_pitch']
                    v_vel[7] = np.clip(j_cmds.get('wrist_roll', 0.0), -1.0, 1.0) * accel_vel_dict['vel_roll']
                    v = v_vel * dt
            else:
                # v_desired and rot_change have already been scaled by the sender's left_trigger
                v_desired = np.array(cmd.get('v_desired', [0.0, 0.0, 0.0])) * gamepad_speed_trans * dt
                rot_change = np.array(cmd.get('rot_change', [0.0, 0.0, 0.0])) * gamepad_speed_rot * dt
                
                # v maps to joint displacements [delta_q_base_x, delta_q_base_y, delta_q_base_theta, delta_q_lift, delta_q_arm, delta_q_yaw, delta_q_ pitch, delta_q_roll]
                v, _ = ikin.compute_ik_step(v_desired, rot_change, control_mode)
                
                # Scale from displacement `v` back to continuous velocity by dividing by `dt`
                v_vel = v / dt
                
                if control_mode == 4:
                    v_vel[0] = 0.0
                    v_vel[1] = 0.0
                
            if np.any(v != 0):
                # Command actual hardware
                robot.base.set_velocity(v_vel[0], v_vel[1], v_vel[2], a_m=accel_base_xy, a_r=accel_base_w)
                if 'lift' not in active_pos_cmds:
                    robot.lift.set_velocity(v_vel[3], a_m=accel_lift)
                robot.arm.set_velocity(v_vel[4], a_m=accel_arm)
                
                pitch_sign_mult = 1.0 if args.disable_flipped_wrist else -1.0
                roll_sign_mult = 1.0 if args.disable_flipped_wrist else -1.0
                
                # Smoothing move_by control commands using a high lookahead targeting horizon
                # and reduced acceleration to seamlessly simulate velocity tracking without stopping abruptly
                lookahead = 10.0
                v_yaw_cmd = min(abs(v_vel[5]), vel_yaw)
                v_pitch_cmd = min(abs(v_vel[6]), vel_pitch)
                v_roll_cmd = min(abs(v_vel[7]), vel_roll)
                
                robot.end_of_arm.move_by('wrist_yaw', v[5] * lookahead, v_yaw_cmd, accel_yaw * 0.5)
                robot.end_of_arm.move_by('wrist_pitch', v[6] * pitch_sign_mult * lookahead, v_pitch_cmd, accel_pitch * 0.5)
                if 'wrist_roll' not in active_pos_cmds:
                    robot.end_of_arm.move_by('wrist_roll', v[7] * roll_sign_mult * lookahead, v_roll_cmd, accel_roll * 0.5)
            else:
                robot.base.set_velocity(0, 0, 0, accel_base_xy*2, accel_base_w*2)
                if 'lift' not in active_pos_cmds:
                    robot.lift.set_velocity(0, a_m=accel_lift)
                robot.arm.set_velocity(0, a_m=accel_arm)
                robot.end_of_arm.move_by('wrist_yaw', 0)
                robot.end_of_arm.move_by('wrist_pitch', 0)
                if 'wrist_roll' not in active_pos_cmds:
                    robot.end_of_arm.move_by('wrist_roll', 0)
                
            robot.push_command()
            
            # Use dynamic sleep to maintain loop frequency
            sleep_time = rate + dt - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            rate = time.time()
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        robot.base.set_velocity(0, 0, 0)
        robot.lift.set_velocity(0)
        robot.arm.set_velocity(0)
        robot.end_of_arm.move_by('wrist_yaw', 0)
        robot.end_of_arm.move_by('wrist_pitch', 0)
        robot.end_of_arm.move_by('wrist_roll', 0)
        robot.push_command()
        
        robot.stop()

if __name__ == '__main__':
    main()
