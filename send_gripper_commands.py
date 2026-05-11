#!/usr/bin/env python3
import time
import argparse
import sys
import numpy as np
import zmq

from gamepad_mapper import GamepadMapper
from stretch4_gripper_modeling_and_control import gripper_networking as gn

def main():
    parser = argparse.ArgumentParser(description='Send Gripper-centric Commands via ZMQ')
    parser.add_argument('-r', '--remote', action='store_true', help='Use this argument when allowing a remote computer to receive commands. Configure gripper_networking.py first.')
    args = parser.parse_args()

    print("Initializing Gamepad Mapper...")
    try:
        gamepad = GamepadMapper()
    except Exception as e:
        print(f"Failed to connect to gamepad: {e}")
        sys.exit(1)
        
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.RCVHWM, 1)
    
    if args.remote:
        address = 'tcp://*:' + str(gn.gripper_cmd_port)
    else:
        address = 'tcp://127.0.0.1:' + str(gn.gripper_cmd_port)
        
    print(f"Binding ZMQ Publisher to {address}")
    socket.bind(address)
    
    gn.print_network_info()

    control_mode = 1
    hz = 30.0
    dt = 1.0 / hz
    rate = time.time()
    
    print("====================================")
    print("Sending Gripper Commands ZMQ Started")
    print("Commands scaled by Left Trigger.")
    print("Ctrl+C to Quit")
    print("====================================")

    try:
        while True:
            cmd = gamepad.get_commands()
            
            if cmd:
                if cmd.get('toggle'):
                    control_mode = (control_mode % 3) + 1
                    mode_names = {
                        1: "Gripper Frame Relative (IK)",
                        2: "Projected Base Frame Relative (IK)",
                        3: "Joint-Space Direct Control"
                    }
                    print(f"--> Switched to Mode {control_mode}: {mode_names[control_mode]}")
                    sys.stdout.flush()

                # Apply left-trigger proportional dampening: halving speed at full press
                left_trigger = cmd.get('left_trigger', 0.0)
                speed_multiplier = 1.0 - (0.5 * left_trigger)

                if control_mode == 3:
                    right_trigger = cmd.get('right_trigger', 0.0)
                    joint_cmds = {
                        'lift': cmd['v_desired'][2] * speed_multiplier,
                        'arm': 0.0,
                        'base_x': 0.0,
                        'base_y': 0.0,
                        'base_theta': 0.0,
                        'wrist_yaw': 0.0,
                        'wrist_pitch': 0.0,
                        'wrist_roll': 0.0
                    }
                    
                    if right_trigger > 0.1:
                        joint_cmds['arm'] = cmd['v_desired'][0] * speed_multiplier
                        joint_cmds['wrist_roll'] = -cmd['v_desired'][1] * speed_multiplier
                        joint_cmds['wrist_pitch'] = cmd['rot_change'][1] * speed_multiplier
                        joint_cmds['wrist_yaw'] = cmd['rot_change'][0] * speed_multiplier
                    else:
                        joint_cmds['base_x'] = cmd['v_desired'][0] * speed_multiplier
                        joint_cmds['base_y'] = cmd['v_desired'][1] * speed_multiplier
                        joint_cmds['base_theta'] = cmd['rot_change'][0] * speed_multiplier
                        joint_cmds['arm'] = cmd['rot_change'][1] * speed_multiplier

                    output_dict = {
                        'control_mode': 3,
                        'joint_velocity_commands': joint_cmds,
                        'grip': cmd.get('grip')
                    }
                else:
                    # The v_desired and rot_change from gamepad are already normalized to [-1, 1]
                    v_desired = np.array(cmd['v_desired']) * speed_multiplier
                    rot_change = np.array(cmd['rot_change']) * speed_multiplier

                    output_dict = {
                        'control_mode': control_mode,
                        'v_desired': v_desired.tolist(),
                        'rot_change': rot_change.tolist(),
                        'grip': cmd.get('grip')
                    }
                socket.send_pyobj(output_dict)
            else:
                # Send zero velocities if no command
                output_dict = {
                    'control_mode': control_mode,
                    'v_desired': [0.0, 0.0, 0.0],
                    'rot_change': [0.0, 0.0, 0.0],
                    'grip': None
                }
                socket.send_pyobj(output_dict)

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
        gamepad.stop()
        print("\nStopped transmitting.")

if __name__ == '__main__':
    main()
