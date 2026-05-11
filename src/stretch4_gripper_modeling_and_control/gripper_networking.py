import subprocess

# Set these values for your network
robot_ip = '100.90.83.97' #
remote_computer_ip = '100.69.89.24'

# Set these to your preferred port numbers
gripper_cmd_port = 4407
gripper_and_joints_port = 4409

network_debugging = False

def print_network_info():
    print("================================================================")
    print("WARNING: If you experience ZMQ connection issues between a") 
    print("remote computer and the robot, please ensure the ports are") 
    print("not blocked by a firewall.")
    print("")
    print("For example,the Uncomplicated Firewall (UFW) can be configured")
    print("to allow incoming TCP connections on port 4409 by running: ")
    print("sudo ufw allow 4409/tcp")
    print("================================================================")
    if network_debugging:
        print("\n[NETWORK DEBUG] Active Network Interfaces:")
        try:
            result = subprocess.run(['ip', '-4', '-br', 'addr'], capture_output=True, text=True, timeout=2)
            print(result.stdout)
            if 'tailscale0' in result.stdout:
                print("[NETWORK DEBUG] -> Tailscale active. If connecting remotely, ensure IPs use the 100.x.x.x Tailscale subnet.")
            else:
                print("[NETWORK DEBUG] -> Tailscale NOT found. Ensure devices are on the same local subnet.")
        except Exception as e:
            print(f"[NETWORK DEBUG] Failed to list interfaces: {e}")
        print("----------------------------------------------------------------\n")
