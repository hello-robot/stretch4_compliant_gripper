import os

def get_default_model_path():
    """
    Attempts to locate the latest_model_planar.yaml from the local HELLO_FLEET directory.
    Returns the absolute path if found, otherwise returns None.
    """
    fleet_path = os.environ.get('HELLO_FLEET_PATH')
    fleet_id = os.environ.get('HELLO_FLEET_ID')
    
    if fleet_path and fleet_id and fleet_id != 'unknown':
        default_path = os.path.join(fleet_path, fleet_id, 'calibration_gripper', 'latest_model_planar.yaml')
        if os.path.exists(default_path):
            return default_path
            
    return None


def get_default_camera_calibration_path():
    """
    Attempts to locate calibration_rgb_head_camera.yaml (the intrinsics of the
    head and gripper cameras) from the local HELLO_FLEET directory.
    Returns the absolute path if found, otherwise returns None.
    """
    fleet_path = os.environ.get('HELLO_FLEET_PATH')
    fleet_id = os.environ.get('HELLO_FLEET_ID')

    if fleet_path and fleet_id and fleet_id != 'unknown':
        default_path = os.path.join(fleet_path, fleet_id, 'calibration_cameras', 'calibration_rgb_head_camera.yaml')
        if os.path.exists(default_path):
            return default_path

    return None
