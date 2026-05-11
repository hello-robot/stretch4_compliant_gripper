import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from visualize_fingertip_model import FingertipVisualizer
from stretch4_gripper_modeling_and_control.fingertip_compliance_config import FingertipComplianceConfig
from stretch4_gripper_modeling_and_control.aruco_to_fingertips import ArucoToFingertips

class ArucoKinematicValidator:
    def __init__(self, model_yaml_path):
        """
        Loads the standalone kinematic fingertip model and compliance bounds.
        """
        self.model = FingertipVisualizer(model_yaml_path)
        self.config = FingertipComplianceConfig()
        self.a2f_transforms = ArucoToFingertips().get_transforms()

        # Cache evaluating nominal poses along the trajectory to avoid repeated spline queries
        self.trajectory_samples = 200
        self.pct_space = np.linspace(-100, 300, self.trajectory_samples)
        
        self.nominal_trajectories = { 'left': [], 'right': [] }
        for side in ['left', 'right']:
            for p in self.pct_space:
                pos, F_rot = self.model.predict(side, p, direction='closing')
                if pos is not None:
                    T = np.eye(4)
                    T[:3, :3] = F_rot
                    T[:3, 3] = pos
                    self.nominal_trajectories[side].append(T)
                    
        # Construct the theoretical World-to-Camera plane frame mapping natively
        pg = self.model.centroid
        normal = self.model.normal.copy()
        centroid = self.model.centroid.copy()
        if np.dot(-centroid, normal) < 0:
            normal = -normal
            
        cam_z = np.array([0.0, 0.0, 1.0])
        z_proj = cam_z - np.dot(cam_z, normal) * normal
        z_proj_norm = np.linalg.norm(z_proj)
        
        if z_proj_norm > 1e-6:
            Y_new = z_proj / z_proj_norm
        else:
            cam_y = np.array([0.0, 1.0, 0.0])
            y_proj = cam_y - np.dot(cam_y, normal) * normal
            Y_new = y_proj / np.linalg.norm(y_proj)
            
        X_new = np.cross(Y_new, normal)
        X_new = X_new / np.linalg.norm(X_new)
        
        T_model = np.eye(4)
        T_model[:3, 0] = X_new
        T_model[:3, 1] = Y_new
        T_model[:3, 2] = normal
        T_model[:3, 3] = centroid
        self.camera_to_world = T_model
        
        self._last_cache_key = None
        self._last_T_nom = None

    def is_pose_feasible(self, side: str, rvec: np.ndarray, tvec: np.ndarray, pos_pct: float = None, direction: str = 'closing') -> bool:
        """
        Determines whether a proposed raw ArUco marker camera pose is physically feasible,
        given the nominal kinematic trajectory and the allowed compliance deformations.

        Args:
            side: 'left' or 'right'
            rvec, tvec: The OpenCV camera frame pose of the ArUco marker.
            pos_pct: Optional actual encoder position percent. If None, checks the whole trajectory.
            direction: Optional opening/closing direction.
        """
        # 1. Convert ArUco Camera Pose to World Pose
        A_rot, _ = cv2.Rodrigues(rvec)
        T_marker_c = np.eye(4)
        T_marker_c[:3, :3] = A_rot
        T_marker_c[:3, 3] = tvec.flatten()

        T_marker_w = self.camera_to_world @ T_marker_c

        # 2. Convert Marker World Pose to Fingertip World Pose
        T_finger_candidate_w = T_marker_w @ self.a2f_transforms[side]

        # 3. Get Nominal Poses to test against
        if pos_pct is not None:
            # Check exactly against the commanded pos_pct using a basic cache to 
            # prevent hitting the spline math multiple times for the same hypotheses
            cache_key = (side, pos_pct, direction)
            if cache_key != self._last_cache_key:
                pos, F_rot = self.model.predict(side, pos_pct, direction=direction)
                if pos is None:
                    return False
                T_nom = np.eye(4)
                T_nom[:3, :3] = F_rot
                T_nom[:3, 3] = pos
                
                self._last_T_nom = T_nom
                self._last_cache_key = cache_key
                
            nominal_poses = [self._last_T_nom]
        else:
            # Check if it fits ANY point globally along the trajectory
            nominal_poses = self.nominal_trajectories[side]

        # 4. Find if any nominal pose makes it feasible
        for T_nom in nominal_poses:
            if self._is_deformation_within_bounds(T_nom, T_finger_candidate_w):
                return True
                
        return False

    def _is_deformation_within_bounds(self, T_nominal_w: np.ndarray, T_candidate_w: np.ndarray) -> bool:
        """
        Strictly enforces the bounds from FingertipComplianceConfig 
        by mathematically extracting the deformation component in the nominal fingertip frame.
        """
        # T_candidate = T_nominal @ T_deformation  ==>  T_deformation = inv(T_nominal) @ T_candidate
        T_deformation = np.linalg.inv(T_nominal_w) @ T_candidate_w

        t_diff = T_deformation[:3, 3]
        
        # Check Translations against defined compliant bounds
        if not (-self.config.max_trans_x_neg_m <= t_diff[0] <= self.config.max_trans_x_pos_m): return False
        if not (-self.config.max_trans_y_neg_m <= t_diff[1] <= self.config.max_trans_y_pos_m): return False
        if not (-self.config.max_trans_z_neg_m <= t_diff[2] <= self.config.max_trans_z_pos_m): return False

        # Extract rotations as Euler Angles assuming 'xyz' (Roll-Pitch-Yaw) natively maps to the fingertip frame's axes
        r_diff = Rotation.from_matrix(T_deformation[:3, :3]).as_euler('xyz', degrees=True)
        tw_x, tw_y, tw_z = r_diff[0], r_diff[1], r_diff[2]
        
        # Check Rotations against defined bounds
        if not (-self.config.max_twist_x_neg_deg <= tw_x <= self.config.max_twist_x_pos_deg): return False
        if not (-self.config.max_twist_y_neg_deg <= tw_y <= self.config.max_twist_y_pos_deg): return False
        if not (-self.config.max_twist_z_neg_deg <= tw_z <= self.config.max_twist_z_pos_deg): return False

        return True
