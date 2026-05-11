from dataclasses import dataclass
import numpy as np

@dataclass
class FingertipComplianceConfig:
    """
    Configuration file defining the feasible set of compliant deformations
    for the Stretch 4 gripper's fingertips. 

    The bounds are defined with respect to the unloaded fingertip coordinate frames:
    - X-axis natively points OUT of the fingertip (longitudinal).
    - Y-axis natively points DOWN (gravitational).
    - Z-axis natively points INWARD (orthogonal to the finger pad).

    Deformations can be aggressively injected into simulated poses to benchmark 
    estimation robustness algorithms before real-world deployment.
    """

    # ----------------------------------------------------------------------
    # Translational Deformations (meters)
    # ----------------------------------------------------------------------
    
    # 1. Y-Axis: Vertical slipping / shear
    # Small translations with respect to the fingertip frame's y-axis.
    max_trans_y_pos_m: float = 0.005
    max_trans_y_neg_m: float = 0.005

    # 2. X-Axis: Longitudinal push / pull
    # Large translations in the positive direction of the fingertip frame's x-axis
    # (finger is compressed backwards toward the base).
    max_trans_x_pos_m: float = 0.020
    
    # Small translations in the negative direction of the fingertip frame's x-axis
    # (finger is pulled outward away from the base).
    max_trans_x_neg_m: float = 0.005

    # 3. Z-Axis: Inward / Outward bending
    # Large translations in the positive direction of the fingertip frame's z-axis
    # (finger bends inward significantly under heavy object compression).
    max_trans_z_pos_m: float = 0.025
    
    # Small translations in the negative direction of the fingertip frame's z-axis
    # (rigid mechanical layout prevents extensive outward splay).
    max_trans_z_neg_m: float = 0.005

    # ----------------------------------------------------------------------
    # Rotational Twisting Deformations (degrees)
    # ----------------------------------------------------------------------

    # Large twisting around the fingertip frame's x-axis
    # (The highly compliant rubber pad rolling significantly over geometry edges).
    max_twist_x_pos_deg: float = 45.0
    max_twist_x_neg_deg: float = 45.0

    # Large twisting around the fingertip frame's y-axis
    # (Finger twists right/left like a steering wheel).
    max_twist_y_pos_deg: float = 40.0
    max_twist_y_neg_deg: float = 40.0

    # Small twisting around the fingertip frame's z-axis
    # (Pitching up/down is mechanically restricted by the thick spine).
    max_twist_z_pos_deg: float = 10.0
    max_twist_z_neg_deg: float = 10.0

    def sample_random_deformation(self):
        """
        Samples a uniformly random feasible translation and rotation 
        within the currently configured compliance bounds.
        Returns:
            translation (np.ndarray): 3x1 translation offset in fingertip frame.
            rotation (np.ndarray): 3x3 rotation matrix representing the twist applied in fingertip frame.
        """
        # Sample translations
        trans_x = np.random.uniform(-self.max_trans_x_neg_m, self.max_trans_x_pos_m)
        trans_y = np.random.uniform(-self.max_trans_y_neg_m, self.max_trans_y_pos_m)
        trans_z = np.random.uniform(-self.max_trans_z_neg_m, self.max_trans_z_pos_m)
        
        # Sample twists (rotations) in degrees, convert to radians
        tw_x = np.deg2rad(np.random.uniform(-self.max_twist_x_neg_deg, self.max_twist_x_pos_deg))
        tw_y = np.deg2rad(np.random.uniform(-self.max_twist_y_neg_deg, self.max_twist_y_pos_deg))
        tw_z = np.deg2rad(np.random.uniform(-self.max_twist_z_neg_deg, self.max_twist_z_pos_deg))
        
        # Compose rotations natively (using scipy or basic rotation matrices)
        import cv2
        R_x, _ = cv2.Rodrigues(np.array([tw_x, 0.0, 0.0]))
        R_y, _ = cv2.Rodrigues(np.array([0.0, tw_y, 0.0]))
        R_z, _ = cv2.Rodrigues(np.array([0.0, 0.0, tw_z]))
        
        R_composite = R_z @ R_y @ R_x
        t_composite = np.array([trans_x, trans_y, trans_z])
        
        return t_composite, R_composite
