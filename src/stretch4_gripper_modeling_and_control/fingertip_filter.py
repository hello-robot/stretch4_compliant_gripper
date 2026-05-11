import numpy as np
import cv2

class ExponentialMovingAverageSE3:
    def __init__(self, alpha=0.3, max_missing=0, max_trans_outlier=0.05, max_rot_outlier_deg=45.0, smooth=False):
        # We weigh the new measurement relative to alpha. 
        # If not smoothing, alpha=1.0 strictly snaps to exactly the newly incoming measurement or rejects if outlier.
        self.alpha = alpha if smooth else 1.0 
        self.max_missing = max_missing
        
        self.max_trans_outlier = max_trans_outlier
        self.max_rot_outlier_rad = np.deg2rad(max_rot_outlier_deg)
        self.smooth = smooth
        
        self.T_prev = None
        self.missing_count = 0

    def _dict_to_T(self, f_dict):
        T = np.eye(4)
        T[:3, 0] = f_dict['x_axis']
        T[:3, 1] = f_dict['y_axis']
        T[:3, 2] = f_dict['z_axis']
        T[:3, 3] = f_dict['pos']
        return T

    def _T_to_dict(self, T):
        return {
            'pos': T[:3, 3],
            'x_axis': T[:3, 0],
            'y_axis': T[:3, 1],
            'z_axis': T[:3, 2]
        }

    def update(self, f_dict_meas):
        """
        Takes raw fingertip dict natively emitted from aruco_to_fingertips.
        Pipes through the mathematical logic returning a robust filtered dictionary, or None.
        """
        if f_dict_meas is None:
            if self.T_prev is not None and self.missing_count < self.max_missing:
                self.missing_count += 1
                return self._T_to_dict(self.T_prev)
            else:
                self.missing_count += 1
                return None
                
        T_meas = self._dict_to_T(f_dict_meas)
        
        if self.T_prev is None:
            self.T_prev = T_meas.copy()
            self.missing_count = 0
            return self._T_to_dict(self.T_prev)
            
        t_prev = self.T_prev[:3, 3]
        R_prev = self.T_prev[:3, :3]
        t_meas = T_meas[:3, 3]
        R_meas = T_meas[:3, :3]
        
        trans_dist = np.linalg.norm(t_meas - t_prev)
        R_diff = R_prev.T @ R_meas
        rvec_diff, _ = cv2.Rodrigues(R_diff)
        rot_dist = np.linalg.norm(rvec_diff)
        
        # Filter strictly rejects structurally erratic measurements
        if trans_dist > self.max_trans_outlier or rot_dist > self.max_rot_outlier_rad:
            # Provide patience resilience to sporadic outlier blips
            if self.missing_count < self.max_missing:
                self.missing_count += 1
                return self._T_to_dict(self.T_prev)
            else:
                # Exhausted patience or zero patience means we abruptly reset track directly to target
                self.T_prev = T_meas.copy()
                self.missing_count = 0
                return self._T_to_dict(self.T_prev)
                
        self.missing_count = 0
        
        # Combine smoothly using geodesic SE(3) exponential mapping
        rvec_interp = self.alpha * rvec_diff
        R_new = R_prev @ cv2.Rodrigues(rvec_interp)[0]
        
        t_diff = t_meas - t_prev
        t_new = t_prev + self.alpha * t_diff
        
        T_new = np.eye(4)
        T_new[:3, :3] = R_new
        T_new[:3, 3] = t_new
        
        self.T_prev = T_new
        return self._T_to_dict(self.T_prev)
