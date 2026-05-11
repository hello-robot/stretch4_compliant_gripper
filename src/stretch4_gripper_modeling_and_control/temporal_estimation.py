import copy
import numpy as np
from scipy.spatial.transform import Rotation as R

class TemporalEstimator:
    """Base class for modeling fingertip frames across time."""
    def __init__(self):
        pass

    def reset(self):
        """Reset the internal temporal state."""
        pass

    def update(self, pos_pct, vis_fingertips):
        """
        Update the temporal estimate given the current state.
        
        Args:
            pos_pct (float): Current position parameter of the gripper actuator.
            vis_fingertips (dict): Dictionary with 'left' and 'right' fingertip frames
                                   as visually estimated from the current image.
                                   Format: {'pos': array, 'x_axis': array, 'y_axis': array, 'z_axis': array}
        
        Returns:
            dict: The temporally estimated fingertip frames.
        """
        raise NotImplementedError


class AdaptiveBaselineEstimator(TemporalEstimator):
    """
    An estimator that computes a temporal baseline of the unloaded, static fingertip frames.
    
    If the gripper actuator is moving (pos_pct changing), the baseline follows the visual
    estimate closely.
    If the gripper actuator is still and the visual frames are stable (e.g. unloaded),
    it heavily averages the signal to build a high-confidence steady frame.
    If the visual frames suddenly deviate while the actuator is still (e.g. a load is applied),
    it stops adapting the baseline so the difference can be visualized as the applied load.
    """
    def __init__(self, pos_pct_thresh=0.5, pos_diff_mm_thresh=1.0, rot_diff_deg_thresh=1.0, alpha_steady=0.05, alpha_jump=1.0):
        super().__init__()
        self.pos_pct_thresh = pos_pct_thresh  # pct change
        self.pos_diff_mm_thresh = pos_diff_mm_thresh
        self.rot_diff_deg_thresh = rot_diff_deg_thresh
        self.alpha_steady = alpha_steady
        self.alpha_jump = alpha_jump
        
        self.last_pos_pct = None
        self.baseline_frames = None
        self.last_vis_fingertips = None

    def reset(self):
        self.last_pos_pct = None
        self.baseline_frames = None
        self.last_vis_fingertips = None

    def update(self, pos_pct, vis_fingertips):
        if pos_pct is None or not vis_fingertips:
            return copy.deepcopy(self.baseline_frames) if self.baseline_frames else {}
            
        if self.baseline_frames is None or self.last_pos_pct is None:
            self.baseline_frames = copy.deepcopy(vis_fingertips)
            self.last_pos_pct = pos_pct
            self.last_vis_fingertips = copy.deepcopy(vis_fingertips)
            return copy.deepcopy(self.baseline_frames)

        # Check if actuator changed
        pct_diff = abs(pos_pct - self.last_pos_pct)
        self.last_pos_pct = pos_pct
        
        new_baseline = {}
        for side in ['left', 'right']:
            if side not in vis_fingertips:
                continue
                
            if side not in self.baseline_frames:
                new_baseline[side] = copy.deepcopy(vis_fingertips[side])
                continue
                
            cur_f = vis_fingertips[side]
            base_f = self.baseline_frames[side]
            
            # Check frame similarity with PREVIOUS visual frame to detect stability
            if self.last_vis_fingertips and side in self.last_vis_fingertips:
                prev_f = self.last_vis_fingertips[side]
                pos_diff_mm = np.linalg.norm(cur_f['pos'] - prev_f['pos']) * 1000.0
                
                R_cur = np.column_stack((cur_f['x_axis'], cur_f['y_axis'], cur_f['z_axis']))
                R_prev = np.column_stack((prev_f['x_axis'], prev_f['y_axis'], prev_f['z_axis']))
                R_diff = R_prev.T @ R_cur
                trace = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
                rot_diff_deg = np.degrees(np.arccos(trace))
            else:
                pos_diff_mm = 0.0
                rot_diff_deg = 0.0
                R_cur = np.column_stack((cur_f['x_axis'], cur_f['y_axis'], cur_f['z_axis']))
                
            R_base = np.column_stack((base_f['x_axis'], base_f['y_axis'], base_f['z_axis']))
            
            if pct_diff > self.pos_pct_thresh:
                # Gripper actuator is moving, reset the baseline to follow immediately
                alpha = self.alpha_jump
            else:
                if pos_diff_mm < self.pos_diff_mm_thresh and rot_diff_deg < self.rot_diff_deg_thresh:
                    # Visual frames are steady over time -> update baseline (adapt to static load)
                    alpha = self.alpha_steady
                else:
                    # Visual frames are changing rapidly (transient impact/load) -> pause baseline updates
                    alpha = 0.0
                    
            # Update position
            new_pos = base_f['pos'] * (1 - alpha) + cur_f['pos'] * alpha
            
            # Update rotation via exponential map
            if alpha == 1.0:
                new_R = R_cur
            elif alpha == 0.0:
                new_R = R_base
            else:
                try:
                    r_base = R.from_matrix(R_base)
                    r_cur = R.from_matrix(R_cur)
                    # Interpolate from r_base to r_cur by alpha
                    r_diff = (r_base.inv() * r_cur).as_rotvec()
                    r_new = r_base * R.from_rotvec(r_diff * alpha)
                    new_R = r_new.as_matrix()
                except Exception:
                    # Fallback to simple averaging and orthogonalization
                    new_R = R_base * (1 - alpha) + R_cur * alpha
                    U, _, Vt = np.linalg.svd(new_R)
                    new_R = U @ Vt
            
            new_f = {
                'pos': new_pos,
                'x_axis': new_R[:, 0],
                'y_axis': new_R[:, 1],
                'z_axis': new_R[:, 2]
            }
            new_baseline[side] = new_f
            
        self.baseline_frames = new_baseline
        self.last_vis_fingertips = copy.deepcopy(vis_fingertips)
        return copy.deepcopy(self.baseline_frames)
