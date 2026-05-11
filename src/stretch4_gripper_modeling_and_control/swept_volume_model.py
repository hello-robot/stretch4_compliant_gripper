import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from stretch4_gripper_modeling_and_control.aruco_to_fingertips import suctioncup_radius

class SweptVolumeModel:
    def __init__(self, kinematic_model, side, start_cfg, end_cfg):
        """
        kinematic_model: FingertipVisualizer instance
        side: 'left' or 'right'
        start_cfg: float (pos_pct) or dict (visual frame containing 'pos', 'x_axis', 'y_axis', 'z_axis')
        end_cfg: float (pos_pct) or dict
        """
        self.model = kinematic_model
        self.side = side
        
        # Determine the start and end pct and their exact frames
        self.start_pct, self.pos_s_act, self.rot_s_act = self._process_cfg(start_cfg)
        self.end_pct, self.pos_e_act, self.rot_e_act = self._process_cfg(end_cfg)
        
        self.m_pos_s, self.m_rot_s = self.model.predict(self.side, self.start_pct)
        self.m_pos_e, self.m_rot_e = self.model.predict(self.side, self.end_pct)
        
        self.valid = not (self.m_pos_s is None or self.m_pos_e is None)
        if not self.valid:
            return
            
        # Compute interpolation boundaries (errors)
        self.trans_err_s = self.pos_s_act - self.m_pos_s
        self.rot_err_s = Rotation.from_matrix(self.rot_s_act @ self.m_rot_s.T)
        
        self.trans_err_e = self.pos_e_act - self.m_pos_e
        self.rot_err_e = Rotation.from_matrix(self.rot_e_act @ self.m_rot_e.T)
        
        # Precompute SLERP function
        key_rots = Rotation.concatenate([self.rot_err_s, self.rot_err_e])
        key_times = [0.0, 1.0]
        self.slerp = Slerp(key_times, key_rots)

    def _process_cfg(self, cfg):
        search_pcts = np.linspace(-100, 300, 401)
        if isinstance(cfg, (int, float)):
            pct = float(cfg)
            pos, rot = self.model.predict(self.side, pct)
            return pct, pos, rot
        else:
            pos_vis = cfg['pos']
            rot_vis = np.column_stack((cfg['x_axis'], cfg['y_axis'], cfg['z_axis']))
            
            dists = []
            valid_pcts = []
            for p in search_pcts:
                m_pos, _ = self.model.predict(self.side, p)
                if m_pos is not None:
                    dists.append(np.linalg.norm(m_pos - pos_vis))
                    valid_pcts.append(p)
            
            if not valid_pcts:
                return 0.0, pos_vis, rot_vis
                
            best_pct = valid_pcts[np.argmin(dists)]
            return best_pct, pos_vis, rot_vis

    def get_frame(self, pct):
        """
        Returns the (pos, rot) for a given pos_pct that falls within or outside 
        the swept volume's bounds, interpolating errors proportionally.
        """
        if not self.valid:
            return None, None
            
        m_pos, m_rot = self.model.predict(self.side, pct)
        if m_pos is None:
            return None, None
            
        if self.end_pct == self.start_pct:
            t = 1.0
        else:
            t = (pct - self.start_pct) / (self.end_pct - self.start_pct)
            
        t_clamped = np.clip(t, 0.0, 1.0)
        
        trans_err_t = self.trans_err_s * (1 - t_clamped) + self.trans_err_e * t_clamped
        rot_err_t = self.slerp([t_clamped])[0].as_matrix()
        
        pos_t = m_pos + trans_err_t
        rot_t = rot_err_t @ m_rot
        
        return pos_t, rot_t

    def get_circle(self, pct):
        """
        Returns the center of the circle representing the top edge of the suction cup,
        and a unit vector representing the normal to the circle.
        """
        pos, rot = self.get_frame(pct)
        if pos is None:
            return None, None
            
        center = pos
        # the normal is the local z-axis, which is the 3rd column of the rotation matrix
        normal = rot[:, 2]
        
        return center, normal

    def get_circle_points(self, pct, num_points=16, radius=suctioncup_radius):
        """
        Returns a list of 3D points comprising the circular top edge of the cup at pos_pct.
        """
        pos, rot = self.get_frame(pct)
        if pos is None:
            return None
        
        center = pos
        normal = rot[:, 2]
        
        # Compute a twist-free orthogonal basis aligned with global Z
        up = np.array([0.0, 0.0, 1.0])
        x_axis = up - np.dot(up, normal) * normal
        norm_x = np.linalg.norm(x_axis)
        if norm_x < 1e-4:
            up = np.array([1.0, 0.0, 0.0])
            x_axis = up - np.dot(up, normal) * normal
            norm_x = np.linalg.norm(x_axis)
        x_axis = x_axis / norm_x
        y_axis = np.cross(normal, x_axis)
        
        pts = []
        for i in range(num_points):
            theta = 2.0 * np.pi * i / num_points
            pt = center + radius * np.cos(theta) * x_axis + radius * np.sin(theta) * y_axis
            pts.append(pt)
        return pts

    def get_sampled_pcts(self, sampling_method='pos_pct', num_samples=30):
        if not self.valid or num_samples < 2:
            return []
            
        if sampling_method == 'pos_pct':
            return np.linspace(self.start_pct, self.end_pct, num_samples)
        elif sampling_method == 'arc_length':
            dense_steps = max(100, num_samples * 5)
            dense_pcts = np.linspace(self.start_pct, self.end_pct, dense_steps)
            
            centers = []
            valid_pcts = []
            for pct in dense_pcts:
                pos, _ = self.get_frame(pct)
                if pos is not None:
                    centers.append(pos)
                    valid_pcts.append(pct)
                    
            if len(valid_pcts) < 2:
                return [self.start_pct, self.end_pct] if self.start_pct != self.end_pct else [self.start_pct]
                
            centers = np.array(centers)
            dists = np.linalg.norm(np.diff(centers, axis=0), axis=1)
            cum_dists = np.zeros(len(valid_pcts))
            cum_dists[1:] = np.cumsum(dists)
            
            total_length = cum_dists[-1]
            if total_length == 0:
                return np.linspace(self.start_pct, self.end_pct, num_samples)
                
            uniform_dists = np.linspace(0, total_length, num_samples)
            sampled_pcts = np.interp(uniform_dists, cum_dists, valid_pcts)
            return sampled_pcts
        else:
            raise ValueError("sampling_method must be either 'pos_pct' or 'arc_length'")
