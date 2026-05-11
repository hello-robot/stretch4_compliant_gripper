import yaml
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation
from stretch4_gripper_modeling_and_control.aruco_to_fingertips import ArucoToFingertips

class FingertipVisualizer:
    def __init__(self, model_path):
        with open(model_path, 'r') as f:
            data = yaml.safe_load(f)
            
        self.model_type = data.get('model_type', 'PlanarFingertipModel')
        self.hysteresis = data.get('hysteresis_compensation', None)
        
        pg = data['plane_geometry']
        self.centroid = np.array(pg['centroid'])
        self.normal = np.array(pg['normal'])
        self.X_plane = np.array(pg['X_plane'])
        self.Y_plane = np.array(pg['Y_plane'])
        
        self.splines = {'left': {}, 'right': {}}
        self.base_frames = {}
        for side in ['left', 'right']:
            fg = data['fingertips'][side]
            self.base_frames[side] = np.array(fg['base_frame'])
            for var in ['u', 'v', 'theta']:
                sp = fg[f'spline_{var}']
                self.splines[side][var] = BSpline(sp['knots'], sp['coeffs'], sp['degree'], extrapolate=False)
                
        self.a2f_transforms = ArucoToFingertips().get_transforms()
        
    def predict(self, side, pct, direction='closing'):
        eff_pct = pct
        if direction == 'opening' and self.hysteresis and side in self.hysteresis:
            h_data = self.hysteresis[side]
            if h_data['degree'] >= 0 and h_data['coeffs']:
                poly = np.poly1d(h_data['coeffs'])
                eff_pct = pct - poly(pct)
                
        try:
            k = self.splines[side]['u'].k
            t = self.splines[side]['u'].t
            eff_pct = float(np.clip(eff_pct, t[k], t[-k-1]))
            
            u = float(self.splines[side]['u'](eff_pct))
            v = float(self.splines[side]['v'](eff_pct))
            theta = float(self.splines[side]['theta'](eff_pct))
            if np.isnan(u) or np.isnan(v) or np.isnan(theta):
                return None, None
        except ValueError:
            return None, None
            
        pos = self.centroid + u * self.X_plane + v * self.Y_plane
        rotvec = theta * self.normal
        R_rel = Rotation.from_rotvec(rotvec).as_matrix()
        F_pred = R_rel @ self.base_frames[side]
        
        return pos, F_pred
