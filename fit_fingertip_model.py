#!/usr/bin/env python3

import os
import sys
import yaml
import glob
import time
import argparse
import numpy as np
import cv2
import shutil
from scipy.interpolate import UnivariateSpline
from scipy.spatial.transform import Rotation

# Import the existing visualization tools


def robust_plane_fit(points, distance_threshold=0.005, max_iterations=1000):
    """
    Fits a 3D plane robustly using RANSAC.
    """
    best_inliers = []
    best_plane = None
    n_points = points.shape[0]
    
    if n_points < 3:
        raise ValueError("Need at least 3 points to fit a plane.")

    for _ in range(max_iterations):
        # 1. Randomly sample 3 points
        sample_indices = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = points[sample_indices]
        
        # 2. Define the plane normal
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        
        # If points are collinear, skip
        if norm_len < 1e-6:
            continue
            
        normal = normal / norm_len
        
        # 3. Calculate distance from all points to the plane
        # Distance = | (P - P1) \cdot normal |
        diffs = points - p1
        distances = np.abs(np.dot(diffs, normal))
        
        # 4. Count inliers
        inlier_indices = np.where(distances < distance_threshold)[0]
        
        if len(inlier_indices) > len(best_inliers):
            best_inliers = inlier_indices
            best_plane = (p1, normal)

    if best_plane is None or len(best_inliers) < 3:
        raise ValueError("RANSAC failed to find a valid plane.")

    # 5. Refine plane using SVD on all inliers
    inlier_points = points[best_inliers]
    centroid = np.mean(inlier_points, axis=0)
    centered = inlier_points - centroid
    U, S, Vt = np.linalg.svd(centered)
    refined_normal = Vt[-1, :]  # The eigenvector with the smallest eigenvalue
    
    return centroid, refined_normal, inlier_points

def clean_float(val):
    """Recursively convert numpy arrays/floats into native python floats for clean YAML serialization."""
    if isinstance(val, (np.float32, np.float64, float)):
        return float(val)
    elif isinstance(val, (np.ndarray, list, tuple)):
        return [clean_float(x) for x in val]
    elif isinstance(val, dict):
        return {k: clean_float(v) for k, v in val.items()}
    return val

import itertools

class BaseFingertipModel:
    def __init__(self):
        self.hyperparameters = {}
        self.errors = {'left': {'trans': 0, 'rot': 0}, 'right': {'trans': 0, 'rot': 0}}
        
    def fit(self, data_list, hyperparameters=None, verbose=True):
        raise NotImplementedError
        
    def predict(self, side, pos_pct, direction='closing'):
        raise NotImplementedError
        
    def get_score(self, errors):
        # A simple scoring function, lower is better.
        trans_err = (errors['left']['trans'] + errors['right']['trans']) / 2.0
        rot_err = (errors['left']['rot'] + errors['right']['rot']) / 2.0
        # Weighted sum: 1 mm = 1 deg (approx). 1 deg is ~0.017 rad.
        return trans_err * 1000.0 + rot_err * (180.0 / np.pi)
        
    def evaluate(self, data_list):
        """Standardized evaluation to calculate true inference errors against a dataset."""
        sequences = {'left': {'pos_pct': [], 'origin': [], 'F': [], 'direction': []}, 
                     'right': {'pos_pct': [], 'origin': [], 'F': [], 'direction': []}}
        
        for entry in data_list:
            pos_pct_before = entry.get('gripper_status_before', {}).get('pos_pct')
            pos_pct_after = entry.get('gripper_status_after', {}).get('pos_pct')
            if pos_pct_before is None or pos_pct_after is None:
                continue
            pos_pct = (pos_pct_before + pos_pct_after) / 2.0
            direction = 'opening' if pos_pct_after > pos_pct_before else 'closing'
            
            for side in ['left', 'right']:
                frame_key = f"{side}_frame"
                if frame_key in entry:
                    sequences[side]['pos_pct'].append(pos_pct)
                    sequences[side]['direction'].append(direction)
                    sequences[side]['origin'].append(np.array(entry[frame_key]['pos']))
                    F = np.zeros((3,3))
                    F[:,0] = entry[frame_key]['x_axis']
                    F[:,1] = entry[frame_key]['y_axis']
                    F[:,2] = entry[frame_key]['z_axis']
                    sequences[side]['F'].append(F)
                    
        evaluation_errors = {'left': {'trans': 0, 'rot': 0}, 'right': {'trans': 0, 'rot': 0}}
        for side in ['left', 'right']:
            if not sequences[side]['pos_pct']: continue
            
            trans_errors = []
            rot_errors = []
            
            for i in range(len(sequences[side]['pos_pct'])):
                p = sequences[side]['pos_pct'][i]
                d = sequences[side]['direction'][i]
                true_pos = sequences[side]['origin'][i]
                true_F = sequences[side]['F'][i]
                
                pred_pos, pred_F = self.predict(side, p, direction=d)
                if pred_pos is None: continue
                
                # Translation Error (Euclidean distance)
                trans_errors.append(np.linalg.norm(pred_pos - true_pos))
                
                # Rotation Error (Angle of R_err)
                # Ensure trace is within valid mathematical bounds [-1, 3] to avoid acos domain float issues
                R_err = pred_F @ true_F.T
                trace = np.clip(np.trace(R_err), -1.0, 3.0)
                angle = np.arccos((trace - 1.0) / 2.0)
                rot_errors.append(angle)
                
            evaluation_errors[side]['trans'] = float(np.mean(trans_errors))
            evaluation_errors[side]['rot'] = float(np.mean(rot_errors))
            
        return evaluation_errors
        
    def hyperparameter_fit(self, data_list, hyperparameter_grid, verbose=True):
        """
        hyperparameter_grid : dict of lists, e.g. {'s_pos': [0.001, 0.01], 's_angle': [0.01, 0.1]}
        """
        self.hyperparameter_grid = hyperparameter_grid
        keys, values = zip(*hyperparameter_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        # Split data systematically (even/odd interleaving) to form a physical Hold-out Validation Set
        # This absolutely prevents Univariate Splines from over-fitting to calibration noise.
        train_data = data_list[::2]
        val_data = data_list[1::2]
        
        best_score = float('inf')
        best_hp = None
        best_val_errs = None
        
        if verbose:
            print("\n--- Hyperparameter Optimization ---")
            print(f"Dataset split: {len(train_data)} Train frames, {len(val_data)} Validation frames")
            print("Score minimizes MAE on the HELD-OUT Validation set to prevent jitter/overfitting.")
            print("Score = (Translational MAE in mm) + (Rotational MAE in degrees)")
            print("\nHyperparameters being optimized:")
            print("  [s_pos_mult]   -> 3D Position tracking B-Spline smoothing stiffness.")
            print("  [s_theta_mult] -> Orientation angle B-Spline smoothing stiffness.")
            print("  [ransac_dist]  -> Metric threshold for identifying structural in-liers on the 3D plane.")
            print("  [hysteresis_degree] -> 0 for scalar, 1 for linear mapping, evaluating gap compensation.")
            print("\nStarting grid search...")
            
        for hp in combinations:
            self.fit(train_data, hyperparameters=hp, verbose=False)
            val_errs = self.evaluate(val_data)
            score = self.get_score(val_errs)
            
            if verbose:
                trans_mae = (val_errs['left']['trans'] + val_errs['right']['trans']) / 2.0 * 1000
                rot_mae = np.rad2deg((val_errs['left']['rot'] + val_errs['right']['rot']) / 2.0)
                print(f"  [Eval] {hp} -> Score: {score:.2f} (Trans: {trans_mae:.2f}mm, Rot: {rot_mae:.2f}deg)")
                
            if score < best_score:
                best_score = score
                best_hp = hp
                best_val_errs = val_errs
                
        if verbose:
            print(f"\nGrid search complete. Best Configuration: {best_hp}")
            print(f"Achieved optimal unseen Validation Score: {best_score:.4f}")
            if best_val_errs:
                final_trans = (best_val_errs['left']['trans'] + best_val_errs['right']['trans']) / 2.0 * 1000
                final_rot = np.rad2deg((best_val_errs['left']['rot'] + best_val_errs['right']['rot']) / 2.0)
                print(f"Final Validation Translational MAE: {final_trans:.2f} mm")
                print(f"Final Validation Rotational MAE:    {final_rot:.2f} deg\n")
                self.hyperparameter_optimization_results = {
                    'best_score': float(best_score),
                    'score_formula': "(Translational MAE in mm) + (Rotational MAE in degrees)",
                    'training_split_frames': len(train_data),
                    'validation_split_frames': len(val_data),
                    'final_validation_translational_mae_mm': float(final_trans),
                    'final_validation_rotational_mae_deg': float(final_rot)
                }
            
        # Retrain strictly on the entire full aggregated dataset using the discovered best generalization hyper-parameters
        self.fit(data_list, hyperparameters=best_hp, verbose=verbose)
        self.errors = self.evaluate(data_list)

class PlanarFingertipModel(BaseFingertipModel):
    def __init__(self):
        super().__init__()
        self.plane_centroid = None
        self.plane_normal = None
        self.X_plane = None
        self.Y_plane = None
        
        # Splines
        self.splines = {'left': {}, 'right': {}}
        self.base_frames = {'left': None, 'right': None}
        self.default_hyperparameters = {'s_pos_mult': 0.001, 's_theta_mult': 0.01, 'ransac_dist': 0.005}
        
    def fit(self, data_list, hyperparameters=None, verbose=True):
        if hyperparameters is None:
            hyperparameters = self.default_hyperparameters
        self.hyperparameters = hyperparameters
        
        if verbose:
            print("Extracting 3D trajectory points...")
        pts = []
        
        # We need to collect pos_pct, origins, and full frames
        sequences = {'left': {'pos_pct': [], 'origin': [], 'F': []}, 
                     'right': {'pos_pct': [], 'origin': [], 'F': []}}
        
        for entry in data_list:
            pos_pct_before = entry.get('gripper_status_before', {}).get('pos_pct')
            pos_pct_after = entry.get('gripper_status_after', {}).get('pos_pct')
            if pos_pct_before is None or pos_pct_after is None:
                continue
                
            pos_pct = (pos_pct_before + pos_pct_after) / 2.0
            for side in ['left', 'right']:
                frame_key = f"{side}_frame"
                if frame_key in entry:
                    pos = np.array(entry[frame_key]['pos'])
                    F = np.zeros((3,3))
                    F[:,0] = entry[frame_key]['x_axis']
                    F[:,1] = entry[frame_key]['y_axis']
                    F[:,2] = entry[frame_key]['z_axis']
                    
                    pts.append(pos)
                    sequences[side]['pos_pct'].append(pos_pct)
                    sequences[side]['origin'].append(pos)
                    sequences[side]['F'].append(F)
                    
        for side in ['left', 'right']:
            # Sort individual sequences by pos_pct
            seq = sequences[side]
            sort_idx = np.argsort(seq['pos_pct'])
            seq['pos_pct'] = np.array(seq['pos_pct'])[sort_idx]
            seq['origin'] = np.array(seq['origin'])[sort_idx]
            seq['F'] = np.array(seq['F'])[sort_idx]
            
        pts = np.array(pts)
        if len(pts) < 10:
            raise ValueError("Not enough valid data points to fit the model.")

        if verbose:
            print(f"Fitting robust 3D plane using RANSAC (dist={self.hyperparameters['ransac_dist']})...")
            
        self.plane_centroid, self.plane_normal, inliers = robust_plane_fit(pts, distance_threshold=self.hyperparameters['ransac_dist'])
        
        # Ensure normal points consistently (e.g. towards +Y or +Z so it's consistent)
        if self.plane_normal[1] < 0:
            self.plane_normal = -self.plane_normal
            
        # Define 2D plane coordinate system
        # Pick an arbitrary X_plane perpendicular to normal
        arbitrary_up = np.array([0, 1, 0])
        if np.abs(np.dot(self.plane_normal, arbitrary_up)) > 0.99:
            arbitrary_up = np.array([1, 0, 0])
            
        self.X_plane = np.cross(self.plane_normal, arbitrary_up)
        self.X_plane /= np.linalg.norm(self.X_plane)
        self.Y_plane = np.cross(self.plane_normal, self.X_plane)
        self.Y_plane /= np.linalg.norm(self.Y_plane)

        if verbose:
            print("Fitting smooth trajectory and orientation functions...")
        
        self.errors = {'left': {'trans': 0, 'rot': 0}, 'right': {'trans': 0, 'rot': 0}}
        
        for side in ['left', 'right']:
            seq = sequences[side]
            n_pts = len(seq['pos_pct'])
            if n_pts == 0: continue
            
            # Project origins to 2D plane
            diffs = seq['origin'] - self.plane_centroid
            u = np.dot(diffs, self.X_plane)
            v = np.dot(diffs, self.Y_plane)
            
            # Fit smoothing splines for U and V vs pos_pct
            # s controls the smoothness. We want highly smoothed generalized functions.
            spline_u = UnivariateSpline(seq['pos_pct'], u, s=n_pts * self.hyperparameters['s_pos_mult'])
            spline_v = UnivariateSpline(seq['pos_pct'], v, s=n_pts * self.hyperparameters['s_pos_mult'])
            self.splines[side]['u'] = spline_u
            self.splines[side]['v'] = spline_v
            
            # Rotation modeling: assume rotation predominantly around normal
            # Define F_base as the robust median or first frame, then find angle sequence
            # Wait, best to use the frame at highest pos_pct as base? Or just the first one?
            F_base = seq['F'][len(seq['F'])//2]
            
            # Project F_base's X axis into the plane
            base_x_proj = F_base[:,0] - np.dot(F_base[:,0], self.plane_normal) * self.plane_normal
            base_x_proj /= np.linalg.norm(base_x_proj)
            
            # We must construct a completely orthonormal F_base that perfectly aligns its Z(?) or rotation axis with normal
            # Actually, standard projection: The assumed model is F_pred = R(theta, n) @ F_base
            # So F_base can just be exactly seq['F'][mid]
            self.base_frames[side] = F_base
            
            angles = []
            for i in range(n_pts):
                F = seq['F'][i]
                # To find theta around self.plane_normal representing the transition F_base -> F
                # We can find the relative rotation R_rel = F @ F_base^T
                R_rel = F @ F_base.T
                
                # We want to extract the angle of rotation purely around plane_normal
                # R_rel * v should rotate v around plane_normal.
                # A robust way: compare the projection of X_axis.
                x_proj = F[:,0] - np.dot(F[:,0], self.plane_normal) * self.plane_normal
                if np.linalg.norm(x_proj) < 1e-4:
                    # Fallback to Y axis if X is parallel to normal
                    x_proj = F[:,1] - np.dot(F[:,1], self.plane_normal) * self.plane_normal
                    base_x_proj = F_base[:,1] - np.dot(F_base[:,1], self.plane_normal) * self.plane_normal
                    
                x_proj /= np.linalg.norm(x_proj)
                
                # Angle from base_x_proj to x_proj around normal
                cross_p = np.cross(base_x_proj, x_proj)
                sin_th = np.dot(cross_p, self.plane_normal)
                cos_th = np.dot(base_x_proj, x_proj)
                theta = np.arctan2(sin_th, cos_th)
                
                # Unwrap angles to prevent 2*pi jumps
                if i > 0:
                    diff = theta - angles[-1]
                    if diff > np.pi: theta -= 2*np.pi
                    elif diff < -np.pi: theta += 2*np.pi
                        
                angles.append(theta)
                
            angles = np.array(angles)
            spline_theta = UnivariateSpline(seq['pos_pct'], angles, s=n_pts * self.hyperparameters['s_theta_mult'])
            self.splines[side]['theta'] = spline_theta
            
        if verbose:
            print(f"Total Trajectory Fit Complete. Quality computed externally via evaluation sets.")
        
    def predict(self, side, pos_pct, direction='closing'):
        """Predict the 3x3 rotation matrix F and 3D origin position given pos_pct and side."""
        if side not in self.splines or 'u' not in self.splines[side]:
            return None, None
            
        u = self.splines[side]['u'](pos_pct)
        v = self.splines[side]['v'](pos_pct)
        theta = self.splines[side]['theta'](pos_pct)
        
        pos = self.plane_centroid + u * self.X_plane + v * self.Y_plane
        
        # R(theta, n)
        rotvec = theta * self.plane_normal
        R_rel = Rotation.from_rotvec(rotvec).as_matrix()
        
        F_pred = R_rel @ self.base_frames[side]
        return pos, F_pred

    def serialize_spline(self, spline):
        # UnivariateSpline internal data is typically t (knots), c (coeffs), k (degree)
        t, c, k = spline._eval_args
        return {
            'knots': clean_float(t),
            'coeffs': clean_float(c),
            'degree': int(k)
        }

    def save(self, filepath, metadata):
        print(f"Saving model to {filepath}...")
        
        # Construct human-readable payload
        out = {
            'metadata': metadata,
            'hyperparameters': self.hyperparameters,
            'hyperparameter_grid': getattr(self, 'hyperparameter_grid', None),
            'hyperparameter_optimization_results': getattr(self, 'hyperparameter_optimization_results', None),
            'fit_quality': {
                'left': {
                    'translational_mae_m': self.errors['left']['trans'],
                    'rotational_mae_rad': self.errors['left']['rot']
                },
                'right': {
                    'translational_mae_m': self.errors['right']['trans'],
                    'rotational_mae_rad': self.errors['right']['rot']
                }
            },
            'model_type': self.__class__.__name__,
            'plane_geometry': {
                'centroid': clean_float(self.plane_centroid),
                'normal': clean_float(self.plane_normal),
                'X_plane': clean_float(self.X_plane),
                'Y_plane': clean_float(self.Y_plane)
            },
            'fingertips': {
                'left': {
                    'base_frame': clean_float(self.base_frames['left']),
                    'spline_u': self.serialize_spline(self.splines['left']['u']),
                    'spline_v': self.serialize_spline(self.splines['left']['v']),
                    'spline_theta': self.serialize_spline(self.splines['left']['theta'])
                },
                'right': {
                    'base_frame': clean_float(self.base_frames['right']),
                    'spline_u': self.serialize_spline(self.splines['right']['u']),
                    'spline_v': self.serialize_spline(self.splines['right']['v']),
                    'spline_theta': self.serialize_spline(self.splines['right']['theta'])
                }
            }
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(out, f, sort_keys=False, default_flow_style=False)



import scipy.optimize as opt

class HysteresisPlanarFingertipModel(PlanarFingertipModel):
    def __init__(self):
        super().__init__()
        self.hysteresis_poly = {'left': None, 'right': None}
        
    def fit(self, data_list, hyperparameters=None, verbose=True):
        if hyperparameters is not None:
            self.hyperparameters = hyperparameters
            
        closing_data = []
        opening_data = []
        for d in data_list:
            pos_pct_b = d.get('gripper_status_before', {}).get('pos_pct')
            pos_pct_a = d.get('gripper_status_after', {}).get('pos_pct')
            if pos_pct_b is None or pos_pct_a is None: continue
            if pos_pct_a > pos_pct_b:
                opening_data.append(d)
            else:
                closing_data.append(d)
                
        if len(closing_data) == 0:
            if verbose: print("Warning: No closing data found. Falling back to all data.")
            closing_data = data_list
            
        # Fit structural geometry and splines STRICTLY to the precision tension closing phase
        super().fit(closing_data, hyperparameters, verbose=False)
        
        degree = self.hyperparameters.get('hysteresis_degree', 0)
        self.hysteresis_poly = {'left': None, 'right': None}
        
        open_seq = {'left': {'p':[], 'pos':[]}, 'right': {'p':[], 'pos':[]}}
        for d in opening_data:
            pos_pct_b = d.get('gripper_status_before', {}).get('pos_pct')
            pos_pct_a = d.get('gripper_status_after', {}).get('pos_pct')
            p = (pos_pct_b + pos_pct_a) / 2.0
            for side in ['left', 'right']:
                if f"{side}_frame" in d:
                    open_seq[side]['p'].append(p)
                    open_seq[side]['pos'].append(np.array(d[f"{side}_frame"]['pos']))
                    
        for side in ['left', 'right']:
            if not open_seq[side]['p']: continue
            
            p_vals = open_seq[side]['p']
            pos_true = open_seq[side]['pos']
            h_vals = []
            valid_p = []
            for i in range(len(p_vals)):
                p = p_vals[i]
                true_pos = pos_true[i]
                
                def loss(eff_p):
                    pred_pos, _ = PlanarFingertipModel.predict(self, side, eff_p) # Bypass our overridden signature
                    if pred_pos is None: return 1000.0
                    return float(np.linalg.norm(pred_pos - true_pos))
                    
                res = opt.minimize_scalar(loss, bounds=(p - 150, p + 150), method='bounded')
                if res.success:
                    h_vals.append(p - res.x)
                    valid_p.append(p)
                    
            if len(h_vals) > 0:
                if degree == 0:
                    self.hysteresis_poly[side] = np.poly1d([np.mean(h_vals)])
                else:
                    self.hysteresis_poly[side] = np.poly1d(np.polyfit(valid_p, h_vals, deg=degree))
            
        if verbose:
            print(f"Total Asymmetric Model Fit Complete. Hysteresis Modeled (Deg {degree}).")

    def predict(self, side, pos_pct, direction='closing'):
        eff_pos_pct = pos_pct
        if direction == 'opening' and side in self.hysteresis_poly and self.hysteresis_poly[side] is not None:
            eff_pos_pct = pos_pct - self.hysteresis_poly[side](pos_pct)
        return super().predict(side, eff_pos_pct, direction)
        
    def save(self, filepath, metadata):
        super().save(filepath, metadata)
        with open(filepath, 'r') as f:
            doc = yaml.safe_load(f)
        doc['model_type'] = self.__class__.__name__
        doc['metadata']['description'] = "Hysteresis-compensated Planar 3D trajectory model."
        doc['hysteresis_compensation'] = {
            'left': {
                'degree': len(self.hysteresis_poly['left'].coeffs) - 1 if self.hysteresis_poly['left'] is not None else -1,
                'coeffs': [float(c) for c in self.hysteresis_poly['left'].coeffs] if self.hysteresis_poly['left'] is not None else []
            },
            'right': {
                'degree': len(self.hysteresis_poly['right'].coeffs) - 1 if self.hysteresis_poly['right'] is not None else -1,
                'coeffs': [float(c) for c in self.hysteresis_poly['right'].coeffs] if self.hysteresis_poly['right'] is not None else []
            }
        }
        with open(filepath, 'w') as f:
            yaml.dump(doc, f, sort_keys=False, default_flow_style=False)

def main():
    parser = argparse.ArgumentParser(description='Fit a kinematic model to gripper calibration data.')
    parser.add_argument('calib_dir', type=str, help='Path to the timestamped gripper calibration directory')
    args = parser.parse_args()

    calib_dir = os.path.abspath(args.calib_dir)
    
    if not os.path.isdir(calib_dir):
        print(f"Error: {calib_dir} is not a valid directory.")
        sys.exit(1)

    # 1. Load Calibration Data
    yaml_files = glob.glob(os.path.join(calib_dir, "*_gripper_data_*.yaml"))
    
    if not yaml_files:
        print("Error: No YAML data files found in the directory.")
        sys.exit(1)
        
    combined_data = []
    robot_id = 'unknown'
    earliest_time = None
    
    for yf in yaml_files:
        with open(yf, 'r') as f:
            doc = yaml.safe_load(f)
            
        metadata = doc.get('metadata', {})
        if 'robot_id' in metadata and robot_id == 'unknown':
            robot_id = metadata['robot_id']
            
        combined_data.extend(doc.get('data', []))
        
    print(f"Loaded {len(combined_data)} synchronized data frames.")

    # 2. Fit the Model via Grid Search
    model = HysteresisPlanarFingertipModel()
    
    # Evaluating Hysteresis (Scalar vs Linear) explicitly
    hyperparameter_grid = {
        's_pos_mult': [0.01, 0.05],
        's_theta_mult': [0.5, 1.0],
        'ransac_dist': [0.001, 0.002],
        'hysteresis_degree': [0, 1]
    }
    
    model.hyperparameter_fit(combined_data, hyperparameter_grid)
    
    # 3. Serialize
    current_time_str = time.strftime("%Y%m%d_%H%M%S")
    metadata = {
        'robot_id': robot_id,
        'model_fit_timestamp': current_time_str,
        'description': "Planar 3D trajectory model with B-Splines for origin and angle vs pos_pct.",
        'source_directory': calib_dir
    }
    
    model_path = os.path.join(calib_dir, f"model_planar_{current_time_str}.yaml")
    model.save(model_path, metadata)
    
    # Save a copy to the standard Hello Robot calibration directory
    fleet_path = os.environ.get('HELLO_FLEET_PATH')
    fleet_id = os.environ.get('HELLO_FLEET_ID', robot_id)
    if fleet_path and fleet_id and fleet_id != 'unknown':
        calib_gripper_dir = os.path.join(fleet_path, fleet_id, 'calibration_gripper')
        os.makedirs(calib_gripper_dir, exist_ok=True)
        latest_path = os.path.join(calib_gripper_dir, "latest_model_planar.yaml")
        shutil.copy2(model_path, latest_path)
        print(f"Copied model to fleet directory: {latest_path}")
    
    print("Done! Optimization and Model Serialization complete.")


if __name__ == '__main__':
    main()
