#!/usr/bin/env python3

import os
import sys
import yaml
import argparse
import numpy as np
import rerun as rr
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation
from stretch4_gripper_modeling_and_control.aruco_to_fingertips import ArucoToFingertips
from stretch4_gripper_modeling_and_control import calibration_utils as cu
from stretch4_gripper_modeling_and_control import visualization_utils as vu

MIN_PCT = -100
MAX_PCT = 300

from stretch4_gripper_modeling_and_control.fingertip_visualizer import FingertipVisualizer


def draw_axes(entity_path, T, size=0.02, radius=0.001, static=False):
    rr.log(entity_path, rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]), static=static)
    rr.log(f"{entity_path}/axes", rr.Arrows3D(
        vectors=[[size, 0, 0], [0, size, 0], [0, 0, size]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        radii=radius
    ), static=static)

def main():
    parser = argparse.ArgumentParser(description='Visualize the 3D Fingertip Kinematic Model using Rerun.')
    parser.add_argument('model_path', type=str, nargs='?', default=None, help='Path to the model YAML file. If not provided, defaults to the latest fleet calibration model.')
    vu.add_suction_cup_argument(parser)
    args = parser.parse_args()
    
    if args.model_path is None:
        args.model_path = cu.get_default_model_path()
        if args.model_path:
            print(f"No model path provided. Defaulting to fleet calibration: {args.model_path}")
        else:
            print("Error: No model path provided and could not locate latest_model_planar.yaml in fleet directory.")
            sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"File not found: {args.model_path}")
        sys.exit(1)
        
    model = FingertipVisualizer(args.model_path)
    
    import rerun.blueprint as rrb
    
    try:
        blueprint = rrb.Blueprint(
            rrb.Spatial3DView(
                name="Fingertip Visualization",
                origin="world",
                item_visibilities={
                    "world/3d_plane/normal": False,
                    "world/aruco_markers": False
                }
            )
        )
        rr.init("Fingertip Model Visualizer", spawn=True, default_blueprint=blueprint)
    except Exception:
        rr.init("Fingertip Model Visualizer", spawn=True)
    
    # 1. Static Geometry
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True) 
    
    T_model = np.eye(4)
    normal = model.normal.copy()
    centroid = model.centroid.copy()
    
    # 1. Ensure camera is "above" the plane
    if np.dot(-centroid, normal) < 0:
        normal = -normal
        
    # 2. Project camera Z-axis onto the plane to define Y-axis (Rerun Up)
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
    
    T_model[:3, 0] = X_new
    T_model[:3, 1] = Y_new
    T_model[:3, 2] = normal
    T_model[:3, 3] = centroid
    T_W_C = np.linalg.inv(T_model)
    
    draw_axes("world/camera", T_W_C, size=0.05, static=True)
    
    pts_l, pts_r = [], []
    pcts = np.linspace(MAX_PCT, MIN_PCT, 400) # Full sweeping curves
    for p in pcts:
        pos_l, _ = model.predict('left', p)
        if pos_l is not None:
            pts_l.append((T_W_C @ np.append(pos_l, 1.0))[:3])
        pos_r, _ = model.predict('right', p)
        if pos_r is not None:
            pts_r.append((T_W_C @ np.append(pos_r, 1.0))[:3])
        
    if pts_l: rr.log("world/trajectories/left", rr.LineStrips3D([pts_l], colors=[[255, 255, 255]], radii=0.001), static=True)
    if pts_r: rr.log("world/trajectories/right", rr.LineStrips3D([pts_r], colors=[[255, 255, 255]], radii=0.001), static=True)
    
    s = 0.08  # 16cm x 16cm plane
    c = np.array([0, 0, 0])
    dx = np.array([s, 0, 0])
    dy = np.array([0, s, 0])
    plane_pts = [c + dx + dy, c - dx + dy, c - dx - dy, c + dx - dy, c + dx + dy]
    rr.log("world/3d_plane/bounds", rr.LineStrips3D([plane_pts], colors=[[128, 128, 128]], radii=0.0005), static=True)    # Gray bounds
    rr.log("world/3d_plane/normal", rr.Arrows3D(origins=[c], vectors=[[0, 0, 0.02]], colors=[[255, 255, 0]], radii=0.001), static=True) # Yellow normal

    if not args.disable_suction_cups:
        from stretch4_gripper_modeling_and_control.aruco_to_fingertips import suctioncup_height, suctioncup_sphere_radius, cylinder_radius, ArucoToFingertips
        
        a2f = ArucoToFingertips()
        current_height = a2f.default_height_above_mounting_surface if a2f.default_height_above_mounting_surface is not None else 0.0
        
        h_cup_top = suctioncup_height.get('cup_top', 0.015)
        h_cyl_top = suctioncup_height.get('cylinder_top', 0.00415)
        h_cyl_bot = suctioncup_height.get('cylinder_bottom', 0.0)
        R_s = suctioncup_sphere_radius
        
        z_c_offset = np.sqrt(max(0, R_s**2 - cylinder_radius**2))
        Z_c_local = h_cyl_top + z_c_offset
        
        num_points = 24
        num_cap_rings = 5
        rings_3d = []
        
        ring_bot = []
        for i in range(num_points):
            ring_bot.append((h_cyl_bot, cylinder_radius, 2.0 * np.pi * i / num_points))
        rings_3d.append(ring_bot)
        
        ring_cyl_top = []
        for i in range(num_points):
            ring_cyl_top.append((h_cyl_top, cylinder_radius, 2.0 * np.pi * i / num_points))
        rings_3d.append(ring_cyl_top)
        
        for z_val in np.linspace(h_cyl_top, h_cup_top, num_cap_rings + 1)[1:]: 
            r_val = np.sqrt(max(0, R_s**2 - (Z_c_local - z_val)**2))
            ring = []
            for i in range(num_points):
                ring.append((z_val, r_val, 2.0 * np.pi * i / num_points))
            rings_3d.append(ring)
            
        verts = []
        normals = []
        base_z = -current_height
        
        verts.append((0, 0, base_z + h_cup_top))   # center top
        normals.append((0, 0, 1.0))
        
        verts.append((0, 0, base_z + h_cyl_bot))   # center bot
        normals.append((0, 0, -1.0))
        
        ring_start_idx = len(verts)
        
        for r_idx, ring in enumerate(rings_3d):
            for (z_val, r_val, theta) in ring:
                x = r_val * np.cos(theta)
                y = r_val * np.sin(theta)
                verts.append((x, y, base_z + z_val))
                
                if r_idx < 2: # Cylinder walls
                    normals.append((np.cos(theta), np.sin(theta), 0.0))
                else: # Spherical cap
                    nx, ny, nz = x, y, (z_val - Z_c_local)
                    n_mag = np.sqrt(nx**2 + ny**2 + nz**2)
                    if n_mag > 1e-6:
                        normals.append((nx/n_mag, ny/n_mag, nz/n_mag))
                    else:
                        normals.append((0, 0, 1.0))
                
        tris = []
        center_idx = 1 # bot
        ring0_base = ring_start_idx
        for i in range(num_points):
            tris.append((center_idx, ring0_base + ((i + 1) % num_points), ring0_base + i))
            
        for r_idx in range(len(rings_3d) - 1):
            lb = ring_start_idx + r_idx * num_points
            ub = ring_start_idx + (r_idx + 1) * num_points
            for i in range(num_points):
                n_i = (i + 1) % num_points
                tris.append((lb + i, lb + n_i, ub + i))
                tris.append((lb + n_i, ub + n_i, ub + i))
                
        center_idx = 0 # top
        tr = ring_start_idx + (len(rings_3d) - 1) * num_points
        for i in range(num_points):
            tris.append((center_idx, tr + i, tr + ((i + 1) % num_points)))
            
        color = [40, 150, 255] # nice blue
        
        # PROMINENT BOOLEAN: Set to True to overlay wireframe outlines over the 3D meshes
        DRAW_WIREFRAMES = False
        
        for side in ['left', 'right']:
            rr.log(f"world/fingertips/{side}/suction_cup", 
                   rr.Mesh3D(vertex_positions=verts, 
                             vertex_normals=normals,
                             triangle_indices=tris, 
                             vertex_colors=[color]*len(verts)), 
                   static=True)
            
            if DRAW_WIREFRAMES:
                # Add wireframes for enhanced structural cues
                wireframe_strips = []
                for r_idx, ring in enumerate(rings_3d):
                    strip = []
                    for (z_val, r_val, theta) in ring:
                        strip.append((r_val * np.cos(theta), r_val * np.sin(theta), base_z + z_val))
                    strip.append(strip[0]) # close loop
                    wireframe_strips.append(strip)
                    
                rr.log(f"world/fingertips/{side}/suction_cup/edges", rr.LineStrips3D(wireframe_strips, colors=[[0, 50, 100]], radii=0.0005), static=True)

    # Helper to calculate and log fingertip frames
    def log_frame(pct, direction):
        for side in ['left', 'right']:
            pos, F = model.predict(side, pct, direction=direction)
            if pos is None: continue
            
            T_finger = np.eye(4)
            T_finger[:3, :3] = F
            T_finger[:3, 3] = pos
            
            T_finger_w = T_W_C @ T_finger
            draw_axes(f"world/fingertips/{side}", T_finger_w, size=0.02)
            
            inv_trans = np.linalg.inv(model.a2f_transforms[side])
            T_marker = T_finger @ inv_trans
            T_marker_w = T_W_C @ T_marker
            draw_axes(f"world/aruco_markers/{side}", T_marker_w, size=0.015)

    # 2. Dynamic Timelines (closing then opening)
    step = 0
    SPEED_FACTOR = 4  # Skip values to playback at a visually normal speed
    
    # Gripper Closing: 300 to -100
    for pct in range(MAX_PCT, MIN_PCT - 1, -SPEED_FACTOR):
        rr.set_time("step", sequence=step)
        rr.log("world/pos_pct", rr.Scalars(pct))
        log_frame(pct, direction='closing')
        step += 1
        
    # Gripper Opening: -100 to 300
    for pct in range(MIN_PCT, MAX_PCT + 1, SPEED_FACTOR):
        rr.set_time("step", sequence=step)
        rr.log("world/pos_pct", rr.Scalars(pct))
        log_frame(pct, direction='opening')
        step += 1

    print("Done generating sequence! Native interface should open automatically.")
    print("Use the timeline slider at the bottom to interact!")

if __name__ == "__main__":
    main()
