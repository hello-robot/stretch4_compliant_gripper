#!/usr/bin/env python3

import cv2
import numpy as np
import cv2.aruco as aruco
from stretch4_gripper_modeling_and_control import aruco_config

_module_a_to_f = None
_module_finger_rots = None

def get_finger_rots():
    global _module_a_to_f, _module_finger_rots
    if _module_finger_rots is None:
        from stretch4_gripper_modeling_and_control import aruco_to_fingertips as af
        _module_a_to_f = af.ArucoToFingertips(default_height_above_mounting_surface=af.suctioncup_height['cup_bottom'])
        _module_finger_rots = _module_a_to_f.get_rotations()
    return _module_finger_rots



def minimum_distance_between_corners(corners):
    # calculate the 6 distances between the corners and return the minimum
    c0 = corners[0]
    dist0 = np.min(np.linalg.norm(corners[1:4] - c0, axis=1))
    c1 = corners[1]
    dist1 = np.min(np.linalg.norm(corners[2:4] - c1, axis=1))
    c2 = corners[2]
    dist2 = np.min(np.linalg.norm(corners[3:4] - c2, axis=1))
    return np.min(np.array([dist0, dist1, dist2]))
    

def is_feasible_finger_marker_pose(marker_position, marker_name):
    """
    Modular constraint check to filter out false positive markers 
    that are physically too far to be on the gripper.
    """
    if marker_name not in ['finger_left', 'finger_right']:
        return True, ""
        
    x, y, z = marker_position
    
    # Tunable boundary cuboid constraints in meters
    max_depth = 0.30
    max_horizontal = 0.20
    max_vertical = 0.15
    
    if not (0.0 < z < max_depth):
        return False, f"Depth {z:.3f}m is not between 0 and {max_depth}m"
    if abs(x) > max_horizontal:
        return False, f"Horizontal abs({x:.3f}m) exceeds {max_horizontal}m"
    if abs(y) > max_vertical:
        return False, f"Vertical abs({y:.3f}m) exceeds {max_vertical}m"
        
    return True, ""


class ArucoMarker:
    def __init__(self, aruco_id, marker_info, config=None, show_debug_images=False, validator=None):
        self.config = config
        self.show_debug_images = show_debug_images
        self.validator = validator
        
        self.aruco_id = aruco_id
        colormap = cv2.COLORMAP_HSV
        offset = 0
        i = (offset + (self.aruco_id * 29)) % 255
        image = np.uint8([[[i]]])
        id_color_image = cv2.applyColorMap(image, colormap)
        bgr = id_color_image[0,0]
        self.id_color = [bgr[2], bgr[1], bgr[0]]
        
        self.frame_id = 'camera_color_optical_frame'
        self.info = marker_info.get(str(self.aruco_id), None)

        if self.info is None:
            self.info = marker_info['default']
        self.length_of_marker_mm = self.info['length_mm']
        self.use_rgb_only = self.info['use_rgb_only']
        
        self.frame_number = None
        self.ready = False
        self.x_axis = None
        self.y_axis = None
        self.z_axis = None
        self.min_dist_between_corners = None
        
    
    def update(self, corners, frame_number, rgb_camera_info, pos_pct=None):
        camera_matrix = rgb_camera_info['camera_matrix']
        distortion_coefficients = rgb_camera_info['distortion_coefficients']

        points_3D = np.array([
            (-self.length_of_marker_mm / 2, self.length_of_marker_mm / 2, 0),
            (self.length_of_marker_mm / 2, self.length_of_marker_mm / 2, 0),
            (self.length_of_marker_mm / 2, -self.length_of_marker_mm / 2, 0),
            (-self.length_of_marker_mm / 2, -self.length_of_marker_mm / 2, 0),
        ])

        name = self.info.get('name') if self.info else None

        if name == 'finger_left':
            # -------------------------------------------------------------------------
            # Chiral Mirroring Heuristic for IPPE Solver Stability
            # -------------------------------------------------------------------------
            # The OpenCV IPPE solver (SOLVEPNP_IPPE_SQUARE) has a numerical "chiral bias"
            # dependent on the ordering of the marker corners. Since the physical fingers
            # open asymmetrically with respect to the camera, the left marker is viewed
            # from an oblique angle that causes the solver's numerical gradient to 
            # subtly prefer the incorrect "flipped" ambiguity pose (lowest reprojection error).
            # The right finger's geometry naturally avoids this issue.
            #
            # To fix this, we mathematically mirror the 2D image points across the central 
            # vertical axis (X = cx) and reorder the corners. This visually perfectly 
            # transforms the left marker's input to look identically like the stable right 
            # marker's geometry to the IPPE solver.
            # -------------------------------------------------------------------------
            cx = camera_matrix[0, 2]
            corners_mirrored = corners.copy()
            corners_mirrored[:, 0] = 2 * cx - corners[:, 0]
            
            # Reorder to match standard ArUco corner sequence in the mirrored view:
            # Original: 0 (TL), 1 (TR), 2 (BR), 3 (BL)
            # Mirrored: 1 becomes TL, 0 becomes TR, 3 becomes BR, 2 becomes BL
            reordered_corners = np.zeros_like(corners_mirrored)
            reordered_corners[0] = corners_mirrored[1]
            reordered_corners[1] = corners_mirrored[0]
            reordered_corners[2] = corners_mirrored[3]
            reordered_corners[3] = corners_mirrored[2]

            retval, rvecs_ret_all, tvecs_ret_all, reproj_errs = cv2.solvePnPGeneric(
                objectPoints=points_3D,
                imagePoints=reordered_corners,
                cameraMatrix=camera_matrix,
                distCoeffs=distortion_coefficients,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            # Map the solutions back from the right-handed mirrored universe.
            # The reflection matrix Ref = diag(-1, 1, 1) inverts the camera X-axis.
            # Applying Ref @ R_m @ Ref maps the mirrored orientation back to the true left frame.
            Ref = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
            rvecs_ret_all = list(rvecs_ret_all)
            tvecs_ret_all = list(tvecs_ret_all)
            for i in range(len(rvecs_ret_all)):
                R_m, _ = cv2.Rodrigues(rvecs_ret_all[i])
                R_orig = Ref @ R_m @ Ref
                rvecs_ret_all[i], _ = cv2.Rodrigues(R_orig)
                
                t_m = tvecs_ret_all[i].reshape(3, 1)
                t_orig = Ref @ t_m
                tvecs_ret_all[i] = t_orig
        else:
            retval, rvecs_ret_all, tvecs_ret_all, reproj_errs = cv2.solvePnPGeneric(
                objectPoints=points_3D,
                imagePoints=corners,
                cameraMatrix=camera_matrix,
                distCoeffs=distortion_coefficients,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            rvecs_ret_all = list(rvecs_ret_all)
            tvecs_ret_all = list(tvecs_ret_all)
        
        # Default to the first solution (lowest reprojection error)
        best_idx = 0
        
        # Heuristic: True normal axis for the right fingertip's ArUco marker will almost always point to the left side 
        # of the image (-X direction), and the left fingertip’s ArUco marker will almost always point to the right side 
        # of the image (+X direction).
        if name in ['finger_left', 'finger_right']:
            if len(rvecs_ret_all) > 1:
                side = 'left' if name == 'finger_left' else 'right'
                best_score = float('-inf')
                for i in range(len(rvecs_ret_all)):
                    A, _ = cv2.Rodrigues(rvecs_ret_all[i])
                    score = A[0,2] if side == 'left' else -A[0,2]
                    if score > best_score:
                        best_score = score
                        best_idx = i

        # Non-Linear Pose Refinement
        if getattr(self, 'config', None) is not None and self.config.enable_pose_refinement:
            try:
                initial_rvec = rvecs_ret_all[best_idx].copy()
                initial_tvec = tvecs_ret_all[best_idx].copy()
                
                # We use the original corners against the original camera matrix,
                # as IPPE mapped the solution back to the physical coordinate system
                if self.config.pose_refinement_method == 'VVS':
                    initial_rvec, initial_tvec = cv2.solvePnPRefineVVS(
                        points_3D, corners, camera_matrix, distortion_coefficients, initial_rvec, initial_tvec
                    )
                else: # Default LM
                    initial_rvec, initial_tvec = cv2.solvePnPRefineLM(
                        points_3D, corners, camera_matrix, distortion_coefficients, initial_rvec, initial_tvec
                    )
                rvecs_ret_all[best_idx] = initial_rvec
                tvecs_ret_all[best_idx] = initial_tvec
            except Exception as e:
                print(f"Notification: Pose refinement failed: {e}")

        # Convert ArUco position estimate to be in meters.
        aruco_position_est = tvecs_ret_all[best_idx].reshape(-1) / 1000.0
        
        # Modular bounding box constraints
        is_feasible, reason = is_feasible_finger_marker_pose(aruco_position_est, name)
        if not is_feasible:
            print(f"Notification: Excluded detected ArUco marker {self.aruco_id} ({name}) due to infeasible 3D pose. Reason: {reason}.")
            return
            
        # Valid pose, persist the local evaluation variables to self
        self.corners = corners
        self.frame_number = frame_number
        self.rgb_camera_info = rgb_camera_info
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients

        rvecs = np.zeros((1, 1, 3), dtype=np.float64)
        tvecs = np.zeros((1, 1, 3), dtype=np.float64)
        rvecs[0][:] = np.transpose(rvecs_ret_all[best_idx])
        tvecs[0][:] = np.transpose(tvecs_ret_all[best_idx])
        self.aruco_rotation = rvecs[0][0]
        self.aruco_position = aruco_position_est
        
        aruco_depth_estimate = self.aruco_position[2]
        self.marker_position = self.aruco_position
        R = np.identity(4)
        R[:3,:3] = cv2.Rodrigues(self.aruco_rotation)[0]
        self.x_axis = R[:3,0]
        self.y_axis = R[:3,1]
        self.z_axis = R[:3,2]

        self.rvecs_ret_all = rvecs_ret_all
        self.tvecs_ret_all = tvecs_ret_all
        self.best_idx = best_idx

        self.ready = True

    def get_min_dist_between_corners(self):
        return minimum_distance_between_corners(self.corners)


    def get_position_and_axes(self):
        # return copies of the position and axes
        pos = np.array(self.marker_position)
        x_axis = np.array(self.x_axis)
        y_axis = np.array(self.y_axis)
        z_axis = np.array(self.z_axis)
        return pos, x_axis, y_axis, z_axis

    def get_info(self):
        # return copy of marker_info
        return self.info.copy()

    def get_marker_poly(self):
        poly_points = np.array(self.corners)
        poly_points = np.round(poly_points).astype(np.int32)
        return poly_points

    def draw_marker_poly(self, image): 
        poly_points = self.get_marker_poly()
        cv2.fillConvexPoly(image, poly_points, (255, 0, 0))
        
     
class ArucoMarkerCollection:
    def __init__(self, marker_info, config=None, show_debug_images=False, use_apriltag_refinement=False, brighten_images=False, validator=None):
        self.show_debug_images = show_debug_images
        self.config = config
        self.use_apriltag_refinement = use_apriltag_refinement
        self.validator = validator
                
        self.marker_info = marker_info
        
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        
        # Detector is initialized in update() dynamically based on resolution
        self.detector = None
        self.last_resolution = None
            
        self.collection = {}
        self.frame_number = 0

        # We keep the old parameters for backwards compatibility if config is not provided
        self.brighten_images = brighten_images if config is None else config.brighten_images
        if self.brighten_images: 
            self.adaptive_equalization = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        else:
            self.adaptive_equalization = None

        
    def __iter__(self):
        # iterates through currently visible ArUco markers
        keys = self.collection.keys()
        for k in keys:
            marker = self.collection[k]
            if marker.frame_number == self.frame_number:
                yield marker

    def draw_markers(self, image):
        return aruco.drawDetectedMarkers(image, self.aruco_corners, self.aruco_ids)

    def update(self, rgb_image, rgb_camera_info, pos_pct=None):
        self.frame_number += 1
        self.rgb_image = rgb_image
        self.rgb_camera_info = rgb_camera_info
        self.gray_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)

        # Equalize the gray scale image to improve ArUco marker
        # detection in low exposure time images. Low exposure reduces
        # motion blur, which interferes with ArUco detecction.
        #
        # https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
        #
        #self.gray_image = cv2.equalizeHist(self.gray_image)
        if self.adaptive_equalization is not None: 
            self.gray_image = self.adaptive_equalization.apply(self.gray_image)
        
        image_height, image_width = self.gray_image.shape
        resolution = (image_width, image_height)
        
        if self.detector is None or self.last_resolution != resolution:
            if self.config is not None:
                params = self.config.get_scaled_detector_parameters(image_width, image_height)
                fallback_params = self.config.get_scaled_detector_parameters(image_width, image_height)
                fallback_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
            else:
                params = aruco.DetectorParameters()
                fallback_params = aruco.DetectorParameters()
                # Use fallback options if no config provided
                if getattr(self, 'use_apriltag_refinement', False):
                    params.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
                else: 
                    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
                fallback_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
            
            self.detector = aruco.ArucoDetector(self.aruco_dict, params)
            self.fallback_detector = aruco.ArucoDetector(self.aruco_dict, fallback_params)
            self.last_resolution = resolution
        
        self.aruco_corners, self.aruco_ids, aruco_rejected_image_points = self.detector.detectMarkers(self.gray_image)
        
        # Dual-Refinement Cascade:
        # If AprilTag refinement is enabled, it may drop markers in poor lighting due to strict straight-line gradient 
        # requirements. We can re-run detection using SUBPIX refinement exclusively for any missing markers, 
        # giving us the "best of both worlds" (high precision when well-lit, robust tracking when poorly lit).
        is_apriltag = getattr(self.config, 'use_apriltag_refinement', False) if self.config else getattr(self, 'use_apriltag_refinement', False)
        if is_apriltag:
            expected_marker_ids = [int(k) for k in self.marker_info.keys() if k != 'default']
            current_ids = set([int(i[0]) for i in self.aruco_ids]) if self.aruco_ids is not None else set()
            
            # Check if we are missing any expected markers
            missing_ids = [mid for mid in expected_marker_ids if mid not in current_ids]
            
            if missing_ids:
                fb_corners, fb_ids, _ = self.fallback_detector.detectMarkers(self.gray_image)
                if fb_ids is not None:
                    # Merge fallback detections for markers we missed
                    fb_ids_flat = [int(i[0]) for i in fb_ids]
                    for i, fb_id in enumerate(fb_ids_flat):
                        if fb_id in missing_ids:
                            # Add this marker to our primary lists
                            if self.aruco_ids is None:
                                self.aruco_ids = np.array([[fb_id]], dtype=np.int32)
                                self.aruco_corners = tuple([fb_corners[i]])
                            else:
                                self.aruco_ids = np.vstack((self.aruco_ids, np.array([[fb_id]], dtype=np.int32)))
                                self.aruco_corners = self.aruco_corners + (fb_corners[i],)
            
        if self.aruco_ids is None:
            num_detected = 0
        else:
            num_detected = len(self.aruco_ids)
        
        if self.aruco_ids is not None: 
            for corners, aruco_id in zip(self.aruco_corners, self.aruco_ids):
                aruco_id = int(aruco_id[0])
                marker = self.collection.get(aruco_id, None)
                if marker is None:
                    new_marker = ArucoMarker(aruco_id, self.marker_info, config=self.config, show_debug_images=self.show_debug_images, validator=self.validator)
                    self.collection[aruco_id] = new_marker

                self.collection[aruco_id].update(corners[0], self.frame_number, self.rgb_camera_info, pos_pct)

    
class ArucoDetector():
    def __init__(self, marker_info=None, config=None, show_debug_images=False, use_apriltag_refinement=False, brighten_images=False, validator=None):
        self.rgb_image = None
        self.camera_info = None
        self.all_points = []
        self.show_debug_images = show_debug_images
        self.config = config
        self.use_apriltag_refinement = use_apriltag_refinement
        self.brighten_images = brighten_images
        self.validator = validator
        self.publish_marker_point_clouds = False
        self.marker_info = marker_info

        if self.marker_info is None:
            self.marker_info = {}
            
        self.aruco_marker_collection = ArucoMarkerCollection(
            self.marker_info, config=self.config, show_debug_images=self.show_debug_images, 
            use_apriltag_refinement=self.use_apriltag_refinement, brighten_images=self.brighten_images,
            validator=self.validator
        )

        
    def update(self, rgb_image, rgb_camera_info, pos_pct=None):
        self.rgb_image = rgb_image
        self.rgb_camera_info = rgb_camera_info
            
        self.aruco_marker_collection.update(self.rgb_image, self.rgb_camera_info, pos_pct)
        
        # save rotation for last
        if self.show_debug_images:
            aruco_image = self.aruco_marker_collection.draw_markers(self.rgb_image)
            #display_aruco_image = cv2.rotate(aruco_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.namedWindow('Detected ArUco Markers', cv2.WINDOW_NORMAL)
            #cv2.imshow('Detected ArUco Markers', display_aruco_image)
            cv2.imshow('Detected ArUco Markers', aruco_image)
            cv2.waitKey(1)

    def get_detected_marker_dict(self):
        out = {}
        for m in self.aruco_marker_collection:
            aruco_id = m.aruco_id
            pos, x_axis, y_axis, z_axis = m.get_position_and_axes()
            min_dist_between_corners = m.get_min_dist_between_corners()
            info = m.get_info()
            
            alt_rvec = None
            alt_tvec = None
            if hasattr(m, 'rvecs_ret_all') and len(m.rvecs_ret_all) > 1:
                alt_idx = 1 - m.best_idx
                if alt_idx < len(m.rvecs_ret_all):
                    alt_rvec = m.rvecs_ret_all[alt_idx]
                    alt_tvec = m.tvecs_ret_all[alt_idx]

            out[aruco_id] = {'pos': pos,
                             'x_axis': x_axis, 'y_axis': y_axis, 'z_axis': z_axis,
                             'min_dist_between_corners': min_dist_between_corners,
                             'alt_rvec': alt_rvec,
                             'alt_tvec': alt_tvec,
                             'info': info}
        return out
    
    def get_detected_markers(self):
        markers = self.get_detected_marker_dict()
        
        # This changes keys to be marker names to make code less
        # sensitive to marker changes. Ideally, only the ArUco
        # detection code, as informed by the YAML file, cares about
        # the marker numbers.
        new_markers = {}
        for marker_num in markers.keys():
            m = markers[marker_num]
            m['info']['marker_id'] = marker_num
            marker_name = m['info']['name']
            new_markers[marker_name] = m
        return(new_markers)

    
def get_special_frames(marker_dict):
    # only find origins of the special frames via translation
    # rpy rotation not implemented, yet
    info = marker_dict['info']
    frames = info.get('frames')
    out = {}
    if frames is not None:
        marker_pos = marker_dict['pos']
        marker_x_axis = marker_dict['x_axis']
        marker_y_axis = marker_dict['y_axis']
        marker_z_axis = marker_dict['z_axis']
        for k in frames:
            t = frames[k]['trans']
            rpy = frames[k]['rpy']
            frame_pos = marker_pos + (t[0] * marker_x_axis) + (t[1] * marker_y_axis) + (t[2] * marker_z_axis)
            frame_x_axis = np.copy(marker_x_axis)
            frame_y_axis = np.copy(marker_y_axis)
            frame_z_axis = np.copy(marker_z_axis)
            out[k] = {'pos': frame_pos, 'x_axis': frame_x_axis, 'y_axis': frame_y_axis, 'z_axis': frame_z_axis}
    return(out)
    
def main(args=None):
    detector = ArucoDetector()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
    
