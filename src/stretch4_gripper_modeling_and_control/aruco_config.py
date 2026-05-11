import cv2
import math
from dataclasses import dataclass
from typing import Union

# =====================================================================
# DEFAULT REFERENCE RESOLUTION & MARKER DIMENSIONS
# =====================================================================
# ArUco detection relies on several pixel-based window sizes 
# (e.g., thresholding blocks, corner refinement areas). Setting these 
# in absolute pixels makes the algorithm fragile when switching cameras 
# or image streaming resolutions.
#
# To solve this, tune your pixel-based parameters for the resolution below.
# When the image is processed, the code will dynamically scale these 
# parameters by the ratio of the true image dimensions to this reference 
# resolution, preserving consistent algorithmic behavior at any scale. 
#
# 1280x800 is the maximum resolution image received from the 
# wrist-mounted gripper camera, maximizing precision.
# =====================================================================
REFERENCE_RESOLUTION_W = 1280
REFERENCE_RESOLUTION_H = 800

# Specify the approximate *maximum* width and height (in pixels) for the 
# ArUco markers WHEN VIEWED AT REFERENCE_RESOLUTION. 
# 
# "Width and Height" here represents the ENTIRE physical ArUco marker pattern's 
# dimensions inside the image, specifically the solid outer black square, 
# EXCLUDING the surrounding white margin/padding needed for contrast.
#
# Knowing this allows dependent algorithmic parameters (like adaptive threshold 
# sliding windows and contour filters) to be automatically tuned or validated.
MAX_EXPECTED_MARKER_WIDTH_REF = 100
MAX_EXPECTED_MARKER_HEIGHT_REF = 100

@dataclass
class ArucoConfig:
    """
    Configuration for ArUco marker detection and pose estimation.
    Designed for easy experimentation with techniques to reduce 
    millimeter-scale noise and orientation jitter in fingertip tracking.
    """
    
    # ---------------------------------------------------------
    # 1. Corner Refinement (Fidelity / Stability)
    # ---------------------------------------------------------
    # Defines how the raw thresholded corners of the square are refined.
    # From aruco_assessment.md: AprilTag uses line-fitting along the entire 
    # border rather than localized subpixel gradients. This drastically reduces 
    # sensitivity to local pixel noise and provides much higher distance stability.
    # 
    # The system now uses a Dual-Refinement Cascade: it attempts AprilTag first 
    # for low noise, and seamlessly falls back to SUBPIX per-marker if AprilTag 
    # fails due to poor lighting. Setting this to True enables the cascade!
    use_apriltag_refinement: bool = True
    
    # Window size (in pixels at reference resolution) around the corner point 
    # used for the refinement process (SUBPIX or APRILTAG).
    # Default OpenCV value is 5. Larger windows help with high res, but might 
    # include visual clutter.
    corner_refinement_win_size_ref: int = 5
    
    # ---------------------------------------------------------
    # 2. Non-Linear Pose Refinement (Fidelity)
    # ---------------------------------------------------------
    # IPPE is an analytical solver (direct math) used for the initial pose.
    # Non-linear refinement iteratively minimizes the reprojection error 
    # across all extracted subpixel corners for maximum precision.
    # 
    # True = Apply cv2.solvePnPRefineLM or cv2.solvePnPRefineVVS to the IPPE output
    # False = Use the raw IPPE pose direct output.
    enable_pose_refinement: bool = True
    
    # Method to use if enable_pose_refinement is True. 
    # Options: 'LM' (Levenberg-Marquardt - very stable) or 'VVS' (Virtual Visual Servoing)
    pose_refinement_method: str = 'LM'
    
    # ---------------------------------------------------------
    # 3. Image Preprocessing (Robustness)
    # ---------------------------------------------------------
    # Uses Contrast Limited Adaptive Histogram Equalization (CLAHE) to help
    # detect markers in dark scenes or fast shutter-speed/low-exposure settings.
    brighten_images: bool = True
    
    # ---------------------------------------------------------
    # 4. ArUco 3 Tracking Enhancements (Robustness)
    # ---------------------------------------------------------
    # Introduced in OpenCV 4.x, this tracks markers under severe occlusion 
    # using global contour analysis.
    use_aruco3_detection: bool = False
    
    # ---------------------------------------------------------
    # 5. Adaptive Thresholding (Detection Sensitivity)
    # ---------------------------------------------------------
    # ArUco first applies an adaptive threshold across a sliding window to block 
    # out the image. These set the min, max, and step size of that sliding block.
    # Values given at the reference resolution.
    #
    # EXTREMELY IMPORTANT TUNING GUIDE based on maximum marker size:
    # If you know the maximum width/height a marker can appear in the 1280x800 image 
    # (e.g., when the gripper is at its closest to the camera), you must ensure that 
    # `adaptive_thresh_win_size_max_ref` encompasses at least the thickness of the 
    # marker's black border, otherwise the center of the border will appear as "holes".
    # 
    # The DICT_6X6 ArUco has an 8x8 inner grid (6 data cells + 1 black border on each side).
    # Thus, the black border thickness is roughly (Marker_Width_in_Pixels / 8).
    #
    # Example: If your maximum marker appears 160 pixels wide in the 1280x800 image:
    #   - Border thickness = 160 / 8 = 20 pixels.
    #   - `adaptive_thresh_win_size_max_ref` should be at *minimum* > 20 (e.g., 23 to 33) 
    #     so the sliding window can "see" the white background adjacent to the border.
    # You may set this parameter to `'auto'` to have it mathematically 
    # calculated to be safely larger than the maximum black border thickness.
    # Alternatively, you can explicitly provide an integer (e.g., 23).
    adaptive_thresh_win_size_min_ref: int = 3
    adaptive_thresh_win_size_max_ref: Union[int, str] = 23 #'auto'
    adaptive_thresh_win_size_step_ref: int = 10

    # ---------------------------------------------------------
    # 6. Contour Filtering (False Positive Rejection)
    # ---------------------------------------------------------
    # These set the minimum and maximum perimeter of contours that the algorithm 
    # will even attempt to evaluate as a marker, represented as a ratio of 
    # max(image_width, image_height).
    #
    # Example tuning based on maximum marker size (e.g. 160px wide on 1280x800 image):
    #   - Max perimeter = 160 * 4 = 640px. 
    #   - Ratio = 640 / 1280 = 0.5. 
    #   - Thus `max_marker_perimeter_rate` could be safely lowered from 4.0 to 0.6 
    #     to skip computing giant false-positive shadows.
    # You may set `max_marker_perimeter_rate` to `'auto'` to use a safe algorithmic bound.
    # Alternatively, you can explicitly provide a float (e.g., 4.0).
    min_marker_perimeter_rate: float = 0.03 #0.03 is the OpenCV default value
    max_marker_perimeter_rate: Union[float, str] = 4.0 #'auto' #4.0 is the OpenCV default value

    def __post_init__(self):
        max_dim = max(MAX_EXPECTED_MARKER_WIDTH_REF, MAX_EXPECTED_MARKER_HEIGHT_REF)
        # DICT_6X6 adds a 1-bit solid black border around the 6x6 data grid -> 8x8 total.
        # Consequently, the thickness of the black border is approx max_dim / 8.
        border_thickness = max_dim / 8.0
        
        # 1. Validate / Auto-compile Adaptive Thresholding Window Max
        if self.adaptive_thresh_win_size_max_ref == 'auto':
            # Needs to be safely larger than the border thickness so the adaptive
            # block "sees" contrast on both edges of the black square.
            auto_val = int(round(border_thickness + 12)) 
            if auto_val % 2 == 0: auto_val += 1
            auto_val = max(auto_val, self.adaptive_thresh_win_size_min_ref + 2)
            self.adaptive_thresh_win_size_max_ref = auto_val
        elif isinstance(self.adaptive_thresh_win_size_max_ref, (int, float)):
            max_ref = int(self.adaptive_thresh_win_size_max_ref)
            # Check if it's wildly too small relative to the declared max size
            if max_ref < border_thickness:
                raise ValueError(
                    f"Config Error: adaptive_thresh_win_size_max_ref ({max_ref}) "
                    f"is too small! A max marker dimension of {max_dim}px has a black border thickness of "
                    f"~{border_thickness}px. Thus, the thresholding window must be strictly greater than "
                    f"the border thickness, else the marker borders will hollow out."
                )
                
        # 2. Validate / Auto-compile Max Perimeter Rate
        max_perimeter_expected = (MAX_EXPECTED_MARKER_WIDTH_REF + MAX_EXPECTED_MARKER_HEIGHT_REF) * 2
        ref_max_dim = max(REFERENCE_RESOLUTION_W, REFERENCE_RESOLUTION_H)
        expected_rate = max_perimeter_expected / ref_max_dim
        
        if self.max_marker_perimeter_rate == 'auto':
            # Give a 20% safety margin buffer on top of the largest expected perimeter
            self.max_marker_perimeter_rate = expected_rate * 1.2
        elif isinstance(self.max_marker_perimeter_rate, (int, float)):
            max_rate = float(self.max_marker_perimeter_rate)
            if max_rate < expected_rate:
                raise ValueError(
                    f"Config Error: max_marker_perimeter_rate ({max_rate:.2f}) is too small! "
                    f"A marker of {MAX_EXPECTED_MARKER_WIDTH_REF}x{MAX_EXPECTED_MARKER_HEIGHT_REF} has a "
                    f"perimeter rate of ~{expected_rate:.2f}. Set it higher so large markers aren't rejected."
                )

    def get_scaled_detector_parameters(self, actual_w: int, actual_h: int) -> cv2.aruco.DetectorParameters:
        """
        Creates an OpenCV DetectorParameters object.
        Pixel-based window sizes are linearly scaled relative to REFERENCE_RESOLUTION.
        """
        params = cv2.aruco.DetectorParameters()
        
        # Calculate scaling factor based on diagonal to handle aspect ratio changes gracefully
        ref_diag = math.hypot(REFERENCE_RESOLUTION_W, REFERENCE_RESOLUTION_H)
        act_diag = math.hypot(actual_w, actual_h)
        scale = act_diag / ref_diag if ref_diag > 0 else 1.0
        
        # Scale parameters (must be integers, adaptive thresholding windows generally require odd numbers)
        def scale_odd(val):
            scaled = int(round(val * scale))
            if scaled % 2 == 0:
                scaled += 1
            return max(3, scaled) # Minimum window size is 3
            
        params.adaptiveThreshWinSizeMin = scale_odd(self.adaptive_thresh_win_size_min_ref)
        params.adaptiveThreshWinSizeMax = scale_odd(self.adaptive_thresh_win_size_max_ref)
        params.adaptiveThreshWinSizeStep = max(1, int(round(self.adaptive_thresh_win_size_step_ref * scale)))
        
        # Apply contour filtering ratios (resolution independent out of the box)
        params.minMarkerPerimeterRate = self.min_marker_perimeter_rate
        params.maxMarkerPerimeterRate = self.max_marker_perimeter_rate
        
        params.cornerRefinementWinSize = max(2, int(round(self.corner_refinement_win_size_ref * scale)))
        
        if self.use_apriltag_refinement:
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        else:
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            
        if self.use_aruco3_detection:
            # useAruco3Detection requires OpenCV 4.x extra options
            params.useAruco3Detection = True
            
        return params
