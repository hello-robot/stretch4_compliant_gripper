import numpy as np
from stretch4_gripper_modeling_and_control import aruco_detector as ad
from stretch4_gripper_modeling_and_control import aruco_to_fingertips as af
from stretch4_gripper_modeling_and_control import fingertip_filter as ff
from stretch4_gripper_modeling_and_control.aruco_config import ArucoConfig

class FingertipDetector:
    def __init__(self, marker_info=None, max_missing=0, smooth=False, validator=None):
        if marker_info is None:
            from stretch4_gripper_modeling_and_control import aruco_marker_info
            marker_info = aruco_marker_info.marker_info
            
        self.config = ArucoConfig()

        self.aruco_detector = ad.ArucoDetector(
            marker_info=marker_info, 
            config=self.config,
            show_debug_images=False,
            validator=validator
        )
        
        fingertip_part = 'cup_top'
        self.aruco_to_fingertips = af.ArucoToFingertips(
            default_height_above_mounting_surface=af.suctioncup_height[fingertip_part]
        )

        self.filters = {
            'left': ff.ExponentialMovingAverageSE3(max_missing=max_missing, smooth=smooth),
            'right': ff.ExponentialMovingAverageSE3(max_missing=max_missing, smooth=smooth)
        }

    def process_image(self, color_image, rgb_camera_info, pos_pct=None):
        self.aruco_detector.update(color_image, rgb_camera_info, pos_pct)
        markers = self.aruco_detector.get_detected_marker_dict()
        raw_fingertips = self.aruco_to_fingertips.get_fingertips(markers)

        fingertips = {}
        for side in ['left', 'right']:
            raw_f = raw_fingertips.get(side, None)
            filt_f = self.filters[side].update(raw_f)
            if filt_f is not None:
                fingertips[side] = filt_f
                if raw_f is not None and 'alt' in raw_f:
                    fingertips[side]['alt'] = raw_f['alt']
                
        return fingertips

    def draw_fingertip_frames(self, fingertips, color_image, rgb_camera_info, draw_both_ippe=False):
        task_relevant_image = np.copy(color_image)
        self.aruco_to_fingertips.draw_fingertip_frames(
            fingertips,
            task_relevant_image,
            rgb_camera_info,
            axis_length_in_m=0.02,
            draw_origins=True,
            write_coordinates=True
        )
        
        if draw_both_ippe:
            alt_fingertips = {s: f['alt'] for s, f in fingertips.items() if 'alt' in f}
            if alt_fingertips:
                self.aruco_to_fingertips.draw_fingertip_frames(
                    alt_fingertips,
                    task_relevant_image,
                    rgb_camera_info,
                    axis_length_in_m=0.02,
                    draw_origins=True,
                    write_coordinates=False,
                    desaturated=True
                )
                
        return task_relevant_image

def add_fingertip_detector_args(parser):
    """Add command line arguments related to the FingertipDetector."""
    parser.add_argument('--max_missing', type=int, default=0, help='Maximum number of consecutive frames to use a previous estimate if the marker is not detected.')
    parser.add_argument('--smooth', action='store_true', help='Smooth the visually-estimated 6-DoF fingertip frames over time using an exponential moving average.')

def process_fingertip_detector_args(args, marker_info=None, validator=None):
    """Initialize a FingertipDetector from parsed command line arguments."""
    max_missing = getattr(args, 'max_missing', 0)
    smooth = getattr(args, 'smooth', False)
    return FingertipDetector(marker_info=marker_info, max_missing=max_missing, smooth=smooth, validator=validator)
