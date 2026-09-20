#!/usr/bin/env python3
"""
hand_over_demo_config.py

Configuration file for the Stretch 4 Object Handover Demo.
Contains all tuning parameters, thresholds, and operational settings
organized by Finite State Machine (FSM) state and system component.

===============================================================================
                       FINITE STATE MACHINE (FSM) DIAGRAM
===============================================================================

                +---------------------------------------+
                |              INITIALIZE               |
                |  - Connect ZMQ PUB & SUB sockets      |
                |  - Wait for valid robot telemetry     |
                +-------------------+-------------------+
                                    |
                                    | Telemetry received
                                    v
                +---------------------------------------+
                |             OPEN_GRIPPER              |
                |  - Command gripper to OPEN (pos_pct)  |
                |  - Wait until gripper reaches open    |
                |    position + hold wait duration      |
                |  - User places object between fingers |
                +-------------------+-------------------+
                                    |
                                    | Open position reached + wait elapsed
                                    v
                +---------------------------------------+
                |             GRASP_OBJECT              |
                |  - Gently close gripper               |
                |  - Detect grasp via compliance:       |
                |    Ap Diff (mm) < -20.00 mm           |
                +-------------------+-------------------+
                                    |
                                    | Contact & compliance threshold reached
                                    v
                +---------------------------------------+
                |              RETRACT_ARM              |
                |  - Retract telescoping arm fully      |
                |    until arm extension <= 0.005 m     |
                |    (or inner mechanical stop reached) |
                +-------------------+-------------------+
                                    |
                                    | Arm fully retracted
                                    v
                +---------------------------------------+
                |              EXTEND_ARM               |
                |  - Extend telescoping arm by +30 cm   |
                |    (+0.30 m) towards recipient        |
                +-------------------+-------------------+
                                    |
                                    | Extension target reached (+30 cm)
                                    v
                +---------------------------------------+
                |             MONITOR_LOAD              |
                |  - Hold arm stationary                |
                |  - Track fingertip loads via ArUco    |
                |  - Trigger handover release when:     |
                |    Both Left Z+ and Right Z+ > 3.0 mm |
                +-------------------+-------------------+
                                    |
                                    | Both Left Z+ > 3.0 and Right Z+ > 3.0
                                    v
                +---------------------------------------+
                |            RELEASE_OBJECT             |
                |  - Command gripper to OPEN (pos_pct)  |
                |  - Allow object to release to person  |
                +-------------------+-------------------+
                                    |
                                    | Gripper opened & duration elapsed
                                    v
                +---------------------------------------+
                |               COMPLETED               |
                |  - Hold stationary (zero velocity)    |
                |  - Handover complete!                 |
                +---------------------------------------+

===============================================================================
"""

# =============================================================================
# 1. SYSTEM, NETWORKING & MODEL CONFIGURATION
# =============================================================================

# Loop frequency (Hz) for the main control and perception loop.
# recv_and_execute_gripper_commands.py operates at 30 Hz.
CONTROL_HZ = 30.0

# Path to the fingertip kinematic model YAML calibration file.
# If set to None, the demo will look up the default fleet calibration model
# at ~/stretch_user/<robot_id>/calibration_gripper/latest_model_planar.yaml.
DEFAULT_MODEL_PATH = None

# Network ports for inter-process ZMQ communication.
# GRIPPER_CMD_PORT: Port on which commands are published to the robot server (recv_and_execute_gripper_commands.py).
# GRIPPER_TELEMETRY_PORT: Port from which images and joint states are subscribed (send_gripper_images_and_joint_states.py).
GRIPPER_CMD_PORT = 4407
GRIPPER_TELEMETRY_PORT = 4409

# Display scaling factor for OpenCV visualization (1.0 = native camera resolution, 1.5 = larger).
DISPLAY_SCALE = 1.0

# Whether to enable exponential moving average smoothing for visual fingertip frames.
SMOOTH_ESTIMATES = False

# Maximum consecutive frames to retain last known fingertip pose if visual marker is temporarily occluded.
MAX_MISSING_FRAMES = 5


# =============================================================================
# 2. STATE: INITIALIZE
# =============================================================================
# In this state, the demo binds the command publisher, connects the telemetry
# subscriber, and waits to receive the first valid robot state packets.

# Maximum time (seconds) to wait for initial telemetry before warning/aborting.
INITIALIZE_TIMEOUT_SECONDS = 15.0


# =============================================================================
# 3. STATE: OPEN_GRIPPER
# =============================================================================
# In this state, the robot drives the gripper to an open position and pauses for a
# configurable duration, giving the user the opportunity to prepare and place
# an object between the fingers before the grasping motion begins.

# Absolute target position percentage for opening the gripper.
# Using an absolute pos_pct target ensures the gripper opens wide regardless
# of whether it started from a negative (closed) or zero position.
# Stretch 4 Gripper pos_pct scale:
#   -100% : Fully closed / maximum squeeze
#      0% : Fingertips just touching (aperture ~0.6 cm)
#   +100% : Moderately open (aperture ~5.4 cm)
#   +150% : Generously open (aperture ~8.6 cm)  <-- Default
#   +200% : Wide open (aperture ~11.7 cm)
#   +300% : Maximum mechanical opening (aperture ~15.8 cm)
OPEN_GRIPPER_TARGET_POS_PCT = 150.0

# Speed (%/s) at which the gripper opens during OPEN_GRIPPER.
# Default: 60.0 %/s ensures fast and responsive opening.
OPEN_GRIPPER_SPEED = 60.0

# Acceleration (%/s^2) for opening.
OPEN_GRIPPER_ACCEL = 60.0

# Positional arrival tolerance (pos_pct) to consider the gripper opened.
# Once current_gripper_pct >= (OPEN_GRIPPER_TARGET_POS_PCT - OPEN_GRIPPER_TOLERANCE_PCT),
# the hold/countdown timer begins.
OPEN_GRIPPER_TOLERANCE_PCT = 10.0

# Duration (in seconds) to pause in OPEN_GRIPPER after the gripper has reached
# the open position, giving the user adequate time to position the object between fingers.
# Default: 3.0 seconds.
OPEN_GRIPPER_WAIT_SECONDS = 3.0

# Maximum timeout (in seconds) in OPEN_GRIPPER as a safety fallback.
# If the gripper doesn't reach the target within this time (e.g. telemetry delay),
# it proceeds to count down and transition anyway.
OPEN_GRIPPER_MAX_TIMEOUT_SECONDS = 8.0


# =============================================================================
# 4. STATE: GRASP_OBJECT
# =============================================================================
# In this state, the robot closes the gripper until a compliant grasp is
# detected via Aperture Difference (Ap Diff).
#
# Aperture Difference is defined as:
#     Ap Diff (mm) = ap_pred - ap_vis
# where:
#     ap_pred = Kinematically predicted aperture based on gripper motor pos_pct
#     ap_vis  = Visually observed aperture between fingertip ArUco markers
# When the gripper fingers contact the object, the visual fingertips stop moving
# inward while the motor continues to command inward, causing Ap Diff to become
# negative. When Ap Diff drops below this threshold, a secure grasp is achieved.

# Aperture Difference threshold (mm) to detect object contact and squeeze.
# Grasped condition is satisfied when: Ap Diff < GRASP_APERTURE_DIFF_THRESHOLD.
# Typical value: -20.00 mm.
GRASP_APERTURE_DIFF_THRESHOLD = -20.00

# Gripper closing speed in percentage per second (%/s).
# Lower values (e.g., 15.0 - 20.0) give the user plenty of time to place and
# align the object, and allow the vision system to accurately capture contact.
GRASP_CLOSE_SPEED = 20.0

# Gripper closing acceleration (%/s^2).
GRASP_CLOSE_ACCEL = 20.0

# Target position percentage for closing.
# -100.0 corresponds to fully closed, ensuring the gripper continues closing
# until the Ap Diff threshold triggers contact detection.
GRASP_TARGET_POS_PCT = -100.0

# Number of consecutive frames that Ap Diff must be < threshold to trigger transition.
# Debouncing prevents single-frame visual tracking jitter from falsely triggering grasp completion.
GRASP_CONSECUTIVE_FRAMES = 1

# Optional pause duration (seconds) before the gripper begins closing,
# giving the user initial time to position the object between open fingers.
GRASP_PRE_WAIT_SECONDS = 0.0

# Displacement percentage (pos_pct_disp) commanded upon confirming a secure grasp.
# When Ap Diff drops below threshold, the FSM transitions to RETRACT_ARM and commands
# the gripper to halt its inward motion and lock its position.
#
# Crucial Technical Detail:
# In previous versions, commanding an absolute pos_pct target (e.g. from telemetry)
# caused the gripper to open backwards slightly right after grasping. Because telemetry
# and status polling have latency (~33ms at 30Hz corresponds to ~0.7% motion at 20%/s),
# any absolute position from recent telemetry was MORE OPEN than the physical motor.
# Furthermore, setting target equal to current position reduced servo PID error to zero,
# causing the compliant fingers to spring back and relax contact force.
#
# Using pos_pct_disp tells the robot's onboard controller to apply a displacement relative
# to its LIVE hardware position:
# - A small negative value (e.g., -1.0% to -2.0%) guarantees the commanded hold target is
#   strictly inward (closing/squeeze direction), completely eliminating any backwards opening,
#   and maintaining solid preloaded tension on the compliant fingers.
# - A value of 0.0% commands the gripper to hold its exact live position.
# - Positive values will open the gripper and must NOT be used.
GRASP_HOLD_DISPLACEMENT_PCT = -1.0

# Speed (%/s) and acceleration (%/s^2) for locking the gripper upon grasp confirmation.
GRASP_HOLD_SPEED = 30.0
GRASP_HOLD_ACCEL = 30.0


# =============================================================================
# 5. STATE: RETRACT_ARM
# =============================================================================
# In this state, the robot fully retracts the telescoping arm into the robot body
# while holding the grasped object.

# Normalized velocity command for arm retraction [-1.0 to 1.0].
# Positive value here represents retraction speed; commanded as -ARM_RETRACT_SPEED.
ARM_RETRACT_SPEED = 0.5

# Arm extension position threshold (in meters) to consider the arm fully retracted.
# On Stretch 4, fully retracted arm position is ~0.00 m (0.0 cm).
ARM_RETRACT_TOLERANCE_M = 0.005

# Number of consecutive loop cycles with near-zero arm motion before concluding
# that the arm has hit the physical inner mechanical stop.
ARM_STALL_CONSECUTIVE_FRAMES = 15

# Threshold for arm position change (meters) below which the arm is considered stalled.
ARM_STALL_MOTION_THRESHOLD_M = 0.0003


# =============================================================================
# 6. STATE: EXTEND_ARM
# =============================================================================
# In this state, the robot extends the telescoping arm forward towards the
# person receiving the object.

# Distance to extend the arm (in meters) relative to the fully retracted position.
# Default: 0.30 m (30 cm).
ARM_EXTENSION_DISTANCE_M = 0.30

# Maximum normalized forward extension velocity command [0.0 to 1.0].
ARM_EXTEND_SPEED = 0.5

# Minimum normalized extension velocity command [0.0 to 1.0] when decelerating near target.
ARM_EXTEND_MIN_SPEED = 0.15

# Proportional gain (kp) for decelerating the arm smoothly as it nears the target extension.
ARM_EXTEND_KP = 3.0

# Positional arrival tolerance (meters) for arm extension.
# When (target - current_pos) <= ARM_EXTEND_TOLERANCE_M, the arm has arrived.
ARM_EXTEND_TOLERANCE_M = 0.003


# =============================================================================
# 7. STATE: MONITOR_LOAD
# =============================================================================
# In this state, the arm is held stationary while monitoring the external upward load
# applied to the gripper by the person pulling on the object.
# Loads are measured along the finger plane normal vector (z_axis, pointing upward).
# When someone lifts upward on the object, the compliant fingertips deflect in the +Z direction.

# Upward load deflection threshold (mm) for the Left fingertip.
LOAD_LEFT_Z_THRESHOLD = 1.0

# Upward load deflection threshold (mm) for the Right fingertip.
LOAD_RIGHT_Z_THRESHOLD = 1.0

# Unified default load threshold (mm). Both left and right fingertip Z+ values
# must exceed this threshold simultaneously to trigger the release.
LOAD_THRESHOLD_Z = 1.0

# Number of consecutive frames that both Left Z+ and Right Z+ must exceed threshold
# before transitioning to RELEASE_OBJECT. Prevents accidental release from momentary vibration.
LOAD_CONSECUTIVE_FRAMES = 1


# =============================================================================
# 8. STATE: RELEASE_OBJECT
# =============================================================================
# In this state, the robot opens the gripper to release the object to the person.

# Absolute target position percentage for releasing the object.
# Using pos_pct = 130.0% (aperture ~6.7 cm) ensures the gripper opens wide enough
# to cleanly release the object without snags or dragging.
RELEASE_TARGET_POS_PCT = 130.0

# Speed (%/s) for releasing.
RELEASE_SPEED = 60.0

# Acceleration (%/s^2) for releasing.
RELEASE_ACCEL = 60.0

# Duration (in seconds) to allow the gripper mechanism to physically open
# before transitioning to COMPLETED.
RELEASE_DURATION_S = 1.8


# =============================================================================
# 9. STATE: COMPLETED
# =============================================================================
# In this state, the handover is finished. The arm and gripper remain stationary.

# If True, the program automatically exits cleanly once handover completes.
# If False, the program stays running with zero velocity so the user can inspect the HUD.
COMPLETED_AUTO_EXIT = False

# Delay (seconds) before auto-exiting if COMPLETED_AUTO_EXIT is True.
COMPLETED_AUTO_EXIT_DELAY_S = 2.0
