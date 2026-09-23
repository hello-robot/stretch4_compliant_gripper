#!/usr/bin/env python3
"""
Unit tests for wrist camera IMU telemetry processing, coordinate transformation,
tri-modal synchronization, and tap/bump detection.
"""

import unittest
import numpy as np
import time
import zmq
from scipy.spatial.transform import Rotation as R

from stretch4_gripper_modeling_and_control import telemetry_utils as tu


class TestIMUCoordinatesAndTransforms(unittest.TestCase):
    def test_coordinate_frame_mapping(self):
        """
        Verify the transformation matrix R_CAM_TO_GRIPPER matches the specified reference frames:
        Camera Frame (RDF):
          +X_cam: Right
          +Y_cam: Down
          +Z_cam: Forward (along optical axis)
        Gripper Frame (FLU):
          +X_grip: Normal to front face of OAK-D-SR (forward direction of robot gripper)
          +Y_grip: Normal to left face of OAK-D-SR (left direction)
          +Z_grip: Normal to top face of OAK-D-SR (up direction)
        """
        R_mat = tu.R_CAM_TO_GRIPPER

        # +Z_cam (forward along camera optical axis) -> +X_grip (forward)
        v_cam_fwd = np.array([0.0, 0.0, 1.0])
        v_grip_fwd = R_mat @ v_cam_fwd
        np.testing.assert_allclose(v_grip_fwd, [1.0, 0.0, 0.0], atol=1e-6)

        # +X_cam (right) -> -Y_grip (since +Y_grip is left)
        v_cam_right = np.array([1.0, 0.0, 0.0])
        v_grip_right = R_mat @ v_cam_right
        np.testing.assert_allclose(v_grip_right, [0.0, -1.0, 0.0], atol=1e-6)

        # +Y_cam (down) -> -Z_grip (since +Z_grip is up)
        v_cam_down = np.array([0.0, 1.0, 0.0])
        v_grip_down = R_mat @ v_cam_down
        np.testing.assert_allclose(v_grip_down, [0.0, 0.0, -1.0], atol=1e-6)

        # Right-handedness check: det(R) == +1 and R.T @ R == I
        np.testing.assert_allclose(np.linalg.det(R_mat), 1.0, atol=1e-6)
        np.testing.assert_allclose(R_mat.T @ R_mat, np.eye(3), atol=1e-6)


class TestBumpDetector(unittest.TestCase):
    def setUp(self):
        self.detector = tu.BumpDetector(threshold=2.5, max_angle_deg=45.0, cooldown_seconds=0.3)

    def test_normal_tap_detection(self):
        """A tap pushing into the front surface produces negative acceleration along X in the gripper frame."""
        # Strong normal tap: ax = -4.0 m/s^2, minor noise on ay and az
        sample = {
            'timestamp': 10.0,
            'sequence_number': 100,
            'gripper_frame': {
                'linear_acceleration': {'x': -4.0, 'y': 0.1, 'z': -0.1}
            }
        }
        event = self.detector.update(sample)
        self.assertIsNotNone(event)
        self.assertEqual(event['sequence_number'], 100)
        self.assertAlmostEqual(event['peak_ax'], -4.0)
        self.assertTrue(self.detector.is_alert_active(10.1, alert_duration=0.5))
        self.assertFalse(self.detector.is_alert_active(10.7, alert_duration=0.5))

    def test_sub_threshold_rejection(self):
        """Accelerations below threshold should not trigger a bump."""
        sample = {
            'timestamp': 10.0,
            'sequence_number': 101,
            'gripper_frame': {
                'linear_acceleration': {'x': -1.2, 'y': 0.0, 'z': 0.0}
            }
        }
        event = self.detector.update(sample)
        self.assertIsNone(event)

    def test_oblique_and_lateral_rejection(self):
        """Accelerations that are lateral (e.g. shearing across gripper) should not trigger normal bump detector."""
        # Dominantly lateral acceleration along +Y (left)
        sample = {
            'timestamp': 10.0,
            'sequence_number': 102,
            'gripper_frame': {
                'linear_acceleration': {'x': -1.0, 'y': 5.0, 'z': 0.0}
            }
        }
        event = self.detector.update(sample)
        self.assertIsNone(event)

    def test_cooldown_debounce(self):
        """Secondary impacts during the cooldown period must be ignored."""
        first_tap = {
            'timestamp': 20.0,
            'sequence_number': 200,
            'gripper_frame': {
                'linear_acceleration': {'x': -5.0, 'y': 0.0, 'z': 0.0}
            }
        }
        event1 = self.detector.update(first_tap)
        self.assertIsNotNone(event1)

        # Ringing sample 50 ms later
        bounce = {
            'timestamp': 20.05,
            'sequence_number': 201,
            'gripper_frame': {
                'linear_acceleration': {'x': -4.5, 'y': 0.0, 'z': 0.0}
            }
        }
        event2 = self.detector.update(bounce)
        self.assertIsNone(event2)

        # New distinct tap after cooldown (350 ms later)
        second_tap = {
            'timestamp': 20.35,
            'sequence_number': 202,
            'gripper_frame': {
                'linear_acceleration': {'x': -3.8, 'y': 0.0, 'z': 0.0}
            }
        }
        event3 = self.detector.update(second_tap)
        self.assertIsNotNone(event3)


class TestTriModalCrossAlignment(unittest.TestCase):
    def test_cross_alignment_offsets(self):
        """
        Verify that cross_align_streams correctly pairs the closest samples across
        joint states, IMU measurements, and camera frame timestamp.
        """
        cam_ts = 50.000 # Camera capture time

        joint_history = [
            {'monotonic_timestamp': 49.980, 'state_number': 10},
            {'monotonic_timestamp': 49.992, 'state_number': 11},
            {'monotonic_timestamp': 50.003, 'state_number': 12}, # Closest to cam (diff = 3 ms)
            {'monotonic_timestamp': 50.015, 'state_number': 13},
        ]

        imu_history = [
            {'timestamp': 49.988, 'sequence_number': 50},
            {'timestamp': 49.998, 'sequence_number': 51}, # Closest to cam (diff = 2 ms)
            {'timestamp': 50.008, 'sequence_number': 52},
        ]

        closest_joint, closest_imu, sync_meta = tu.cross_align_streams(
            joint_history, imu_history, cam_ts, image_seq=42
        )

        self.assertEqual(closest_joint['state_number'], 12)
        self.assertEqual(closest_imu['sequence_number'], 51)
        self.assertEqual(sync_meta['image']['sequence_number'], 42)
        self.assertAlmostEqual(sync_meta['closest_joint_state']['offset_to_image_ms'], 3.0, places=3)
        self.assertAlmostEqual(sync_meta['closest_imu_measurement']['offset_to_image_ms'], -2.0, places=3)
        self.assertAlmostEqual(sync_meta['joint_to_imu_offset_ms'], 5.0, places=3)

        # Verify cross-referenced fields on individual samples
        # IMU sequence 51 (ts=49.998) is closest to joint state 12 (ts=50.003, diff=-0.005s)
        imu_sample = next(m for m in imu_history if m['sequence_number'] == 51)
        self.assertEqual(imu_sample['closest_joint_state_number'], 12)
        self.assertAlmostEqual(imu_sample['time_relative_to_joint'], -0.005, places=4)


class TestZeroMQRoundTrip(unittest.TestCase):
    def test_fast_telemetry_pubsub(self):
        """Verify serialization and ultra-low latency reception over local ZMQ socket."""
        context = zmq.Context()
        pub = context.socket(zmq.PUB)
        pub.setsockopt(zmq.SNDHWM, 1)
        pub.bind('tcp://127.0.0.1:4419') # test port

        sub = context.socket(zmq.SUB)
        sub.setsockopt(zmq.SUBSCRIBE, b'')
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.connect('tcp://127.0.0.1:4419')

        time.sleep(0.1) # allow socket subscription handshake

        test_msg = {
            'type': 'telemetry',
            'timestamp': time.monotonic(),
            'sequence_number': 999,
            'imu': {
                'linear_acceleration': {'x': -0.1, 'y': 0.2, 'z': 0.0},
            },
            'joint_state': {'gripper': {'pos_pct': 55.0, 'effort': -1.2}},
            'bump_detected': False
        }

        pub.send_pyobj(test_msg)

        # Poll with 100 ms timeout
        poller = zmq.Poller()
        poller.register(sub, zmq.POLLIN)
        events = dict(poller.poll(100))
        self.assertIn(sub, events)

        recv_msg = sub.recv_pyobj()
        self.assertEqual(recv_msg['sequence_number'], 999)
        self.assertEqual(recv_msg['joint_state']['gripper']['pos_pct'], 55.0)

        pub.close()
        sub.close()
        context.term()


if __name__ == '__main__':
    unittest.main()
