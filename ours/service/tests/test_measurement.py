"""Independent physical checks for the ground-plane projection and pixel area."""

import math
import unittest

import numpy as np

from ours.service.measurement import GroundProjector, measure_mask


def geometry(**changes):
    return {"width_px": 40, "height_px": 30, "center_range_m": 100, "range_status": "Normal",
            "gimbal_pitch_deg": -90, "gimbal_roll_deg": 0, "gimbal_yaw_deg": 0,
            "focal_x_px": 1000, "focal_y_px": 1000, "principal_x_px": 19.5, "principal_y_px": 14.5,
            **changes}


class MeasurementTests(unittest.TestCase):
    def test_nadir_rectangle_exact_area_extent_and_isolated_pixel(self):
        mask = np.zeros((30, 40), bool)
        mask[10:20, 10:30] = True
        result = measure_mask(mask, geometry())
        self.assertEqual(result["status"], "estimated_ground_plane")
        self.assertAlmostEqual(result["area_m2"], 2.0)
        self.assertAlmostEqual(result["extent_long_m"], 2.0, places=5)
        self.assertAlmostEqual(result["extent_short_m"], 1.0, places=5)
        mask[:] = False
        mask[15, 20] = True
        result = measure_mask(mask, geometry())
        self.assertAlmostEqual(result["area_m2"], 0.01)
        self.assertAlmostEqual(result["extent_short_m"], 0.1, places=5)

    def test_holes_and_islands_area_is_not_contour_center_area(self):
        mask = np.zeros((30, 40), bool)
        mask[5:25, 5:35] = True
        mask[10:20, 10:30] = False
        mask[1, 1] = True
        result = measure_mask(mask, geometry())
        self.assertAlmostEqual(result["area_m2"], (600 - 200 + 1) * 0.01)
        self.assertTrue(any(c["parent"] >= 0 for c in result["ground_contours"]))

    def test_oblique_projection_matches_direct_pinhole_equations(self):
        g = geometry(gimbal_pitch_deg=-45)
        projector = GroundProjector(g)
        np.testing.assert_allclose(projector.project([[19.5, 14.5]]), [[0, 0]], atol=1e-12)
        # Independent yaw=0, roll=0 equations: E=H*x/(sinθ+y*cosθ),
        # N=H*(cosθ-y*sinθ)/(sinθ+y*cosθ)-R*cosθ.
        x, y = 0.01, 0.005
        sin = cos = math.sqrt(0.5)
        height = 100 * sin
        expected = [height * x / (sin + y * cos),
                    height * (cos - y * sin) / (sin + y * cos) - 100 * cos]
        np.testing.assert_allclose(projector.project([[29.5, 19.5]])[0], expected, atol=1e-10)
        mask = np.zeros((30, 40), bool)
        mask[15, 20] = True
        result = measure_mask(mask, g)
        # Center-local ground scales at 45 degrees: 0.1 and sqrt(2)*0.1 m/px.
        self.assertAlmostEqual(result["area_m2"], math.sqrt(2) * 0.01, delta=0.00005)

    def test_yaw_roll_rotate_extent_but_preserve_nadir_area(self):
        mask = np.zeros((30, 40), bool)
        mask[10:20, 10:30] = True
        for yaw, roll in ((90, 0), (76.7, 25), (180, -30)):
            result = measure_mask(mask, geometry(gimbal_yaw_deg=yaw, gimbal_roll_deg=roll))
            self.assertAlmostEqual(result["area_m2"], 2)
            self.assertAlmostEqual(result["extent_long_m"], 2, places=5)
        east = GroundProjector(geometry(gimbal_pitch_deg=-45, gimbal_yaw_deg=90))
        self.assertLess(east.project([[29.5, 14.5]])[0, 1], 0)  # image right points south

    def test_calibrated_intrinsics_override_approximation(self):
        g = geometry(calibrated_intrinsics={"fx_px": 2000, "fy_px": 2000, "cx_px": 19.5, "cy_px": 14.5})
        mask = np.ones((30, 40), bool)
        result = measure_mask(mask, g)
        self.assertAlmostEqual(result["area_m2"], 1200 * 0.05**2)
        self.assertTrue(result["touches_image_border"])

    def test_unavailable_horizon_and_empty_masks_never_produce_sizes(self):
        mask = np.ones((30, 40), bool)
        for changes, status in (
            ({"center_range_m": None}, "missing_valid_range"),
            ({"gimbal_roll_deg": None}, "missing_valid_attitude"),
            ({"orientation": 6}, "orientation_requires_normalization"),
            ({"gimbal_pitch_deg": -1, "focal_y_px": 10}, "ray_at_or_above_horizon"),
        ):
            result = measure_mask(mask, geometry(**changes))
            self.assertEqual(result["status"], status)
            self.assertIsNone(result["area_m2"])
        empty = measure_mask(np.zeros((30, 40), bool), geometry())
        self.assertEqual(empty["status"], "empty_mask")
        self.assertEqual(empty["area_m2"], 0)

    def test_area_changes_with_pixel_position_and_union_has_no_double_count(self):
        g = geometry(gimbal_pitch_deg=-45, focal_y_px=100)
        far = np.zeros((30, 40), bool)
        near = far.copy()
        far[1:5, 10:14] = True
        near[25:29, 10:14] = True
        self.assertGreater(measure_mask(far, g)["area_m2"], measure_mask(near, g)["area_m2"])
        self.assertAlmostEqual(measure_mask(far | far, g)["area_m2"], measure_mask(far, g)["area_m2"])
