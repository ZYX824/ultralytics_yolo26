"""Project original image pixels onto the horizontal laser-target plane.

Output coordinates are east/north meters relative to the center laser target.
DJI conventions: pitch=0 horizontal, pitch=-90 nadir, yaw=0 north, yaw=90 east,
positive roll rotates the camera's right axis downward. Camera and laser axes
must coincide. Sizes are ground-plane estimates, not 3D surface measurements.
"""

import math

import cv2
import numpy as np

from ours.extraction.extract_gsd import number, positive


class GeometryUnavailable(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class GroundProjector:
    def __init__(self, geometry):
        self.geometry = geometry
        if geometry.get("orientation", 1) != 1:
            raise GeometryUnavailable("orientation_requires_normalization", "Original pixel orientation must be 1")
        distance = positive(geometry.get("center_range_m"))
        pitch = number(geometry.get("gimbal_pitch_deg"))
        roll = number(geometry.get("gimbal_roll_deg"))
        yaw = number(geometry.get("gimbal_yaw_deg"))
        if distance is None or str(geometry.get("range_status", "")).lower() != "normal":
            raise GeometryUnavailable("missing_valid_range", "A positive Normal laser range is required")
        if pitch is None or not -90 <= pitch < 0 or roll is None or yaw is None:
            raise GeometryUnavailable("missing_valid_attitude", "Downward pitch, roll and yaw are required")
        calibration = geometry.get("calibrated_intrinsics")
        zoom = number(geometry.get("digital_zoom_ratio"))
        if zoom is not None and zoom > 1.001 and calibration is None:
            raise GeometryUnavailable("digital_zoom_requires_calibration", "Supply intrinsics for the zoomed image")
        if calibration is not None:
            fx, fy = positive(calibration.get("fx_px")), positive(calibration.get("fy_px"))
            cx, cy = number(calibration.get("cx_px")), number(calibration.get("cy_px"))
        else:
            fx, fy = positive(geometry.get("focal_x_px")), positive(geometry.get("focal_y_px"))
            cx, cy = number(geometry.get("principal_x_px")), number(geometry.get("principal_y_px"))
        if fx is None or fy is None or cx is None or cy is None:
            raise GeometryUnavailable("missing_intrinsics", "Positive focal lengths and principal point are required")
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        theta, roll, yaw = np.radians([-pitch, roll, yaw])
        forward_horizontal = np.array([math.sin(yaw), math.cos(yaw), 0.0])
        right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        self.forward = math.cos(theta) * forward_horizontal - math.sin(theta) * up
        down = -math.sin(theta) * forward_horizontal - math.cos(theta) * up
        self.right = math.cos(roll) * right + math.sin(roll) * down
        self.down = -math.sin(roll) * right + math.cos(roll) * down
        self.height = distance * math.sin(theta)  # always recomputed, never takeoff altitude
        self.center_offset = distance * self.forward[:2]

    def project(self, pixels_xy):
        """Intersect viewing rays with ground; reject any horizon/backward ray."""
        pixels = np.asarray(pixels_xy, dtype=np.float64)
        normalized = (pixels - [self.cx, self.cy]) / [self.fx, self.fy]
        rays = normalized[..., :1] * self.right + normalized[..., 1:] * self.down + self.forward
        if np.any(rays[..., 2] >= -1e-8 * np.linalg.norm(rays, axis=-1)):
            raise GeometryUnavailable("ray_at_or_above_horizon", "Mask reaches a ray without a finite ground intersection")
        hits = rays[..., :2] * (-self.height / rays[..., 2])[..., None] - self.center_offset
        if not np.isfinite(hits).all():
            raise GeometryUnavailable("nonfinite_projection", "Ground projection is not finite")
        return hits


def unavailable(status, reason, mask):
    return {"status": status, "reason": reason, "area_m2": None,
            "extent_long_m": None, "extent_short_m": None, "extent_corners_en_m": [],
            "ground_contours": [], "touches_image_border": bool(
                mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
            )}


def measure_mask(mask, geometry, contours=None):
    """Measure projected pixel cells, including single pixels, holes and islands.

    Area is the sum of the projected four-corner polygon of each foreground
    pixel. This avoids pixel-center contour area bias and subtracts holes
    naturally. The enclosing rectangle uses all projected cell corners.
    Processing strips bounds temporary memory even for large masks.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not all(mask.shape):
        raise ValueError("mask must be a nonempty 2D array")
    if mask.shape != (geometry.get("height_px"), geometry.get("width_px")):
        raise ValueError("Mask grid does not match image geometry")
    result = unavailable("empty_mask", "No foreground pixels", mask)
    if not mask.any():
        result["area_m2"] = 0.0
        return result
    try:
        projector = GroundProjector(geometry)
        hulls, area = [], 0.0
        offsets = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
        # Pixel centers are integers; corners cover full cells on the same grid.
        for start in range(0, mask.shape[0], 64):
            ys, xs = np.nonzero(mask[start:start + 64])
            for offset in range(0, len(xs), 32768):
                pixels = np.column_stack((xs[offset:offset + 32768], ys[offset:offset + 32768] + start))
                quads = projector.project(pixels[:, None, :] + offsets)
                local = quads - quads[:, :1, :]
                following = np.roll(local, -1, axis=1)
                area += float(np.abs(np.sum(
                    local[..., 0] * following[..., 1] - local[..., 1] * following[..., 0], axis=1
                )).sum() / 2)
                hulls.append(cv2.convexHull(quads.reshape(-1, 2).astype(np.float32)).reshape(-1, 2))
        hull = cv2.convexHull(np.concatenate(hulls)).reshape(-1, 2)
        rectangle = cv2.minAreaRect(hull)
        long_side, short_side = sorted(rectangle[1], reverse=True)
        if contours is None:
            found, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = [{"points_xy": c.reshape(-1, 2).tolist(), "parent": int(hierarchy[0, i, 3])}
                        for i, c in enumerate(found)]
        ground = [{"points_en_m": projector.project(c["points_xy"]).tolist(), "parent": c["parent"]}
                  for c in contours]
        result.update(status="estimated_ground_plane", reason=None, area_m2=area,
                      extent_long_m=float(long_side), extent_short_m=float(short_side),
                      extent_corners_en_m=cv2.boxPoints(rectangle).astype(float).tolist(),
                      ground_contours=ground)
    except GeometryUnavailable as exc:
        result.update(status=exc.status, reason=str(exc))
    return result


def measurement_method(geometry):
    return {
        "model": "horizontal_center_target_plane",
        "coordinates": "east_north_meters_relative_to_center_laser_target",
        "origin_latitude_deg": geometry.get("center_target_latitude_deg"),
        "origin_longitude_deg": geometry.get("center_target_longitude_deg"),
        "height_formula": "center_range_m * sin(abs(gimbal_pitch_deg))",
        "area": "sum of foreground pixel cell quadrilaterals projected onto target plane",
        "extent": "minimum area enclosing rectangle of projected foreground cells; not crack length or width",
        "ground_contours": "projected pixel-center contour points; holes identified by hierarchy parent",
        "intrinsics_source": "user_calibrated" if geometry.get("calibrated_intrinsics") else geometry.get("intrinsics_source"),
        "assumptions": ["laser target at optical center", "aligned laser and camera axes",
                        "horizontal ground at target elevation", "undistorted pixels and matching camera intrinsics",
                        "objects represented by their ground-plane footprint"],
    }
