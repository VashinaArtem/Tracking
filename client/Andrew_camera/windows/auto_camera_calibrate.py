#!/usr/bin/env python3

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from calibration import CalibrationManager
from feetech_servo import FeetechServo


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CameraCalibrationResult:
    image_size: Tuple[int, int]
    camera_matrix: List[List[float]]
    dist_coeffs: List[float]
    rms_error: float
    num_views: int
    model: str = "pinhole"


def find_chessboard(gray: np.ndarray, pattern_size: Tuple[int, int]) -> Optional[np.ndarray]:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners


def draw_overlay(
    frame: np.ndarray,
    pose_idx: int,
    total_poses: int,
    pan_n: float,
    tilt_n: float,
    collected_pose: int,
    need_pose: int,
    total_views: int,
    best_rms: Optional[float],
    status: str,
    pattern_size: Tuple[int, int],
    corners: Optional[np.ndarray] = None,
) -> np.ndarray:
    vis = frame.copy()
    if corners is not None:
        cv2.drawChessboardCorners(vis, pattern_size, corners, True)
    h, w = vis.shape[:2]
    bg_top_left = (5, 5)
    bg_bottom_right = (min(420, w - 5), 135)
    cv2.rectangle(vis, bg_top_left, bg_bottom_right, (0, 0, 0), thickness=-1)
    cv2.rectangle(vis, bg_top_left, bg_bottom_right, (60, 60, 60), thickness=1)
    y = 25
    line_h = 22
    def put(text, color=(255, 255, 255)):
        nonlocal y
        cv2.putText(vis, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += line_h
    put(f"Pose {pose_idx}/{total_poses}  pan={pan_n:+.2f} tilt={tilt_n:+.2f}")
    put(f"Pose views {collected_pose}/{need_pose}  Total {total_views}")
    if best_rms is not None and np.isfinite(best_rms):
        put(f"Best RMS {best_rms:.6f}", (0, 255, 255))
    put(status, (0, 200, 255))
    cv2.circle(vis, (w // 2, h // 2), 8, (255, 0, 0), 2)
    return vis


def calibrate_camera(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    image_size: Tuple[int, int],
    use_fisheye: bool,
) -> Optional[Tuple[float, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]]:
    if len(objpoints) < 5:
        return None
    if not use_fisheye:
        flags = (
            cv2.CALIB_RATIONAL_MODEL
            | cv2.CALIB_ZERO_TANGENT_DIST
            | cv2.CALIB_FIX_K3
        )
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None, flags=flags
        )
        return ret, mtx, dist, rvecs, tvecs
    # Fisheye calibration path
    objp_fe: List[np.ndarray] = [op.reshape(1, -1, 3) for op in objpoints]
    imgp_fe: List[np.ndarray] = []
    for ip in imgpoints:
        # ip may be (N,1,2) or (N,2). Ensure (1,N,2)
        if ip.ndim == 3 and ip.shape[1] == 1:
            ip2 = ip.reshape(1, -1, 2)
        else:
            ip2 = ip.reshape(1, -1, 2)
        imgp_fe.append(ip2)
    K = np.eye(3, dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    fe_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
    ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objp_fe, imgp_fe, image_size, K, D, None, None, fe_flags, criteria
    )
    return ret, K, D, rvecs, tvecs


def save_calibration_json(path: Path, result: CameraCalibrationResult) -> bool:
    try:
        payload = asdict(result)
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Saved camera calibration to {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save camera calibration: {e}")
        return False


def generate_positions_center_out(step: float, span: float) -> List[Tuple[float, float]]:
    step = max(0.02, min(1.0, step))
    span = max(0.1, min(1.0, span))
    positions: List[Tuple[float, float]] = [(0.0, 0.0)]
    r = step
    # Expand in rings until span
    while r <= span + 1e-6:
        ring: List[Tuple[float, float]] = [
            ( r,  0.0),
            (-r,  0.0),
            ( 0.0,  r),
            ( 0.0, -r),
            ( r,  r),
            (-r,  r),
            ( r, -r),
            (-r, -r),
        ]
        clamped = [(float(np.clip(px, -1.0, 1.0)), float(np.clip(py, -1.0, 1.0))) for px, py in ring]
        # Deduplicate in case clipping collapses points
        uniq = []
        seen = set()
        for p in clamped:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
        positions.extend(uniq)
        r += step
    return positions


def run_auto_calibration(args) -> int:
    calib_mgr = CalibrationManager(args.calibration_file)
    calib_mgr.load()

    pan_servo: Optional[FeetechServo] = None
    tilt_servo: Optional[FeetechServo] = None
    if not args.mock_servos:
        pan_cal = calib_mgr.get_calibration("pan")
        tilt_cal = calib_mgr.get_calibration("tilt")
        if pan_cal is None or tilt_cal is None:
            logger.error("Missing servo calibration in calibration.json for 'pan' and/or 'tilt'")
            return 2
        pan_servo = FeetechServo(
            args.pan_id, args.port,
            min_position=pan_cal.min_position,
            max_position=pan_cal.max_position,
            center_position=pan_cal.center_position,
        )
        tilt_servo = FeetechServo(
            args.tilt_id, args.port,
            min_position=tilt_cal.min_position,
            max_position=tilt_cal.max_position,
            center_position=tilt_cal.center_position,
            port_handler=pan_servo.port_handler,
        )
        if not pan_servo.connect():
            logger.error("Failed to connect pan servo")
            return 2
        if not tilt_servo.connect():
            logger.error("Failed to connect tilt servo")
            pan_servo.disconnect()
            return 2

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error(f"Failed to open camera {args.camera}")
        if pan_servo and tilt_servo:
            tilt_servo.disconnect(); pan_servo.disconnect()
        return 3

    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        time.sleep(0.2)

    ok, frame = cap.read()
    if not ok or frame is None:
        logger.error("Failed to read initial frame from camera")
        cap.release()
        if pan_servo and tilt_servo:
            tilt_servo.disconnect(); pan_servo.disconnect()
        return 3
    h, w = frame.shape[:2]
    image_size = (w, h)
    logger.info(f"Camera opened: {w}x{h}")

    pattern_size = (args.chess_cols, args.chess_rows)
    square = float(args.square_size)
    objp_template = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp_template[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp_template *= square

    # Determine expansion step
    if args.step > 0:
        step = args.step
    else:
        # Back-compat: derive step from grid and span
        step = (args.span * 2.0) / max(2, args.grid - 1)
    positions = generate_positions_center_out(step, args.span)
    total_poses = len(positions)
    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []

    best_rms = float("inf")
    best_result: Optional[CameraCalibrationResult] = None
    no_improve_steps = 0

    try:
        stop_expansion = False
        missed_poses = 0
        for idx, (pan_n, tilt_n) in enumerate(positions):
            if not args.mock_servos:
                assert pan_servo is not None and tilt_servo is not None
                pan_servo.set_position_normalized(pan_n)
                tilt_servo.set_position_normalized(tilt_n)
                time.sleep(args.dwell)

            collected = 0
            attempts = 0
            pose_start = time.perf_counter()
            while (
                collected < args.frames_per_pose
                and attempts < args.max_attempts_per_pose
                and (time.perf_counter() - pose_start) < args.search_timeout
            ):
                ok, frame = cap.read()
                attempts += 1
                if not ok or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners = find_chessboard(gray, pattern_size)
                if corners is None:
                    if args.preview:
                        remaining = max(0.0, args.search_timeout - (time.perf_counter() - pose_start))
                        vis = draw_overlay(
                            frame,
                            pose_idx=idx + 1,
                            total_poses=total_poses,
                            pan_n=pan_n,
                            tilt_n=tilt_n,
                            collected_pose=collected,
                            need_pose=args.frames_per_pose,
                            total_views=len(imgpoints),
                            best_rms=(best_rms if np.isfinite(best_rms) else None),
                            status=f"Searching chessboard... {remaining:.1f}s",
                            pattern_size=pattern_size,
                        )
                        cv2.imshow("Auto Camera Calibration", vis)
                        cv2.waitKey(1)
                    continue
                objpoints.append(objp_template.copy())
                imgpoints.append(corners)
                collected += 1

                if args.preview:
                    vis = draw_overlay(
                        frame,
                        pose_idx=idx + 1,
                        total_poses=total_poses,
                        pan_n=pan_n,
                        tilt_n=tilt_n,
                        collected_pose=collected,
                        need_pose=args.frames_per_pose,
                        total_views=len(imgpoints),
                        best_rms=(best_rms if np.isfinite(best_rms) else None),
                        status="Captured corners",
                        pattern_size=pattern_size,
                        corners=corners,
                    )
                    cv2.imshow("Auto Camera Calibration", vis)
                    cv2.waitKey(1)

            if collected == 0:
                missed_poses += 1
                logger.info(
                    f"Chessboard not visible at pose {idx+1} (pan={pan_n:.2f}, tilt={tilt_n:.2f}); missed {missed_poses}/{args.allow_missed_poses}"
                )
                if args.preview and frame is not None:
                    vis = draw_overlay(
                        frame,
                        pose_idx=idx + 1,
                        total_poses=total_poses,
                        pan_n=pan_n,
                        tilt_n=tilt_n,
                        collected_pose=0,
                        need_pose=args.frames_per_pose,
                        total_views=len(imgpoints),
                        best_rms=(best_rms if np.isfinite(best_rms) else None),
                        status=f"Not visible ({missed_poses}/{args.allow_missed_poses})",
                        pattern_size=pattern_size,
                    )
                    cv2.imshow("Auto Camera Calibration", vis)
                    cv2.waitKey(200)
                if missed_poses > args.allow_missed_poses:
                    logger.info("Stopping expansion: exceeded allowed consecutive missed poses")
                    stop_expansion = True
                    break
                else:
                    continue
            else:
                missed_poses = 0

            if len(imgpoints) >= max(5, args.min_views):
                calib = calibrate_camera(objpoints, imgpoints, image_size, args.fisheye)
                if calib is not None:
                    rms, K, D, _, _ = calib
                    improved = rms + 1e-9 < best_rms - args.min_delta
                    logger.info(f"Pose {idx+1}/{len(positions)} | total views {len(imgpoints)} | RMS {rms:.5f} | {'improved' if improved else 'no change'}")
                    if improved:
                        best_rms = rms
                        best_result = CameraCalibrationResult(
                            image_size=image_size,
                            camera_matrix=K.tolist(),
                            dist_coeffs=D.flatten().tolist(),
                            rms_error=float(rms),
                            num_views=len(imgpoints),
                            model=("fisheye" if args.fisheye else "pinhole"),
                        )
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                    if args.preview and frame is not None:
                        vis = draw_overlay(
                            frame,
                            pose_idx=idx + 1,
                            total_poses=total_poses,
                            pan_n=pan_n,
                            tilt_n=tilt_n,
                            collected_pose=collected,
                            need_pose=args.frames_per_pose,
                            total_views=len(imgpoints),
                            best_rms=float(best_rms) if np.isfinite(best_rms) else None,
                            status=f"RMS {rms:.6f} ({'improved' if improved else 'no change'})",
                            pattern_size=pattern_size,
                        )
                        cv2.imshow("Auto Camera Calibration", vis)
                        cv2.waitKey(1)

            if best_result and len(imgpoints) >= args.max_views:
                logger.info("Reached max views; stopping")
                break
            if best_result and no_improve_steps >= args.patience:
                logger.info("Early stop: no improvement over patience window")
                break
        # If we broke due to chessboard not visible, that's okay: we keep best_result as of last successful poses

        if not best_result:
            logger.warning("Calibration did not meet minimum view requirements; attempting best-effort save")
            fallback_result: Optional[CameraCalibrationResult] = None
            if len(objpoints) >= 5:
                calib = calibrate_camera(objpoints, imgpoints, image_size, args.fisheye)
                if calib is not None:
                    rms, K, D, _, _ = calib
                    fallback_result = CameraCalibrationResult(
                        image_size=image_size,
                        camera_matrix=K.tolist(),
                        dist_coeffs=D.flatten().tolist(),
                        rms_error=float(rms),
                        num_views=len(imgpoints),
                        model=("fisheye" if args.fisheye else "pinhole"),
                    )
            if fallback_result is None:
                fx_fy = float(max(image_size)) * 0.8
                cx = float(image_size[0]) / 2.0
                cy = float(image_size[1]) / 2.0
                K = [[fx_fy, 0.0, cx], [0.0, fx_fy, cy], [0.0, 0.0, 1.0]]
                D = ([0.0, 0.0, 0.0, 0.0] if args.fisheye else [0.0, 0.0, 0.0, 0.0, 0.0])
                fallback_result = CameraCalibrationResult(
                    image_size=image_size,
                    camera_matrix=K,
                    dist_coeffs=D,
                    rms_error=1e9,
                    num_views=len(imgpoints),
                    model=("fisheye" if args.fisheye else "pinhole"),
                )
            out_path = Path(args.output)
            save_calibration_json(out_path, fallback_result)
            logger.info(
                f"Saved fallback calibration with {fallback_result.num_views} views and RMS {fallback_result.rms_error:.6f}"
            )
            return 0

        out_path = Path(args.output)
        save_calibration_json(out_path, best_result)
        logger.info(f"Best RMS: {best_result.rms_error:.6f} with {best_result.num_views} views")
        return 0

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if pan_servo and tilt_servo:
            tilt_servo.disable_torque(); pan_servo.disable_torque()
            tilt_servo.disconnect(); pan_servo.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Auto camera calibration using robot pan/tilt sweep")
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0')
    parser.add_argument('--pan-id', type=int, default=1)
    parser.add_argument('--tilt-id', type=int, default=2)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--calibration-file', type=str, default='calibration.json')
    parser.add_argument('--output', type=str, default='camera_calibration.json')
    parser.add_argument('--width', type=int, default=0)
    parser.add_argument('--height', type=int, default=0)
    parser.add_argument('--mock-servos', action='store_true')

    parser.add_argument('--chess-cols', type=int, default=9, help='Inner corners along columns')
    parser.add_argument('--chess-rows', type=int, default=6, help='Inner corners along rows')
    parser.add_argument('--square-size', type=float, default=0.0245, help='Square size in meters')
    parser.add_argument('--fisheye', action='store_true', help='Use fisheye calibration model (cv2.fisheye)')

    parser.add_argument('--grid', type=int, default=5, help='Deprecated: used to derive step when --step <= 0')
    parser.add_argument('--span', type=float, default=0.7, help='Normalized sweep half-span from center (0-1)')
    parser.add_argument('--step', type=float, default=0.15, help='Normalized increment per expansion ring')
    parser.add_argument('--dwell', type=float, default=0.4, help='Seconds to settle at each pose')
    parser.add_argument('--frames-per-pose', type=int, default=2)
    parser.add_argument('--max-attempts-per-pose', type=int, default=20)
    parser.add_argument('--search-timeout', type=float, default=2.0, help='Seconds to search per pose before giving up')
    parser.add_argument('--allow-missed-poses', type=int, default=2, help='Consecutive invisible poses allowed before stopping')
    parser.add_argument('--min-views', type=int, default=12)
    parser.add_argument('--max-views', type=int, default=80)
    parser.add_argument('--patience', type=int, default=6, help='Early stop steps without improvement')
    parser.add_argument('--min-delta', type=float, default=1e-4, help='Min RMS improvement to count')
    parser.add_argument('--preview', action='store_true')

    args = parser.parse_args()
    exit_code = run_auto_calibration(args)
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()


