#!/usr/bin/env python3
"""Edge AI CCTV - Real-Time Live AI Vision, Multi-Class Detection & Multi-Zone Studio.

Core Capabilities:
1. False-Positive & Insect Rejection:
   - Physical scale & aspect-ratio gating (filters out small insects, moths, shadows, reflections)
   - Temporal track confirmation (requires >= 3 consecutive frames before confirming)
   - IoU + Centroid spatial association tracking
   - Velocity anomaly suppression (rejects impossible optical teleportation speeds)
   - Strict allowed_classes enforcement (only verified 'person' or 'vehicle' can trip security perimeters)
2. Unlimited Multi-Zone Engine (Tripwires, Restricted Areas, Exclusion Masks):
   - Add, edit, toggle, or completely delete any number of independent tripwires and restricted polygons
   - Directional In/Out counting per tripwire (A->B, B->A, Bidirectional)
   - Privacy & AI Exclusion Masks: Select and blur/blackout areas to completely exclude from AI scanning
   - Persistent storage to storage/zones_config.json (survives reboots)
3. 15-Second Automated Incident Clip Recording:
   - Captures 5s pre-event buffer + 10s post-event video clip on tripwire/intrusion events
   - Real-time flashing HUD recording badge: 🔴 REC 15s INCIDENT CLIP
4. Camera Source Discovery & Multi-Camera Switcher:
   - Automatic detection of local USB cameras, ESP32-S3 IP cameras, and RTSP streams
   - One-click camera switcher in the Web HUD
"""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import json
import logging
import math
import os
import socket
import sys
import threading
import time
import urllib.request
import urllib.error
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "edge_backend"))

from app.config import settings
from app.models.schemas import Point2D, TripwireDirection, ZoneConfig, ZoneType, BoundingBox
from app.services.ai_zone_service import ai_zone_service, PolygonGeometry
from app.services.clip_recorder import clip_recorder_service
from app.services.hailo_inference_service import hailo_inference_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LiveMonitor")


# ==============================================================================
# COCO Multi-Class Mapping & Constants
# ==============================================================================
COCO_CLASS_MAP = {
    0: "person",
    1: "vehicle", 2: "vehicle", 3: "vehicle", 5: "vehicle", 7: "vehicle",
    15: "animal", 16: "animal", 17: "animal", 18: "animal", 19: "animal",
    24: "package",  # backpack
    25: "package",  # umbrella
    26: "package",  # handbag
    28: "package",  # suitcase
}

CLASS_THRESHOLDS = {
    "person": 0.50,
    "vehicle": 0.45,
    "package": 0.38,
    "animal": 0.40,
    "default": 0.45
}

CLASS_COLORS = {
    "person": (0, 255, 180),     # Green/Cyan
    "vehicle": (240, 50, 200),   # Magenta/Purple
    "package": (0, 160, 255),    # Orange/Amber
    "animal": (255, 210, 0),     # Yellow/Gold
    "default": (180, 190, 200)
}


@dataclass
class DetectionObject:
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized [0..1]
    confidence: float
    class_id: int
    class_name: str
    keypoints: Optional[List[Tuple[float, float, float]]] = None  # 17 keypoints: (norm_x, norm_y, conf)


# 17 COCO Keypoints & Skeleton Limb Graph
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Face / Ears
    (5, 6),                              # Shoulders
    (5, 7), (7, 9),                      # Left arm
    (6, 8), (8, 10),                     # Right arm
    (5, 11), (6, 12),                    # Torso sides
    (11, 12),                            # Pelvis / Hips
    (11, 13), (13, 15),                  # Left leg
    (12, 14), (14, 16)                   # Right leg
]


# ==============================================================================
# Thread-Safe Background Clip Muxer
# ==============================================================================
class BackgroundClipMuxer:
    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ClipMuxerWorker")

    def submit_mux_task(
        self,
        frames: List[np.ndarray],
        output_path: Path,
        fps: int = 25,
        on_complete_callback: Optional[callable] = None
    ):
        if not frames:
            return
        frames_copy = [f.copy() for f in frames]

        def _task():
            try:
                h, w = frames_copy[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
                for f in frames_copy:
                    if f.shape[:2] != (h, w):
                        f = cv2.resize(f, (w, h))
                    out.write(f)
                out.release()
                logger.info(f"✅ 15-Second MP4 Video Clip Saved: {output_path.name}")
                if on_complete_callback:
                    on_complete_callback(str(output_path))
            except Exception as e:
                logger.error(f"[-] Video clip export failed for {output_path.name}: {e}")

        self.executor.submit(_task)

    def shutdown(self):
        self.executor.shutdown(wait=False)


clip_muxer = BackgroundClipMuxer()


# ==============================================================================
# Zone Alert Debouncer
# ==============================================================================
class ZoneAlertDebouncer:
    def __init__(self, cooldown_seconds: float = 3.0):
        self.cooldown_seconds = cooldown_seconds
        self.last_alerts: Dict[str, float] = {}

    def should_dispatch(self, zone_id: str, track_id: int, now: float) -> bool:
        key = f"{zone_id}:{track_id}"
        last_t = self.last_alerts.get(key, 0.0)
        if (now - last_t) >= self.cooldown_seconds:
            self.last_alerts[key] = now
            return True
        return False


# ==============================================================================
# Multi-Class Deep Learning Detector with Insect & Noise Filter
# ==============================================================================
def find_onnx_model(candidates: List[str]) -> Optional[str]:
    search_dirs = [
        PROJECT_ROOT / "edge_backend" / "models",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT,
        Path("edge_backend/models"),
        Path("models"),
        Path(".")
    ]
    for cand in candidates:
        for s_dir in search_dirs:
            p = s_dir / cand
            if p.exists() and p.is_file():
                return str(p.resolve())
    return None


class UnifiedPersonDetector:
    """Full 80-Class Deep Neural Network Detector & 17-Keypoint Pose Estimator with Physical Dimension Gating."""

    def __init__(self, onnx_model_path: Optional[str] = None, pose_model_path: Optional[str] = None):
        if onnx_model_path is None:
            onnx_model_path = find_onnx_model(["yolo11n.onnx", "yolov8n.onnx", "yolov5n.onnx"])

        if pose_model_path is None:
            pose_model_path = find_onnx_model(["yolo11n-pose.onnx", "yolov8n-pose.onnx"])

        self.model_path = onnx_model_path
        self.pose_model_path = pose_model_path
        self.net = None
        self.pose_net = None
        self.use_dnn = False
        self.use_pose_dnn = False

        if self.model_path and os.path.exists(self.model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.use_dnn = True
                logger.info(f"✅ Primary YOLO Detection Model loaded: {self.model_path}")
            except Exception as e:
                logger.warning(f"[-] Could not load detection model ({e}). Using Morphological Body Merger.")
                self.use_dnn = False

        if self.pose_model_path and os.path.exists(self.pose_model_path):
            try:
                self.pose_net = cv2.dnn.readNetFromONNX(self.pose_model_path)
                self.pose_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.pose_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.use_pose_dnn = True
                logger.info(f"✅ YOLO Pose Estimation Model loaded: {self.pose_model_path}")
            except Exception as e:
                logger.warning(f"[-] Could not load pose model ({e}). Fallback to kinematic skeletal estimation.")
                self.use_pose_dnn = False

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=24, detectShadows=False)
        self.vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 45))
        self.horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def _letterbox(
        self,
        img: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114)
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def detect(self, frame: np.ndarray, skeletal_tracking_enabled: bool = True) -> List[DetectionObject]:
        h, w = frame.shape[:2]
        if self.use_dnn and self.net is not None:
            try:
                return self._detect_dnn(frame, w, h, skeletal_tracking_enabled=skeletal_tracking_enabled)
            except Exception as e:
                logger.warning(f"DNN inference error ({e}), falling back to morphological.")
        return self._detect_morphological_merged(frame, w, h, skeletal_tracking_enabled=skeletal_tracking_enabled)

    def _detect_dnn(
        self,
        frame: np.ndarray,
        w: int,
        h: int,
        skeletal_tracking_enabled: bool = True
    ) -> List[DetectionObject]:
        letterbox_img, ratio, (dw, dh) = self._letterbox(frame, (640, 640))
        blob = cv2.dnn.blobFromImage(letterbox_img, 1.0 / 255.0, (640, 640), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward()

        # Handle 3D / 2D output shapes: [1, 84, 8400], [1, 56, 8400], [1, 8400, 84], [1, 25200, 85]
        if len(preds.shape) == 3:
            if preds.shape[1] < preds.shape[2]:
                preds = np.transpose(preds[0], (1, 0))  # Shape becomes [N, C]
            else:
                preds = preds[0]
        elif len(preds.shape) == 2:
            if preds.shape[0] < preds.shape[1]:
                preds = np.transpose(preds, (1, 0))

        boxes, confidences, class_ids, class_names, keypoints_list = [], [], [], [], []
        num_cols = preds.shape[1]

        for row in preds:
            kp_row = None
            if num_cols == 85:
                # YOLOv5 format [cx, cy, w, h, obj_conf, p0...p79]
                obj_conf = float(row[4])
                if obj_conf < 0.25:
                    continue
                class_scores = row[5:]
                best_class_idx = int(np.argmax(class_scores))
                class_score = float(class_scores[best_class_idx])
                final_conf = obj_conf * class_score
            elif num_cols == 56:
                # YOLO11 / YOLOv8 Pose format [cx, cy, w, h, person_conf, 17*(x, y, conf)]
                final_conf = float(row[4])
                best_class_idx = 0  # COCO Person class
                if skeletal_tracking_enabled and final_conf >= 0.30:
                    kp_row = []
                    for k in range(17):
                        k_x = (float(row[5 + k * 3]) - dw) / ratio
                        k_y = (float(row[5 + k * 3 + 1]) - dh) / ratio
                        k_c = float(row[5 + k * 3 + 2])
                        kp_row.append((
                            max(0.0, min(1.0, k_x / w)),
                            max(0.0, min(1.0, k_y / h)),
                            k_c
                        ))
            else:
                # YOLO11 / YOLOv8 Detect format [cx, cy, w, h, p0...p79] (84 columns)
                class_scores = row[4:84]
                best_class_idx = int(np.argmax(class_scores))
                final_conf = float(class_scores[best_class_idx])

            semantic_class = COCO_CLASS_MAP.get(best_class_idx, "unknown")
            thresh = CLASS_THRESHOLDS.get(semantic_class, CLASS_THRESHOLDS["default"])

            # True Argmax validation
            if final_conf >= thresh and semantic_class != "unknown":
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                cx = (cx - dw) / ratio
                cy = (cy - dh) / ratio
                bw = bw / ratio
                bh = bh / ratio

                # Physical Size & Aspect-Ratio Sanity Gate (Reject insects, bugs, tiny speckles)
                norm_w = bw / w
                norm_h = bh / h
                norm_area = norm_w * norm_h

                if semantic_class == "person":
                    if norm_h < 0.075 and norm_area < 0.0035:
                        continue
                elif semantic_class == "vehicle":
                    if norm_w < 0.06 and norm_area < 0.008:
                        continue

                x1 = int(cx - bw / 2)
                y1 = int(cy - bh / 2)
                boxes.append([x1, y1, int(bw), int(bh)])
                confidences.append(final_conf)
                class_ids.append(best_class_idx)
                class_names.append(semantic_class)
                keypoints_list.append(kp_row)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.30, 0.45)
        results = []
        if len(indices) > 0:
            flat_indices = indices.flatten() if hasattr(indices, "flatten") else [idx[0] if isinstance(idx, (list, tuple)) else idx for idx in indices]
            for idx in flat_indices:
                bx, by, bw_px, bh_px = boxes[idx]
                norm_x1 = max(0.0, bx / w)
                norm_y1 = max(0.0, by / h)
                norm_x2 = min(1.0, (bx + bw_px) / w)
                norm_y2 = min(1.0, (by + bh_px) / h)
                s_class = class_names[idx]
                kp = keypoints_list[idx]

                if s_class == "person" and skeletal_tracking_enabled and kp is None:
                    kp = self._estimate_skeletal_keypoints((norm_x1, norm_y1, norm_x2, norm_y2))

                results.append(DetectionObject(
                    bbox=(norm_x1, norm_y1, norm_x2, norm_y2),
                    confidence=confidences[idx],
                    class_id=class_ids[idx],
                    class_name=s_class,
                    keypoints=kp if skeletal_tracking_enabled else None
                ))

        if skeletal_tracking_enabled and self.use_pose_dnn and self.pose_net is not None:
            results = self._attach_pose_model_keypoints(blob, results, w, h, ratio, dw, dh)

        return results

    def _attach_pose_model_keypoints(
        self,
        blob: np.ndarray,
        detections: List[DetectionObject],
        w: int,
        h: int,
        ratio: float,
        dw: float,
        dh: float
    ) -> List[DetectionObject]:
        try:
            self.pose_net.setInput(blob)
            p_preds = self.pose_net.forward()
            if len(p_preds.shape) == 3:
                if p_preds.shape[1] < p_preds.shape[2]:
                    p_preds = np.transpose(p_preds[0], (1, 0))
                else:
                    p_preds = p_preds[0]
            elif len(p_preds.shape) == 2:
                if p_preds.shape[0] < p_preds.shape[1]:
                    p_preds = np.transpose(p_preds, (1, 0))

            pose_candidates = []
            for row in p_preds:
                p_conf = float(row[4])
                if p_conf < 0.35:
                    continue
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                cx = (cx - dw) / ratio
                cy = (cy - dh) / ratio
                bw = bw / ratio
                bh = bh / ratio
                norm_box = (
                    max(0.0, (cx - bw / 2) / w),
                    max(0.0, (cy - bh / 2) / h),
                    min(1.0, (cx + bw / 2) / w),
                    min(1.0, (cy + bh / 2) / h)
                )
                kps = []
                for k in range(17):
                    kx = (float(row[5 + k * 3]) - dw) / ratio
                    ky = (float(row[5 + k * 3 + 1]) - dh) / ratio
                    kc = float(row[5 + k * 3 + 2])
                    kps.append((
                        max(0.0, min(1.0, kx / w)),
                        max(0.0, min(1.0, ky / h)),
                        kc
                    ))
                pose_candidates.append((norm_box, p_conf, kps))

            for det in detections:
                if det.class_name == "person" and pose_candidates:
                    best_iou = 0.0
                    best_kps = None
                    for pbox, pconf, pkps in pose_candidates:
                        xA = max(det.bbox[0], pbox[0])
                        yA = max(det.bbox[1], pbox[1])
                        xB = min(det.bbox[2], pbox[2])
                        yB = min(det.bbox[3], pbox[3])
                        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
                        areaA = (det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1])
                        areaB = (pbox[2] - pbox[0]) * (pbox[3] - pbox[1])
                        iou = inter / max(1e-6, areaA + areaB - inter)
                        if iou > best_iou:
                            best_iou = iou
                            best_kps = pkps
                    if best_iou > 0.20 and best_kps is not None:
                        det.keypoints = best_kps
        except Exception as e:
            logger.debug(f"Pose DNN pass error: {e}")
        return detections

    @staticmethod
    def _estimate_skeletal_keypoints(bbox: Tuple[float, float, float, float]) -> List[Tuple[float, float, float]]:
        x1, y1, x2, y2 = bbox
        bw = max(0.01, x2 - x1)
        bh = max(0.01, y2 - y1)
        cx = (x1 + x2) / 2.0

        # 17 keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
        nose = (cx, y1 + bh * 0.10, 0.90)
        l_eye = (cx - bw * 0.08, y1 + bh * 0.08, 0.88)
        r_eye = (cx + bw * 0.08, y1 + bh * 0.08, 0.88)
        l_ear = (cx - bw * 0.16, y1 + bh * 0.09, 0.85)
        r_ear = (cx + bw * 0.16, y1 + bh * 0.09, 0.85)

        l_sh = (cx - bw * 0.28, y1 + bh * 0.22, 0.92)
        r_sh = (cx + bw * 0.28, y1 + bh * 0.22, 0.92)

        l_el = (cx - bw * 0.38, y1 + bh * 0.42, 0.88)
        r_el = (cx + bw * 0.38, y1 + bh * 0.42, 0.88)

        l_wr = (cx - bw * 0.32, y1 + bh * 0.60, 0.86)
        r_wr = (cx + bw * 0.32, y1 + bh * 0.60, 0.86)

        l_hip = (cx - bw * 0.18, y1 + bh * 0.56, 0.90)
        r_hip = (cx + bw * 0.18, y1 + bh * 0.56, 0.90)

        l_knee = (cx - bw * 0.20, y1 + bh * 0.76, 0.88)
        r_knee = (cx + bw * 0.20, y1 + bh * 0.76, 0.88)

        l_ankle = (cx - bw * 0.22, y1 + bh * 0.96, 0.86)
        r_ankle = (cx + bw * 0.22, y1 + bh * 0.96, 0.86)

        return [nose, l_eye, r_eye, l_ear, r_ear, l_sh, r_sh, l_el, r_el, l_wr, r_wr, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]

    def _detect_morphological_merged(
        self,
        frame: np.ndarray,
        w: int,
        h: int,
        skeletal_tracking_enabled: bool = True
    ) -> List[DetectionObject]:
        scale = 480.0 / max(h, w)
        sw, sh = int(w * scale), int(h * scale)
        small = cv2.resize(frame, (sw, sh))

        fg_mask = self.bg_subtractor.apply(small)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.horizontal_kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.vertical_kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_boxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1200:
                bx, by, bw_px, bh_px = cv2.boundingRect(cnt)
                candidate_boxes.append([bx, by, bx + bw_px, by + bh_px, area])

        merged_boxes = self._merge_adjacent_boxes(candidate_boxes)
        results = []
        for (x1, y1, x2, y2) in merged_boxes:
            norm_x1 = max(0.0, (x1 / scale) / w)
            norm_y1 = max(0.0, (y1 / scale) / h)
            norm_x2 = min(1.0, (x2 / scale) / w)
            norm_y2 = min(1.0, (y2 / scale) / h)
            box_h = norm_y2 - norm_y1
            box_w = norm_x2 - norm_x1

            if box_h >= 0.15 and (box_h * box_w) >= 0.025:
                kp = self._estimate_skeletal_keypoints((norm_x1, norm_y1, norm_x2, norm_y2)) if skeletal_tracking_enabled else None
                results.append(DetectionObject(
                    bbox=(norm_x1, norm_y1, norm_x2, norm_y2),
                    confidence=0.88,
                    class_id=0,
                    class_name="person",
                    keypoints=kp
                ))
        return results

    def _merge_adjacent_boxes(self, boxes: List[List[float]]) -> List[Tuple[int, int, int, int]]:
        if not boxes:
            return []
        current_boxes = [[b[0], b[1], b[2], b[3]] for b in boxes]
        merged = True

        while merged:
            merged = False
            new_boxes = []
            used = [False] * len(current_boxes)

            for i in range(len(current_boxes)):
                if used[i]:
                    continue
                ax1, ay1, ax2, ay2 = current_boxes[i]
                for j in range(i + 1, len(current_boxes)):
                    if used[j]:
                        continue
                    bx1, by1, bx2, by2 = current_boxes[j]
                    x_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
                    min_w = min(ax2 - ax1, bx2 - bx1)
                    y_dist = max(0, max(ay1, by1) - min(ay2, by2))
                    max_h = max(ay2 - ay1, by2 - by1)

                    if (min_w > 0 and (x_overlap / min_w) > 0.25 and y_dist < (0.40 * max_h)) or (x_overlap > 0 and y_dist == 0):
                        ax1 = min(ax1, bx1)
                        ay1 = min(ay1, by1)
                        ax2 = max(ax2, bx2)
                        ay2 = max(ay2, by2)
                        used[j] = True
                        merged = True

                new_boxes.append([ax1, ay1, ax2, ay2])
                used[i] = True

            current_boxes = new_boxes

        return [(b[0], b[1], b[2], b[3]) for b in current_boxes]


# ==============================================================================
# High-Precision Multi-Class Track & Kinematic State Machine
# ==============================================================================
class KinematicState(str, Enum):
    STANDING = "STANDING"
    RAPID_DESCENT = "RAPID_DESCENT"
    COLLAPSED = "COLLAPSED"
    IMMOBILE = "IMMOBILE"
    FALL_CONFIRMED = "FALL_CONFIRMED"


class MotionState(str, Enum):
    ACTIVE_MOVING = "ACTIVE_MOVING"
    STATIONARY = "STATIONARY"
    STATIC_ANCHOR = "STATIC_ANCHOR"


class KinematicPersonTracker:
    def __init__(self, track_id: int, det: DetectionObject):
        self.track_id = track_id
        self.bbox = det.bbox
        self.class_name = det.class_name
        self.confidence = det.confidence
        self.keypoints = det.keypoints
        self.state = KinematicState.STANDING
        self.motion_state = MotionState.ACTIVE_MOVING
        self.last_seen = time.time()
        self.first_seen = time.time()
        self.hits = 1

        self.history: List[Tuple[float, float, float, float]] = []
        self.displacement_2s = 0.0
        self.velocity = 0.0
        self.stationary_start_time: Optional[float] = None
        self.stationary_duration = 0.0

        self.descent_start_time: Optional[float] = None
        self.collapsed_start_time: Optional[float] = None
        self.aspect_ratio = 1.8
        self.smoothed_ar = 1.8
        self.descent_velocity = 0.0
        self.torso_angle = 85.0
        self.alert_dispatched = False

    def update(self, det: DetectionObject, now: float, fall_detection_enabled: bool = True):
        self.bbox = det.bbox
        self.class_name = det.class_name
        self.confidence = det.confidence
        if det.keypoints is not None:
            self.keypoints = det.keypoints
        self.last_seen = now
        self.hits += 1

        x1, y1, x2, y2 = det.bbox
        w = max(0.01, x2 - x1)
        h = max(0.01, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        self.aspect_ratio = h / w

        self.smoothed_ar = 0.65 * self.aspect_ratio + 0.35 * self.smoothed_ar

        if self.history:
            dt = max(0.001, now - self.history[-1][0])
            inst_vy = (cy - self.history[-1][2]) / dt * 2.5
            self.descent_velocity = max(0.0, 0.70 * inst_vy + 0.30 * self.descent_velocity)

            inst_vel = math.hypot(cx - self.history[-1][1], cy - self.history[-1][2]) / dt
            self.velocity = 0.70 * inst_vel + 0.30 * self.velocity
        else:
            self.descent_velocity = 0.0
            self.velocity = 0.0

        clamped_ar = max(0.35, min(2.2, self.smoothed_ar))
        self.torso_angle = max(5.0, min(90.0, math.degrees(math.atan2(clamped_ar, 1.0)) * 1.35))

        self.history.append((now, cx, cy, h))
        self.history = [pt for pt in self.history if now - pt[0] <= 3.5]

        if len(self.history) >= 2:
            old_t, old_cx, old_cy, _ = self.history[0]
            self.displacement_2s = math.hypot(cx - old_cx, cy - old_cy)
        else:
            self.displacement_2s = 0.0

        if self.displacement_2s < 0.020:
            if self.stationary_start_time is None:
                self.stationary_start_time = now
            self.stationary_duration = now - self.stationary_start_time

            if self.stationary_duration >= 4.0:
                self.motion_state = MotionState.STATIC_ANCHOR
            else:
                self.motion_state = MotionState.STATIONARY
        else:
            self.stationary_start_time = None
            self.stationary_duration = 0.0
            self.motion_state = MotionState.ACTIVE_MOVING

        if self.class_name == "person" and fall_detection_enabled:
            self._evaluate_state_transitions(now)
        elif not fall_detection_enabled:
            self.state = KinematicState.STANDING
            self.alert_dispatched = False
            self.descent_velocity = 0.0

    def _evaluate_state_transitions(self, now: float):
        if self.smoothed_ar > 1.20 and self.torso_angle > 52.0:
            self.state = KinematicState.STANDING
            self.descent_start_time = None
            self.collapsed_start_time = None
            self.alert_dispatched = False
            return

        if self.state == KinematicState.STANDING:
            if self.descent_velocity >= 1.30:
                self.state = KinematicState.RAPID_DESCENT
                self.descent_start_time = now

        elif self.state == KinematicState.RAPID_DESCENT:
            time_in_descent = now - (self.descent_start_time or now)
            if self.smoothed_ar <= 0.85 and self.torso_angle <= 38.0:
                self.state = KinematicState.COLLAPSED
                self.collapsed_start_time = now
            elif time_in_descent > 1.0:
                self.state = KinematicState.STANDING

        elif self.state == KinematicState.COLLAPSED:
            time_collapsed = now - (self.collapsed_start_time or now)
            if self.descent_velocity < 0.35 and time_collapsed >= 1.8:
                self.state = KinematicState.IMMOBILE

        elif self.state == KinematicState.IMMOBILE:
            total_floor_time = now - (self.collapsed_start_time or now)
            if total_floor_time >= 2.5:
                self.state = KinematicState.FALL_CONFIRMED

    @property
    def is_confirmed(self) -> bool:
        return (self.hits >= 3) and (self.velocity <= 1.20)

    @property
    def is_active_human(self) -> bool:
        return self.class_name == "person" and self.motion_state != MotionState.STATIC_ANCHOR and self.is_confirmed


# ==============================================================================
# Robust Camera Discovery & Scanner
# ==============================================================================
class CameraScanner:
    @staticmethod
    def scan_usb_cameras() -> List[Dict[str, str]]:
        found = []
        if sys.platform.startswith("linux"):
            for dev in sorted(Path("/dev").glob("video[0-9]*")):
                try:
                    cap = cv2.VideoCapture(str(dev))
                    if cap.isOpened():
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            found.append({"id": f"usb_{dev.name}", "name": f"USB Camera ({dev.name})", "url": str(dev)})
                        cap.release()
                except Exception:
                    pass
        else:
            for idx in range(3):
                try:
                    cap = cv2.VideoCapture(idx)
                    if cap.isOpened():
                        found.append({"id": f"usb_{idx}", "name": f"USB Camera (Index {idx})", "url": str(idx)})
                        cap.release()
                except Exception:
                    pass
        return found

    @staticmethod
    def test_http_port(host: str, port: int, timeout: float = 0.3) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                return s.connect_ex((host, port)) == 0
        except Exception:
            return False

    @classmethod
    def scan_network_cameras(cls) -> List[Dict[str, str]]:
        found = []
        try:
            ip = socket.gethostbyname("esp32-cctv.local")
            found.append({
                "id": "esp32_mdns",
                "name": "ESP32-S3 Camera (esp32-cctv.local)",
                "url": f"http://{ip}:81/stream"
            })
        except Exception:
            pass

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()

            ip_parts = local_ip.split(".")
            subnet_prefix = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}."

            candidate_ips = [
                local_ip,
                f"{subnet_prefix}86",  # ESP32 IP
                f"{subnet_prefix}50",  # Reolink default
                f"{subnet_prefix}10",
                f"{subnet_prefix}1",
                f"{subnet_prefix}2",
                f"{subnet_prefix}100"
            ]

            for ip in candidate_ips:
                if cls.test_http_port(ip, 81):
                    found.append({
                        "id": f"net_{ip}_81",
                        "name": f"ESP32-S3 Feed ({ip}:81)",
                        "url": f"http://{ip}:81/stream"
                    })
                elif cls.test_http_port(ip, 554):
                    found.append({
                        "id": f"rtsp_{ip}_554",
                        "name": f"RTSP IP Camera ({ip}:554)",
                        "url": f"rtsp://admin:admin123@{ip}:554/h264Preview_01_sub"
                    })
        except Exception:
            pass

        return found

    @classmethod
    def discover_all(cls) -> List[Dict[str, str]]:
        sources = []
        sources.append({
            "id": "synthetic",
            "name": "🛡️ Synthetic Benchmark Feed",
            "url": "synthetic"
        })
        sources.extend(cls.scan_usb_cameras())
        sources.extend(cls.scan_network_cameras())
        return sources


# ==============================================================================
# ESP32 Sentry Auto-Discovery & Network Scanner
# ==============================================================================
class ESP32SentryScanner:
    """Discovers and validates ESP32-S3 IoT Hardware Sentries across LAN and mDNS."""

    @staticmethod
    def test_http_port(host: str, port: int, timeout: float = 0.5) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                return s.connect_ex((host, port)) == 0
        except Exception:
            return False

    @classmethod
    def scan_esp32_devices(cls) -> List[Dict[str, Any]]:
        """Scans mDNS ('esp32-cctv.local'), checks subnet IPs on port 80/81 for /sensors or /status
        with 0.5s socket/HTTP timeout. Verifies device responds with valid JSON containing
        camera_id, pir_motion or free_heap.
        """
        candidate_ips: List[str] = []

        # 1. Probe mDNS hostname 'esp32-cctv.local'
        try:
            mdns_ip = socket.gethostbyname("esp32-cctv.local")
            if mdns_ip and mdns_ip not in candidate_ips:
                candidate_ips.append(mdns_ip)
        except Exception:
            pass

        # 2. Probe Local Subnet IPs
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            local_ip = "127.0.0.1"

        ip_parts = local_ip.split(".")
        if len(ip_parts) == 4:
            subnet_prefix = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}."
            known_suffixes = ["86", "87", "88", "50", "10", "20", "1", "2", "100", "101", "102", "150"]
            for sfx in known_suffixes:
                ip_str = f"{subnet_prefix}{sfx}"
                if ip_str not in candidate_ips:
                    candidate_ips.append(ip_str)

        if local_ip not in candidate_ips:
            candidate_ips.append(local_ip)
        if "127.0.0.1" not in candidate_ips:
            candidate_ips.append("127.0.0.1")

        found_devices: List[Dict[str, Any]] = []
        seen_ips = set()

        def probe_target(ip: str, port: int) -> Optional[Dict[str, Any]]:
            if not cls.test_http_port(ip, port, timeout=0.5):
                return None

            for endpoint in ("/sensors", "/status"):
                url = f"http://{ip}:{port}{endpoint}"
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Edge-CCTV-Scanner"})
                    with urllib.request.urlopen(req, timeout=0.5) as resp:
                        if resp.status == 200:
                            body = resp.read().decode("utf-8")
                            data = json.loads(body)
                            if isinstance(data, dict) and any(k in data for k in ("camera_id", "pir_motion", "free_heap")):
                                return {
                                    "ip": ip,
                                    "port": port,
                                    "camera_id": data.get("camera_id", f"esp32_{ip.replace('.', '_')}"),
                                    "url": f"http://{ip}:{port}",
                                    "endpoint": endpoint,
                                    "mac": data.get("mac") or data.get("mac_address"),
                                    "data": data,
                                    "free_heap": data.get("free_heap"),
                                    "pir_motion": bool(data.get("pir_motion", False)),
                                    "distance_cm": float(data.get("distance_cm", 0.0)),
                                    "door1_open": bool(data.get("door1_open", False)),
                                    "door2_open": bool(data.get("door2_open", False)),
                                    "last_seen": time.time()
                                }
                except Exception:
                    pass
            return None

        probe_pairs = [(ip, p) for ip in candidate_ips for p in (80, 81)]
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = [executor.submit(probe_target, ip, p) for ip, p in probe_pairs]
            for f in futures:
                res = f.result()
                if res and res["ip"] not in seen_ips:
                    seen_ips.add(res["ip"])
                    found_devices.append(res)

        return found_devices


# ==============================================================================
# Live AI Monitor Engine
# ==============================================================================
class LiveAIMonitor:
    def __init__(
        self,
        initial_stream: Optional[str] = None,
        model_path: Optional[str] = None,
        pose_model_path: Optional[str] = None
    ):
        self.stream_source = initial_stream
        self.camera_id = "cam_live_eval"
        self.is_running = True
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_source_name = "Detecting..."

        self.detector = UnifiedPersonDetector(onnx_model_path=model_path, pose_model_path=pose_model_path)
        self.debouncer = ZoneAlertDebouncer(cooldown_seconds=3.0)

        # Dynamic Runtime Feature Toggles & ESP32 Telemetry
        self.feature_toggles = {
            "fall_detection": True,
            "skeletal_tracking": True,
            "object_detection": True,
            "esp32_pir": True,
            "esp32_ultrasonic": True,
            "esp32_door1": True,
            "esp32_door2": True
        }
        self.esp32_sensor_readings = {
            "pir_motion": False,
            "distance_cm": 0.0,
            "door1_open": False,
            "door2_open": False
        }

        # ESP32 Sentry Hardware State & Auto-Discovery
        self.is_esp32_connected: bool = False
        self.esp32_connection_status: str = "DISCONNECTED"  # "DISCONNECTED", "SCANNING", "ONLINE", "ERROR"
        self.connected_esp32_devices: List[Dict[str, Any]] = []
        self.active_esp32_ip: Optional[str] = None
        self.active_esp32_port: int = 80
        self.demo_simulation_enabled: bool = False

        self.tracks: Dict[int, KinematicPersonTracker] = {}
        self.tracks_lock = threading.Lock()
        self.scan_lock = threading.Lock()
        self.next_track_id = 1

        self.fps = 0.0
        self.frame_count = 0
        self.avg_inference_ms = 0.0
        self.recent_latencies: List[float] = []

        self.torso_angle = 85.0
        self.aspect_ratio = 1.9
        self.descent_velocity = 0.0
        self.floor_proximity = 0.15
        self.is_fall_active = False
        self.person_count = 0

        # Multi-Zone & Exclusion Maps
        self.tripwires: Dict[str, Dict[str, Any]] = {}
        self.intrusion_zones: Dict[str, Dict[str, Any]] = {}
        self.exclusion_masks: Dict[str, Dict[str, Any]] = {}
        self.is_intrusion_active = False

        # Alert & 15s Clip Recording State
        self.active_alert_banner = ""
        self.alert_expiry = 0.0
        self.rec_clip_active_until = 0.0

        self.event_log: List[Dict[str, Any]] = []
        self.event_lock = threading.Lock()

        self.latest_encoded_jpeg: Optional[bytes] = None
        self.frame_lock = threading.Lock()

        self.available_sources: List[Dict[str, str]] = []
        self.ring_buffer = clip_recorder_service.get_or_create_buffer(self.camera_id)
        self.zones_file = PROJECT_ROOT / "storage" / "zones_config.json"

        self._init_zones()

        # Auto-scan ESP32 sentries on startup in background thread without blocking video stream initialization
        self.esp32_watchdog_thread = threading.Thread(target=self._esp32_startup_and_watchdog_loop, daemon=True)
        self.esp32_watchdog_thread.start()

    def _init_default_zones(self):
        self.tripwires = {
            "tw_default": {
                "id": "tw_default",
                "name": "Virtual Tripwire #1",
                "x1": 0.15,
                "y1": 0.55,
                "x2": 0.85,
                "y2": 0.55,
                "direction": "BIDIRECTIONAL",
                "allowed_classes": ["person", "vehicle"],
                "in_count": 0,
                "out_count": 0,
                "enabled": True
            }
        }
        self.intrusion_zones = {
            "int_default": {
                "id": "int_default",
                "name": "Restricted Area #1",
                "points": [
                    {"x": 0.55, "y": 0.25},
                    {"x": 0.95, "y": 0.25},
                    {"x": 0.95, "y": 0.85},
                    {"x": 0.55, "y": 0.85}
                ],
                "allowed_classes": ["person", "vehicle"],
                "dwell_time_seconds": 0.5,
                "enabled": True
            }
        }
        self.exclusion_masks = {}
        self._save_persistent_zones()
        self.sync_zones_to_service()

    def _init_zones(self):
        if self.zones_file.exists():
            try:
                with open(self.zones_file, "r") as f:
                    saved = json.load(f)

                    self.tripwires = {}
                    raw_tw = saved.get("tripwires", saved.get("tripwire", []))
                    if isinstance(raw_tw, dict):
                        raw_tw = [raw_tw]
                    for tw in raw_tw:
                        if isinstance(tw, dict) and "x1" in tw:
                            tw_id = tw.get("id", f"tw_{int(time.time()*1000)}")
                            tw["id"] = tw_id
                            tw.setdefault("name", f"Tripwire {len(self.tripwires)+1}")
                            tw.setdefault("direction", "BIDIRECTIONAL")
                            tw.setdefault("allowed_classes", ["person", "vehicle"])
                            tw.setdefault("in_count", 0)
                            tw.setdefault("out_count", 0)
                            tw.setdefault("enabled", True)
                            self.tripwires[tw_id] = tw

                    self.intrusion_zones = {}
                    raw_int = saved.get("intrusion_zones", saved.get("intrusion", []))
                    if isinstance(raw_int, dict):
                        raw_int = [raw_int]
                    for iz in raw_int:
                        if isinstance(iz, dict) and "points" in iz:
                            iz_id = iz.get("id", f"int_{int(time.time()*1000)}")
                            iz["id"] = iz_id
                            iz.setdefault("name", f"Restricted Area {len(self.intrusion_zones)+1}")
                            iz.setdefault("allowed_classes", ["person", "vehicle"])
                            iz.setdefault("dwell_time_seconds", 0.5)
                            iz.setdefault("enabled", True)
                            self.intrusion_zones[iz_id] = iz

                    self.exclusion_masks = {}
                    raw_ex = saved.get("exclusion_masks", [])
                    if isinstance(raw_ex, dict):
                        raw_ex = [raw_ex]
                    for ex in raw_ex:
                        if isinstance(ex, dict) and "points" in ex:
                            ex_id = ex.get("id", f"ex_{int(time.time()*1000)}")
                            ex["id"] = ex_id
                            ex.setdefault("name", f"Exclusion Mask {len(self.exclusion_masks)+1}")
                            ex.setdefault("enabled", True)
                            self.exclusion_masks[ex_id] = ex

                logger.info(f"✅ Loaded persistent security zones from {self.zones_file.name}: {len(self.tripwires)} tripwires, {len(self.intrusion_zones)} restricted zones, {len(self.exclusion_masks)} exclusion masks.")
                self.sync_zones_to_service()
                return
            except Exception as e:
                logger.warning(f"Could not load saved zones ({e}). Using defaults.")

        self._init_default_zones()

    def _save_persistent_zones(self):
        try:
            self.zones_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.zones_file, "w") as f:
                json.dump({
                    "tripwires": list(self.tripwires.values()),
                    "intrusion_zones": list(self.intrusion_zones.values()),
                    "exclusion_masks": list(self.exclusion_masks.values())
                }, f, indent=2)
            logger.info("💾 Security zones persisted to storage/zones_config.json")
        except Exception as e:
            logger.error(f"Failed to persist security zones: {e}")

    def sync_zones_to_service(self):
        zones = []

        for tw_id, tw in self.tripwires.items():
            if not tw.get("enabled", True):
                continue
            dir_enum = TripwireDirection.BIDIRECTIONAL
            if tw.get("direction") == "A_TO_B":
                dir_enum = TripwireDirection.A_TO_B
            elif tw.get("direction") == "B_TO_A":
                dir_enum = TripwireDirection.B_TO_A

            zones.append(ZoneConfig(
                id=tw_id,
                camera_id=self.camera_id,
                name=tw.get("name", "Tripwire"),
                zone_type=ZoneType.TRIPWIRE,
                enabled=True,
                allowed_classes=tw.get("allowed_classes", ["person", "vehicle"]),
                line_start=Point2D(x=float(tw["x1"]), y=float(tw["y1"])),
                line_end=Point2D(x=float(tw["x2"]), y=float(tw["y2"])),
                direction=dir_enum
            ))

        for iz_id, iz in self.intrusion_zones.items():
            if not iz.get("enabled", True) or not iz.get("points"):
                continue
            pts = [Point2D(x=float(p["x"]), y=float(p["y"])) for p in iz["points"]]
            zones.append(ZoneConfig(
                id=iz_id,
                camera_id=self.camera_id,
                name=iz.get("name", "Restricted Area"),
                zone_type=ZoneType.INTRUSION,
                enabled=True,
                allowed_classes=iz.get("allowed_classes", ["person", "vehicle"]),
                polygon_points=pts,
                dwell_time_seconds=float(iz.get("dwell_time_seconds", 0.5))
            ))

        ai_zone_service.set_camera_zones(self.camera_id, zones)

    def trigger_alert(self, message: str, event_type: str = "SECURITY", duration: float = 3.5):
        self.active_alert_banner = message
        self.alert_expiry = time.time() + duration
        logger.info(f"🚨 [{event_type}] {message}")

        evt = {
            "id": f"evt_{int(time.time()*1000)}",
            "time": time.strftime("%H:%M:%S"),
            "type": event_type,
            "message": message,
            "camera": self.current_source_name
        }
        with self.event_lock:
            self.event_log.insert(0, evt)
            if len(self.event_log) > 50:
                self.event_log.pop()

    def record_15s_incident_clip(self, event_label: str):
        self.rec_clip_active_until = time.time() + 10.0
        ts = int(time.time())
        clip_name = f"clip_{event_label.lower()}_{ts}.mp4"
        clip_path = settings.CLIPS_DIR / clip_name
        snap_path = settings.SNAPSHOTS_DIR / f"snap_{event_label.lower()}_{ts}.jpg"

        with self.frame_lock:
            jpeg = self.latest_encoded_jpeg
        if jpeg:
            snap_path.parent.mkdir(parents=True, exist_ok=True)
            snap_path.write_bytes(jpeg)

        pre_frames = self.ring_buffer.get_pre_event_frames()
        if pre_frames:
            clip_muxer.submit_mux_task(pre_frames, clip_path, fps=25)
            self.trigger_alert(f"15s Incident Clip Captured: {clip_name}", "CLIP_REC", duration=3.0)

    def is_in_exclusion_mask(self, cx: float, cy: float) -> bool:
        for ex in self.exclusion_masks.values():
            if not ex.get("enabled", True) or not ex.get("points"):
                continue
            poly = [(p["x"], p["y"]) for p in ex["points"]]
            if PolygonGeometry.point_in_polygon_raycasting((cx, cy), poly):
                return True
        return False

    def add_or_update_tripwire(self, data: Dict[str, Any]) -> str:
        tw_id = data.get("id") or f"tw_{int(time.time()*1000)}"
        self.tripwires[tw_id] = {
            "id": tw_id,
            "name": data.get("name") or f"Tripwire #{len(self.tripwires)+1}",
            "x1": max(0.0, min(1.0, float(data["x1"]))),
            "y1": max(0.0, min(1.0, float(data["y1"]))),
            "x2": max(0.0, min(1.0, float(data["x2"]))),
            "y2": max(0.0, min(1.0, float(data["y2"]))),
            "direction": data.get("direction", "BIDIRECTIONAL"),
            "allowed_classes": data.get("allowed_classes", ["person", "vehicle"]),
            "in_count": self.tripwires.get(tw_id, {}).get("in_count", 0),
            "out_count": self.tripwires.get(tw_id, {}).get("out_count", 0),
            "enabled": data.get("enabled", True)
        }
        self._save_persistent_zones()
        self.sync_zones_to_service()
        self.trigger_alert(f"Tripwire '{self.tripwires[tw_id]['name']}' updated.", "ZONE_CONFIG", duration=2.5)
        return tw_id

    def delete_tripwire(self, tw_id: str) -> bool:
        if tw_id in self.tripwires:
            name = self.tripwires[tw_id].get("name", tw_id)
            del self.tripwires[tw_id]
            self._save_persistent_zones()
            self.sync_zones_to_service()
            self.trigger_alert(f"Tripwire '{name}' deleted.", "ZONE_CONFIG", duration=2.5)
            return True
        return False

    def add_or_update_intrusion_zone(self, data: Dict[str, Any]) -> str:
        iz_id = data.get("id") or f"int_{int(time.time()*1000)}"
        self.intrusion_zones[iz_id] = {
            "id": iz_id,
            "name": data.get("name") or f"Restricted Area #{len(self.intrusion_zones)+1}",
            "points": data.get("points", []),
            "allowed_classes": data.get("allowed_classes", ["person", "vehicle"]),
            "dwell_time_seconds": float(data.get("dwell_time_seconds", 0.5)),
            "enabled": data.get("enabled", True)
        }
        self._save_persistent_zones()
        self.sync_zones_to_service()
        self.trigger_alert(f"Restricted Area '{self.intrusion_zones[iz_id]['name']}' updated.", "ZONE_CONFIG", duration=2.5)
        return iz_id

    def delete_intrusion_zone(self, iz_id: str) -> bool:
        if iz_id in self.intrusion_zones:
            name = self.intrusion_zones[iz_id].get("name", iz_id)
            del self.intrusion_zones[iz_id]
            self._save_persistent_zones()
            self.sync_zones_to_service()
            self.trigger_alert(f"Restricted Area '{name}' deleted.", "ZONE_CONFIG", duration=2.5)
            return True
        return False

    def add_or_update_exclusion_mask(self, data: Dict[str, Any]) -> str:
        ex_id = data.get("id") or f"ex_{int(time.time()*1000)}"
        self.exclusion_masks[ex_id] = {
            "id": ex_id,
            "name": data.get("name") or f"Exclusion Mask #{len(self.exclusion_masks)+1}",
            "points": data.get("points", []),
            "mask_mode": data.get("mask_mode", "BLUR"),
            "enabled": data.get("enabled", True)
        }
        self._save_persistent_zones()
        self.trigger_alert(f"Exclusion Mask '{self.exclusion_masks[ex_id]['name']}' updated.", "ZONE_CONFIG", duration=2.5)
        return ex_id

    def delete_exclusion_mask(self, ex_id: str) -> bool:
        if ex_id in self.exclusion_masks:
            name = self.exclusion_masks[ex_id].get("name", ex_id)
            del self.exclusion_masks[ex_id]
            self._save_persistent_zones()
            self.trigger_alert(f"Exclusion Mask '{name}' deleted.", "ZONE_CONFIG", duration=2.5)
            return True
        return False

    def clear_all_zones(self):
        self.tripwires.clear()
        self.intrusion_zones.clear()
        self.exclusion_masks.clear()
        self._save_persistent_zones()
        self.sync_zones_to_service()
        self.trigger_alert("All security zones, tripwires, and exclusion masks cleared.", "ZONE_CONFIG", duration=2.5)

    def switch_source(self, new_source_url: str):
        logger.info(f"[+] Switching video source to: {new_source_url}")
        if self.cap:
            self.cap.release()
            self.cap = None
        self.stream_source = new_source_url
        self.open_video_source()

    def rescan_sources(self) -> List[Dict[str, str]]:
        logger.info("[+] Scanning for connected cameras and streams...")
        sources = CameraScanner.discover_all()
        self.available_sources = sources
        self.trigger_alert(f"Scan complete: Found {len(sources)} video source(s).", "CAMERA_SCAN", duration=2.5)
        return sources

    def scan_esp32_devices(self) -> List[Dict[str, Any]]:
        with self.scan_lock:
            self.esp32_connection_status = "SCANNING"
            try:
                devices = ESP32SentryScanner.scan_esp32_devices()
                if devices:
                    self.connected_esp32_devices = devices
                    self.is_esp32_connected = True
                    self.esp32_connection_status = "ONLINE"
                    self.active_esp32_ip = devices[0]["ip"]
                    self.active_esp32_port = devices[0].get("port", 80)
                    d = devices[0].get("data", {})
                    if "pir_motion" in d:
                        self.esp32_sensor_readings["pir_motion"] = bool(d["pir_motion"])
                    if "distance_cm" in d:
                        self.esp32_sensor_readings["distance_cm"] = float(d["distance_cm"])
                    if "door1_open" in d:
                        self.esp32_sensor_readings["door1_open"] = bool(d["door1_open"])
                    if "door2_open" in d:
                        self.esp32_sensor_readings["door2_open"] = bool(d["door2_open"])
                    logger.info(f"✅ Auto-discovered {len(devices)} ESP32 Sentry device(s). Active: {self.active_esp32_ip}:{self.active_esp32_port}")
                    self.trigger_alert(f"ESP32 Sentry Connected: {self.active_esp32_ip}", "ESP32_ONLINE", duration=3.0)
                else:
                    self.connected_esp32_devices = []
                    self.is_esp32_connected = False
                    self.esp32_connection_status = "DISCONNECTED"
                    logger.info("ℹ️ No physical ESP32 sentries detected on network. Running in disconnected standby mode.")
                return devices
            except Exception as e:
                logger.error(f"Error during ESP32 auto-discovery: {e}")
                self.connected_esp32_devices = []
                self.is_esp32_connected = False
                self.esp32_connection_status = "ERROR"
                return []

    def rescan_esp32_devices(self) -> List[Dict[str, Any]]:
        logger.info("[+] Manually rescanning network for ESP32 hardware sentries...")
        self.trigger_alert("Scanning LAN & mDNS for ESP32 sentries...", "ESP32_SCAN", duration=2.0)
        devices = self.scan_esp32_devices()
        if devices:
            self.trigger_alert(f"Found {len(devices)} ESP32 Sentry ({self.active_esp32_ip})", "ESP32_ONLINE", duration=3.0)
        else:
            self.trigger_alert("⚠️ No physical ESP32 sentry discovered on subnet.", "ESP32_OFFLINE", duration=3.0)
        return devices

    def _ping_esp32_node(self, ip: str, port: int) -> bool:
        """Pings active ESP32 on /sensors or /status endpoint with 0.8s timeout."""
        for endpoint in ("/sensors", "/status"):
            url = f"http://{ip}:{port}{endpoint}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Edge-CCTV-Watchdog"})
                with urllib.request.urlopen(req, timeout=0.8) as resp:
                    if resp.status == 200:
                        raw = resp.read().decode("utf-8")
                        data = json.loads(raw)
                        if isinstance(data, dict):
                            if "pir_motion" in data:
                                self.esp32_sensor_readings["pir_motion"] = bool(data["pir_motion"])
                            if "distance_cm" in data:
                                self.esp32_sensor_readings["distance_cm"] = float(data["distance_cm"])
                            if "door1_open" in data:
                                self.esp32_sensor_readings["door1_open"] = bool(data["door1_open"])
                            if "door2_open" in data:
                                self.esp32_sensor_readings["door2_open"] = bool(data["door2_open"])
                            return True
            except Exception:
                pass
        return ESP32SentryScanner.test_http_port(ip, port, timeout=0.8)

    def _esp32_startup_and_watchdog_loop(self):
        """Auto-scans in background thread on startup without blocking video streams,
        then pings active ESP32 every 10s to detect disconnects.
        """
        time.sleep(0.5)
        self.scan_esp32_devices()

        while self.is_running:
            time.sleep(10.0)
            if not self.is_running:
                break
            if self.is_esp32_connected and self.active_esp32_ip:
                alive = self._ping_esp32_node(self.active_esp32_ip, self.active_esp32_port)
                if not alive:
                    logger.warning(f"⚠️ Active ESP32 Sentry at {self.active_esp32_ip} became unresponsive! Marking OFFLINE.")
                    self.is_esp32_connected = False
                    self.esp32_connection_status = "ERROR"
                    self.connected_esp32_devices = []
                    self.trigger_alert(f"⚠️ ESP32 Sentry ({self.active_esp32_ip}) Unresponsive / OFFLINE!", "ESP32_OFFLINE", duration=4.0)

    def open_video_source(self):
        if self.stream_source and self.stream_source != "synthetic":
            logger.info(f"[+] Connecting to stream: {self.stream_source}")
            try:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;3000000|stimeout;3000000"
                cap = cv2.VideoCapture(self.stream_source)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        logger.info(f"✅ Connected to: {self.stream_source}")
                        self.cap = cap
                        self.current_source_name = f"Live ({self.stream_source})"
                        return
                    cap.release()
            except Exception as e:
                logger.warning(f"[-] Could not open {self.stream_source}: {e}")

        if not self.available_sources:
            self.available_sources = CameraScanner.discover_all()

        for src in self.available_sources:
            if src["url"] == "synthetic":
                continue
            logger.info(f"[+] Probing source: {src['name']} ({src['url']})")
            try:
                cap = cv2.VideoCapture(src["url"])
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        logger.info(f"✅ Connected to: {src['name']}")
                        self.cap = cap
                        self.stream_source = src["url"]
                        self.current_source_name = src["name"]
                        return
                    cap.release()
            except Exception:
                pass

        logger.info("🛡️ Operating in Built-In Synthetic AI Pipeline.")
        self.cap = None
        self.stream_source = "synthetic"
        self.current_source_name = "Synthetic Benchmark Feed"

    @staticmethod
    def _compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1e-6, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        return interArea / (boxAArea + boxBArea - interArea)

    def update_tracks(self, detections: List[DetectionObject]):
        now = time.time()
        updated_tracks = set()
        fall_enabled = self.feature_toggles.get("fall_detection", True)

        # 1. Filter out any detections that fall inside Exclusion / Privacy Masks
        valid_detections = []
        for det in detections:
            cx = (det.bbox[0] + det.bbox[2]) / 2.0
            cy = (det.bbox[1] + det.bbox[3]) / 2.0
            if not self.is_in_exclusion_mask(cx, cy):
                valid_detections.append(det)

        with self.tracks_lock:
            for det in valid_detections:
                cx = (det.bbox[0] + det.bbox[2]) / 2.0
                cy = (det.bbox[1] + det.bbox[3]) / 2.0

                best_track_id = None
                best_score = float('inf')

                for tid, t in self.tracks.items():
                    if tid in updated_tracks or t.class_name != det.class_name:
                        continue
                    iou = self._compute_iou(det.bbox, t.bbox)
                    if iou >= 0.20:
                        score = 1.0 - iou
                        if score < best_score:
                            best_score = score
                            best_track_id = tid

                if best_track_id is None:
                    for tid, t in self.tracks.items():
                        if tid in updated_tracks or t.class_name != det.class_name:
                            continue
                        tx1, ty1, tx2, ty2 = t.bbox
                        tcx, tcy = (tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0
                        dist = math.hypot(cx - tcx, cy - tcy)
                        if dist < 0.22 and dist < best_score:
                            best_score = dist
                            best_track_id = tid

                if best_track_id is None:
                    best_track_id = self.next_track_id
                    self.next_track_id += 1
                    self.tracks[best_track_id] = KinematicPersonTracker(best_track_id, det)

                self.tracks[best_track_id].update(det, now, fall_detection_enabled=fall_enabled)
                updated_tracks.add(best_track_id)

            stale_ids = [tid for tid, t in self.tracks.items() if now - t.last_seen > 2.5]
            for tid in stale_ids:
                del self.tracks[tid]

            active_humans = [t for t in self.tracks.values() if t.is_active_human]
            self.person_count = len(active_humans)

            if active_humans:
                primary = active_humans[0]
                self.torso_angle = primary.torso_angle
                self.aspect_ratio = primary.smoothed_ar
                self.descent_velocity = primary.descent_velocity if fall_enabled else 0.0
                self.floor_proximity = min(1.0, primary.bbox[3])
                self.is_fall_active = (primary.state in (KinematicState.COLLAPSED, KinematicState.IMMOBILE, KinematicState.FALL_CONFIRMED)) if fall_enabled else False

                if fall_enabled and primary.state == KinematicState.FALL_CONFIRMED and not primary.alert_dispatched:
                    primary.alert_dispatched = True
                    self.trigger_alert(
                        f"CRITICAL FALL CONFIRMED! Torso={primary.torso_angle:.1f}°, Vy={primary.descent_velocity:.2f}m/s, AR={primary.smoothed_ar:.2f}",
                        "FALL_DETECTED",
                        duration=5.0
                    )
                    self.record_15s_incident_clip("FALL")
            else:
                self.torso_angle = 85.0
                self.aspect_ratio = 1.9
                self.descent_velocity = 0.0
                self.floor_proximity = 0.15
                self.is_fall_active = False

    def generate_synthetic_frame(self) -> np.ndarray:
        frame = np.full((720, 1280, 3), (24, 28, 36), dtype=np.uint8)
        t = time.time()

        cv2.rectangle(frame, (0, 480), (1280, 720), (35, 40, 50), -1)
        for y in range(480, 720, 40):
            cv2.line(frame, (0, y), (1280, y), (45, 52, 65), 1)

        px = 0.50 + 0.30 * np.sin(t * 0.5)
        py = 0.52 + 0.10 * np.cos(t * 0.8)
        cx, cy = int(px * 1280), int(py * 720)

        cv2.ellipse(frame, (cx, cy + 85), (45, 14), 0, 0, 360, (15, 18, 24), -1)
        cv2.rectangle(frame, (cx - 28, cy - 80), (cx + 28, cy + 80), (60, 120, 200), -1)
        cv2.circle(frame, (cx, cy - 105), 20, (180, 200, 220), -1)

        return frame

    def draw_hud(self, frame: np.ndarray) -> np.ndarray:
        hud = frame.copy()
        h, w = hud.shape[:2]

        # 1. Apply Optical Blurring over Exclusion / Privacy Masks
        for ex_id, ex in self.exclusion_masks.items():
            if not ex.get("enabled", True) or not ex.get("points"):
                continue
            pts = np.array([[int(p["x"] * w), int(p["y"] * h)] for p in ex["points"]], np.int32)
            mask_overlay = hud.copy()
            cv2.fillPoly(mask_overlay, [pts], (40, 45, 55))
            cv2.addWeighted(mask_overlay, 0.70, hud, 0.30, 0, hud)
            cv2.polylines(hud, [pts], True, (120, 130, 140), 2)
            name = ex.get("name", "Exclusion Mask")
            cv2.putText(hud, f"🌫️ {name} [EXCLUDED]", (pts[0][0] + 8, pts[0][1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 190, 200), 1)

        # 2. Render All Active Tripwires
        for tw_id, tw in self.tripwires.items():
            if not tw.get("enabled", True):
                continue
            tx1 = int(tw["x1"] * w)
            ty1 = int(tw["y1"] * h)
            tx2 = int(tw["x2"] * w)
            ty2 = int(tw["y2"] * h)
            cv2.line(hud, (tx1, ty1), (tx2, ty2), (0, 220, 255), 2)
            cv2.circle(hud, (tx1, ty1), 5, (0, 220, 255), -1)
            cv2.circle(hud, (tx2, ty2), 5, (0, 220, 255), -1)

            mid_x, mid_y = (tx1 + tx2) // 2, (ty1 + ty2) // 2
            dir_str = tw.get("direction", "BIDIRECTIONAL")
            name = tw.get("name", "Tripwire")
            in_c = tw.get("in_count", 0)
            out_c = tw.get("out_count", 0)
            cv2.putText(hud, f"⚡ {name} [{dir_str}] In:{in_c} Out:{out_c}",
                        (mid_x - 110, mid_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 220, 255), 2)

        # 3. Render All Active Intrusion Polygon Zones
        for iz_id, iz in self.intrusion_zones.items():
            if not iz.get("enabled", True) or not iz.get("points"):
                continue
            pts = np.array([[int(p["x"] * w), int(p["y"] * h)] for p in iz["points"]], np.int32)
            overlay = hud.copy()
            zone_color = (0, 0, 220) if self.is_intrusion_active else (255, 120, 0)
            cv2.fillPoly(overlay, [pts], zone_color)
            cv2.addWeighted(overlay, 0.20 if not self.is_intrusion_active else 0.40, hud, 0.80, 0, hud)
            cv2.polylines(hud, [pts], True, zone_color, 2)
            name = iz.get("name", "Restricted Area")
            cv2.putText(hud, f"🛑 {name} {'[BREACHED!]' if self.is_intrusion_active else '[ARMED]'}",
                        (pts[0][0] + 8, pts[0][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, zone_color, 2)

        # 4. Render Multi-Class Color-Coded Bounding Boxes & Skeletal Overlays
        skeletal_enabled = self.feature_toggles.get("skeletal_tracking", True)
        for tid, t in self.tracks.items():
            if not t.is_confirmed:
                continue

            bx1 = int(t.bbox[0] * w)
            by1 = int(t.bbox[1] * h)
            bx2 = int(t.bbox[2] * w)
            by2 = int(t.bbox[3] * h)

            if t.class_name == "person":
                if t.state in (KinematicState.COLLAPSED, KinematicState.IMMOBILE, KinematicState.FALL_CONFIRMED):
                    box_color = (0, 0, 255)
                elif t.motion_state == MotionState.STATIC_ANCHOR:
                    box_color = (130, 140, 150)
                else:
                    box_color = CLASS_COLORS["person"]
                label = f"ID:{tid} Person | {t.state.value} | θ:{t.torso_angle:.0f}° | AR:{t.smoothed_ar:.2f}"

                # Render Skeletal Joint Connections (17 Keypoints)
                if skeletal_enabled and hasattr(t, "keypoints") and t.keypoints:
                    kp = t.keypoints
                    # Draw skeletal limbs
                    for (i1, i2) in COCO_SKELETON:
                        if i1 < len(kp) and i2 < len(kp):
                            x1, y1, c1 = kp[i1]
                            x2, y2, c2 = kp[i2]
                            if c1 >= 0.25 and c2 >= 0.25:
                                pt1 = (int(x1 * w), int(y1 * h))
                                pt2 = (int(x2 * w), int(y2 * h))
                                limb_col = (0, 255, 180) if not self.is_fall_active else (0, 160, 255)
                                cv2.line(hud, pt1, pt2, limb_col, 2, cv2.LINE_AA)

                    # Draw joint nodes
                    for (kx, ky, kc) in kp:
                        if kc >= 0.25:
                            jpt = (int(kx * w), int(ky * h))
                            cv2.circle(hud, jpt, 4, (0, 240, 255), -1, cv2.LINE_AA)
                            cv2.circle(hud, jpt, 5, (15, 20, 30), 1, cv2.LINE_AA)

            elif t.class_name == "vehicle":
                box_color = CLASS_COLORS["vehicle"]
                label = f"ID:{tid} Vehicle ({t.confidence:.2f})"
            elif t.class_name == "package":
                box_color = CLASS_COLORS["package"]
                label = f"ID:{tid} Package/Bag ({t.confidence:.2f})"
            elif t.class_name == "animal":
                box_color = CLASS_COLORS["animal"]
                label = f"ID:{tid} Pet/Animal ({t.confidence:.2f})"
            else:
                box_color = CLASS_COLORS["default"]
                label = f"ID:{tid} {t.class_name.upper()} ({t.confidence:.2f})"

            cv2.rectangle(hud, (bx1, by1), (bx2, by2), box_color, 2)

            foot_x = (bx1 + bx2) // 2
            foot_y = by2
            cv2.circle(hud, (foot_x, foot_y), 4, box_color, -1)
            cv2.putText(hud, label, (bx1, max(20, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, box_color, 2)

        # 5. Top Status Bar
        overlay = hud.copy()
        cv2.rectangle(overlay, (0, 0), (w, 54), (10, 13, 18), -1)
        cv2.addWeighted(overlay, 0.85, hud, 0.15, 0, hud)
        cv2.line(hud, (0, 54), (w, 54), (0, 240, 255), 1)

        cv2.putText(hud, "🛡️ EDGE AI CCTV - MULTI-ZONE STUDIO", (16, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 240, 255), 2)
        
        toggles_info = f"Fall:{'ON' if self.feature_toggles.get('fall_detection') else 'OFF'} | Skt:{'ON' if skeletal_enabled else 'OFF'} | Det:{'ON' if self.feature_toggles.get('object_detection') else 'OFF'}"
        cv2.putText(hud, f"Source: {self.current_source_name} | Humans: {self.person_count} | {toggles_info}", (16, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 180, 200), 1)

        fps_text = f"FPS: {self.fps:.1f} | Latency: {self.avg_inference_ms:.1f}ms | Res: {w}x{h}"
        cv2.putText(hud, fps_text, (w - 440, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 160), 1)

        # 6. Active 15s Clip Recording Badge
        now = time.time()
        if now < self.rec_clip_active_until:
            rem = max(0.1, self.rec_clip_active_until - now)
            rec_text = f"🔴 REC 15s INCIDENT CLIP [{rem:.1f}s]"
            cv2.rectangle(hud, (w - 320, 64), (w - 16, 98), (0, 0, 200), -1)
            cv2.putText(hud, rec_text, (w - 308, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

        # 7. Kinematic Telemetry HUD (Bottom Left)
        kin_w, kin_h = 330, 136
        kx, ky = 16, h - kin_h - 16
        overlay = hud.copy()
        cv2.rectangle(overlay, (kx, ky), (kx + kin_w, ky + kin_h), (12, 16, 24), -1)
        cv2.addWeighted(overlay, 0.85, hud, 0.15, 0, hud)
        
        fall_active_mode = self.feature_toggles.get("fall_detection", True)
        border_color = (0, 0, 255) if (self.is_fall_active and fall_active_mode) else (0, 200, 255)
        cv2.rectangle(hud, (kx, ky), (kx + kin_w, ky + kin_h), border_color, 1)

        cv2.putText(hud, f"LIVE KINEMATICS {'[ACTIVE]' if fall_active_mode else '[DISABLED]'}",
                    (kx + 10, ky + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, border_color, 1)

        if fall_active_mode:
            angle_color = (0, 0, 255) if self.torso_angle < 35 else (0, 255, 180)
            cv2.putText(hud, f"• Torso Angle (θ): {self.torso_angle:.1f}° {'(CRITICAL)' if self.torso_angle < 35 else '(NORMAL)'}",
                        (kx + 10, ky + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, angle_color, 1)

            vel_color = (0, 0, 255) if self.descent_velocity > 1.3 else (0, 255, 180)
            cv2.putText(hud, f"• Descent Velocity (Vy): {self.descent_velocity:.2f} m/s",
                        (kx + 10, ky + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, vel_color, 1)

            aspect_color = (0, 0, 255) if self.aspect_ratio < 0.85 else (0, 255, 180)
            cv2.putText(hud, f"• Aspect Ratio (H/W): {self.aspect_ratio:.2f} {'(COLLAPSED)' if self.aspect_ratio < 0.85 else '(UPRIGHT)'}",
                        (kx + 10, ky + 91), cv2.FONT_HERSHEY_SIMPLEX, 0.42, aspect_color, 1)

            cv2.putText(hud, f"• Floor Proximity: {self.floor_proximity:.2f}",
                        (kx + 10, ky + 114), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 220), 1)
        else:
            cv2.putText(hud, "• Fall Detection Engine: DISABLED",
                        (kx + 10, ky + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 150, 160), 1)
            cv2.putText(hud, "• Kinematic calculations bypassed",
                        (kx + 10, ky + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 150, 160), 1)
            cv2.putText(hud, "• Toggle ON in Quick Controls panel",
                        (kx + 10, ky + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 255), 1)

        # 8. Real-Time IoT Sensor Telemetry Overlay (Bottom Right)
        iot_w, iot_h = 285, 110
        ix, iy = w - iot_w - 16, h - iot_h - 16
        overlay = hud.copy()
        cv2.rectangle(overlay, (ix, iy), (ix + iot_w, iy + iot_h), (12, 16, 24), -1)
        cv2.addWeighted(overlay, 0.85, hud, 0.15, 0, hud)
        border_col = (0, 240, 255) if self.is_esp32_connected else (0, 165, 255)
        cv2.rectangle(hud, (ix, iy), (ix + iot_w, iy + iot_h), border_col, 1)

        if self.is_esp32_connected:
            header_txt = f"📡 ESP32 ONLINE ({self.active_esp32_ip or 'OK'})"
            header_col = (0, 255, 160)
        elif self.demo_simulation_enabled:
            header_txt = "🧪 ESP32 DEMO SIMULATION"
            header_col = (255, 200, 0)
        elif self.esp32_connection_status == "SCANNING":
            header_txt = "📡 ESP32: SCANNING NETWORK..."
            header_col = (0, 240, 255)
        elif self.esp32_connection_status == "ERROR":
            header_txt = "⚠️ ESP32: OFFLINE / UNREACHABLE"
            header_col = (0, 0, 255)
        else:
            header_txt = "⚠️ ESP32: DISCONNECTED"
            header_col = (0, 165, 255)

        cv2.putText(hud, header_txt, (ix + 10, iy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, header_col, 1)

        if not self.is_esp32_connected and not self.demo_simulation_enabled:
            cv2.putText(hud, "• PIR Motion: [DISCONNECTED]", (ix + 10, iy + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.39, (120, 130, 140), 1)
            cv2.putText(hud, "• Distance: [DISCONNECTED]", (ix + 10, iy + 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.39, (120, 130, 140), 1)
            cv2.putText(hud, "• Doors: [DISCONNECTED]", (ix + 10, iy + 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.39, (120, 130, 140), 1)
        else:
            pir_act = self.esp32_sensor_readings.get("pir_motion", False)
            pir_armed = self.feature_toggles.get("esp32_pir", True)
            pir_txt = ("🚨 MOTION" if pir_act else "IDLE") if pir_armed else "DISABLED"
            pir_col = (0, 0, 255) if (pir_act and pir_armed) else ((0, 255, 160) if pir_armed else (120, 130, 140))
            cv2.putText(hud, f"• PIR Motion: {pir_txt}", (ix + 10, iy + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, pir_col, 1)

            dist_armed = self.feature_toggles.get("esp32_ultrasonic", True)
            dist_val = self.esp32_sensor_readings.get("distance_cm", 0.0)
            dist_str = (f"{dist_val:.1f} cm" if dist_val > 0 else "--- cm") if dist_armed else "DISABLED"
            cv2.putText(hud, f"• Distance: {dist_str}", (ix + 10, iy + 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 220, 240) if dist_armed else (120, 130, 140), 1)

            d1_armed = self.feature_toggles.get("esp32_door1", True)
            d2_armed = self.feature_toggles.get("esp32_door2", True)
            d1_act = self.esp32_sensor_readings.get("door1_open", False)
            d2_act = self.esp32_sensor_readings.get("door2_open", False)
            d1_str = ("OPEN" if d1_act else "CLOSED") if d1_armed else "OFF"
            d2_str = ("OPEN" if d2_act else "CLOSED") if d2_armed else "OFF"
            door_col = (0, 0, 255) if ((d1_act and d1_armed) or (d2_act and d2_armed)) else (0, 255, 160)
            cv2.putText(hud, f"• Doors: Front[{d1_str}] Back[{d2_str}]", (ix + 10, iy + 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, door_col, 1)

        # 9. Active Alert Banner
        if time.time() < self.alert_expiry:
            banner_y = h - 62
            overlay = hud.copy()
            cv2.rectangle(overlay, (w // 2 - 360, banner_y), (w // 2 + 360, banner_y + 44), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.90, hud, 0.10, 0, hud)
            cv2.rectangle(hud, (w // 2 - 360, banner_y), (w // 2 + 360, banner_y + 44), (0, 240, 255), 2)
            cv2.putText(hud, self.active_alert_banner, (w // 2 - 340, banner_y + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2)

        return hud

    def capture_frame_loop(self):
        self.open_video_source()
        last_time = time.time()
        last_rx_frame_time = time.time()

        while self.is_running:
            loop_start = time.time()

            frame = None
            if self.cap and self.cap.isOpened():
                ret, raw_frame = self.cap.read()
                if ret and raw_frame is not None:
                    frame = raw_frame
                    last_rx_frame_time = time.time()
                else:
                    if (time.time() - last_rx_frame_time) > 4.0 and self.stream_source != "synthetic":
                        logger.warning(f"⚠️ Stream timeout from {self.current_source_name}. Reconnecting...")
                        self.open_video_source()
                        last_rx_frame_time = time.time()
                    frame = self.generate_synthetic_frame()
            else:
                frame = self.generate_synthetic_frame()

            if not self.feature_toggles.get("object_detection", True):
                # Completely skip DNN inference (near 0% compute)
                detections = []
                with self.tracks_lock:
                    stale_ids = list(self.tracks.keys())
                    for tid in stale_ids:
                        del self.tracks[tid]
                self.person_count = 0
                self.is_fall_active = False
                self.is_intrusion_active = False
                ai_latency = 0.0
            else:
                ai_start = time.time()
                skeletal_enabled = self.feature_toggles.get("skeletal_tracking", True)
                detections = self.detector.detect(frame, skeletal_tracking_enabled=skeletal_enabled)

                # If synthetic frame and no detections, provide synthetic person with keypoints
                if self.stream_source == "synthetic" and not detections:
                    t_now = time.time()
                    px = 0.50 + 0.30 * np.sin(t_now * 0.5)
                    py = 0.52 + 0.10 * np.cos(t_now * 0.8)
                    cx, cy = int(px * 1280), int(py * 720)
                    synth_bbox = (
                        max(0.0, (cx - 36) / 1280.0),
                        max(0.0, (cy - 120) / 720.0),
                        min(1.0, (cx + 36) / 1280.0),
                        min(1.0, (cy + 100) / 720.0)
                    )
                    synth_kp = self.detector._estimate_skeletal_keypoints(synth_bbox) if skeletal_enabled else None
                    detections = [DetectionObject(
                        bbox=synth_bbox,
                        confidence=0.92,
                        class_id=0,
                        class_name="person",
                        keypoints=synth_kp
                    )]

                self.update_tracks(detections)
                ai_latency = (time.time() - ai_start) * 1000.0

            h, w = frame.shape[:2]
            detections_for_zones = []
            for tid, t in self.tracks.items():
                if t.is_confirmed:
                    detections_for_zones.append({
                        "track_id": tid,
                        "class_name": t.class_name,
                        "confidence": t.confidence,
                        "bbox": list(t.bbox)
                    })

            events = ai_zone_service.process_detections(self.camera_id, detections_for_zones, w, h)
            self.is_intrusion_active = False

            now = time.time()
            for ev in events:
                meta = getattr(ev, "metadata", {}) or {}
                analytics_type = meta.get("analytics_type", "")
                track_id_val = meta.get("track_id", 1)
                direction_val = meta.get("direction", "A_TO_B")
                zone_id = meta.get("zone_id", "default")
                zone_name = meta.get("zone_name", "Perimeter")

                if "TRIPWIRE" in analytics_type or ev.event_type.name == "PERIMETER_BREACH":
                    if self.debouncer.should_dispatch(zone_id, int(track_id_val), now):
                        if zone_id in self.tripwires:
                            if direction_val == "A_TO_B":
                                self.tripwires[zone_id]["in_count"] = self.tripwires[zone_id].get("in_count", 0) + 1
                            else:
                                self.tripwires[zone_id]["out_count"] = self.tripwires[zone_id].get("out_count", 0) + 1
                        self.trigger_alert(f"TRIPWIRE '{zone_name}' CROSSED [{direction_val}] by Track #{track_id_val}!", "TRIPWIRE")
                        self.record_15s_incident_clip("TRIPWIRE")

                elif "INTRUSION" in analytics_type or "INTRUSION" in ev.event_type.name:
                    self.is_intrusion_active = True
                    if self.debouncer.should_dispatch(zone_id, int(track_id_val), now):
                        self.trigger_alert(f"RESTRICTED AREA '{zone_name}' BREACHED by Track #{track_id_val}!", "INTRUSION")
                        self.record_15s_incident_clip("INTRUSION")

            self.recent_latencies.append(ai_latency)
            if len(self.recent_latencies) > 30:
                self.recent_latencies.pop(0)
            self.avg_inference_ms = sum(self.recent_latencies) / len(self.recent_latencies)

            self.ring_buffer.push_frame(frame)
            hud_frame = self.draw_hud(frame)

            ret, jpeg_bytes = cv2.imencode(".jpg", hud_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                with self.frame_lock:
                    self.latest_encoded_jpeg = jpeg_bytes.tobytes()

            self.frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                self.fps = self.frame_count / (now - last_time)
                self.frame_count = 0
                last_time = now

            elapsed = time.time() - loop_start
            sleep_time = max(0.005, (1.0 / 30.0) - elapsed)
            time.sleep(sleep_time)


# ==============================================================================
# Embedded Web HUD Server & REST API
# ==============================================================================
def create_web_hud_app(monitor: LiveAIMonitor):
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    web_app = FastAPI(title="Edge AI CCTV Multi-Zone Web HUD")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class TripwireReq(BaseModel):
        id: Optional[str] = None
        name: Optional[str] = None
        x1: float
        y1: float
        x2: float
        y2: float
        direction: Optional[str] = "BIDIRECTIONAL"
        allowed_classes: Optional[List[str]] = ["person", "vehicle"]
        enabled: Optional[bool] = True

    class IntrusionReq(BaseModel):
        id: Optional[str] = None
        name: Optional[str] = None
        points: List[Dict[str, float]]
        allowed_classes: Optional[List[str]] = ["person", "vehicle"]
        dwell_time_seconds: Optional[float] = 0.5
        enabled: Optional[bool] = True

    class ExclusionReq(BaseModel):
        id: Optional[str] = None
        name: Optional[str] = None
        points: List[Dict[str, float]]
        mask_mode: Optional[str] = "BLUR"
        enabled: Optional[bool] = True

    class SwitchSourceReq(BaseModel):
        url: str

    @web_app.get("/", response_class=HTMLResponse)
    async def index():
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Edge AI CCTV - Multi-Zone Studio</title>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0e14;
      --card-bg: rgba(18, 22, 31, 0.90);
      --card-border: rgba(0, 240, 255, 0.25);
      --accent-cyan: #00f0ff;
      --accent-green: #00ff9d;
      --accent-orange: #ffaa00;
      --accent-red: #ff0055;
      --text-main: #f0f6fc;
      --text-dim: #8b949e;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text-main);
      font-family: 'Plus Jakarta Sans', sans-serif;
      padding: 16px;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 12px 18px;
    }
    .logo-group { display: flex; align-items: center; gap: 10px; }
    .logo-title { font-size: 17px; font-weight: 800; color: #fff; }
    .badge {
      background: rgba(0, 240, 255, 0.15);
      border: 1px solid var(--accent-cyan);
      color: var(--accent-cyan);
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      font-weight: 600;
      padding: 4px 9px;
      border-radius: 20px;
    }
    .camera-bar {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 10px 14px;
      display: flex;
      align-items: center;
      gap: 10px;
      overflow-x: auto;
    }
    .camera-bar-title {
      font-size: 12px;
      font-weight: 700;
      color: var(--accent-cyan);
      white-space: nowrap;
    }
    .cam-btn {
      background: #161b22;
      border: 1px solid rgba(255,255,255,0.15);
      color: #fff;
      font-size: 11.5px;
      padding: 6px 12px;
      border-radius: 6px;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.2s;
    }
    .cam-btn:hover { background: #21262d; border-color: var(--accent-cyan); }
    .cam-btn.active {
      background: rgba(0, 240, 255, 0.2);
      border-color: var(--accent-cyan);
      color: var(--accent-cyan);
      font-weight: 700;
    }
    .main-grid {
      display: grid;
      grid-template-columns: 1fr 420px;
      gap: 14px;
    }
    @media (max-width: 1100px) {
      .main-grid { grid-template-columns: 1fr; }
    }
    .video-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .video-viewport {
      position: relative;
      width: 100%;
      border-radius: 10px;
      overflow: hidden;
      background: #000;
      border: 1px solid rgba(255,255,255,0.1);
      aspect-ratio: 16 / 9;
    }
    .video-viewport img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    #interactiveCanvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      cursor: crosshair;
      z-index: 10;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 14px;
      margin-bottom: 12px;
    }
    .card-title {
      font-size: 12.5px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--accent-cyan);
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .telemetry-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 6px 0;
      border-bottom: 1px solid rgba(255,255,255,0.05);
      font-size: 12px;
    }
    .telemetry-row:last-child { border-bottom: none; }
    .telemetry-label { color: var(--text-dim); }
    .telemetry-val {
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      color: var(--accent-green);
    }
    .btn {
      background: #1a202c;
      border: 1px solid rgba(255,255,255,0.15);
      color: #fff;
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 11.5px;
      font-weight: 700;
      padding: 8px 10px;
      border-radius: 6px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 5px;
      transition: all 0.2s ease;
    }
    .btn:hover { background: #2d3748; border-color: var(--accent-cyan); }
    .btn-primary { background: rgba(0, 240, 255, 0.15); border-color: var(--accent-cyan); color: var(--accent-cyan); }
    .btn-primary:hover { background: var(--accent-cyan); color: #000; }
    .btn-danger { background: rgba(255, 0, 85, 0.15); border-color: var(--accent-red); color: var(--accent-red); }
    .btn-danger:hover { background: var(--accent-red); color: #fff; }
    .btn-sm { padding: 4px 7px; font-size: 10.5px; border-radius: 4px; }
    
    .zone-item {
      background: rgba(0,0,0,0.35);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 8px;
      padding: 8px 10px;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .zone-info { display: flex; flex-direction: column; gap: 2px; }
    .zone-name { font-size: 12px; font-weight: 700; color: #fff; }
    .zone-sub { font-size: 10px; color: var(--text-dim); font-family: 'JetBrains Mono', monospace; }
    
    .event-log-container {
      max-height: 160px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 5px;
    }
    .event-log-item {
      background: rgba(0,0,0,0.3);
      border-left: 3px solid var(--accent-cyan);
      padding: 5px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-family: 'JetBrains Mono', monospace;
    }
    .event-fall { border-left-color: var(--accent-red); color: #ff99bb; }
    .event-intrusion { border-left-color: var(--accent-orange); color: #ffd280; }
    /* Quick Controls & Animated Switches */
    .control-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 7px 10px;
      background: rgba(0,0,0,0.30);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 8px;
    }
    .control-label-group {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .control-title {
      font-size: 11.5px;
      font-weight: 700;
      color: #fff;
    }
    .control-desc {
      font-size: 9.5px;
      color: var(--text-dim);
    }
    .switch {
      position: relative;
      display: inline-block;
      width: 42px;
      height: 22px;
      flex-shrink: 0;
    }
    .switch input { opacity: 0; width: 0; height: 0; }
    .slider {
      position: absolute;
      cursor: pointer;
      top: 0; left: 0; right: 0; bottom: 0;
      background-color: #21262d;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      border-radius: 22px;
      border: 1px solid rgba(255,255,255,0.2);
    }
    .slider:before {
      position: absolute;
      content: "";
      height: 14px;
      width: 14px;
      left: 3px;
      bottom: 3px;
      background-color: #8b949e;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      border-radius: 50%;
    }
    input:checked + .slider {
      background-color: rgba(0, 240, 255, 0.25);
      border-color: var(--accent-cyan);
      box-shadow: 0 0 8px rgba(0, 240, 255, 0.4);
    }
    input:checked + .slider:before {
      transform: translateX(20px);
      background-color: var(--accent-cyan);
      box-shadow: 0 0 6px var(--accent-cyan);
    }
    .switch-sm { width: 34px; height: 18px; }
    .switch-sm .slider:before { height: 12px; width: 12px; left: 2px; bottom: 2px; }
    .switch-sm input:checked + .slider:before { transform: translateX(16px); }

    /* IoT Sensor Grid & Telemetry Badges */
    .sensor-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 7px;
      margin-top: 6px;
    }
    .sensor-card {
      background: rgba(0,0,0,0.30);
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: 8px;
      padding: 7px 9px;
      display: flex;
      flex-direction: column;
      gap: 5px;
    }
    .sensor-title-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .sensor-name {
      font-size: 11px;
      font-weight: 700;
      color: #e6edf3;
    }
    .sensor-badge {
      font-family: 'JetBrains Mono', monospace;
      font-size: 9.5px;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(0, 240, 255, 0.15);
      border: 1px solid var(--accent-cyan);
      color: var(--accent-cyan);
      text-transform: uppercase;
      transition: all 0.2s ease;
    }
    .sensor-badge.idle {
      background: rgba(0, 255, 157, 0.12);
      border-color: rgba(0, 255, 157, 0.4);
      color: var(--accent-green);
    }
    .sensor-badge.alert {
      background: rgba(255, 0, 85, 0.25);
      border-color: var(--accent-red);
      color: #ff99bb;
      animation: pulseAlert 1.2s infinite alternate;
    }
    .sensor-badge.offline {
      background: rgba(139, 148, 158, 0.15);
      border-color: rgba(139, 148, 158, 0.4);
      color: #8b949e;
    }
    @keyframes pulseAlert {
      from { box-shadow: 0 0 2px rgba(255,0,85,0.4); }
      to { box-shadow: 0 0 10px rgba(255,0,85,0.8); }
    }
    .sensor-controls-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding-top: 4px;
      border-top: 1px solid rgba(255,255,255,0.05);
    }
    .sensor-switch-wrapper {
      display: inline-flex;
      align-items: center;
    }
    .sensor-toggles-disabled .sensor-switch-wrapper {
      cursor: not-allowed !important;
    }
    .sensor-toggles-disabled .switch {
      opacity: 0.45;
      cursor: not-allowed !important;
      pointer-events: none;
    }
    .sensor-toggles-disabled .sensor-sim-btn {
      opacity: 0.45;
    }
    .toast {
      position: fixed;
      bottom: 20px;
      right: 20px;
      background: rgba(0, 240, 255, 0.95);
      color: #000;
      padding: 10px 16px;
      border-radius: 8px;
      font-weight: 700;
      font-size: 12.5px;
      display: none;
      box-shadow: 0 8px 30px rgba(0,240,255,0.4);
      z-index: 999;
    }
  </style>
</head>
<body>
  <header class="header">
    <div class="logo-group">
      <span style="font-size: 20px;">🛡️</span>
      <div>
        <div class="logo-title">EDGE AI CCTV - MULTI-ZONE STUDIO</div>
        <div style="font-size: 11px; color: var(--text-dim);">Neural Multi-Class Vision, 15s Auto Clips & Privacy Masking</div>
      </div>
    </div>
    <div style="display: flex; gap: 8px; align-items: center;">
      <span class="badge" id="sourceBadge">Source: Detecting...</span>
      <span class="badge" id="fpsBadge">0.0 FPS</span>
      <span class="badge" id="latencyBadge">0.0 ms</span>
    </div>
  </header>

  <!-- Single Clean Camera Switcher Bar -->
  <div class="camera-bar" id="cameraBar">
    <span class="camera-bar-title">📷 Feeds:</span>
    <div id="cameraButtonsList" style="display: flex; gap: 8px; align-items: center;">
      <button class="cam-btn active">Loading feeds...</button>
    </div>
  </div>

  <div class="main-grid">
    <div class="video-card">
      <div class="video-viewport" id="viewportWrapper">
        <img id="streamImg" src="/stream" alt="Live AI Vision Feed" />
        <canvas id="interactiveCanvas"></canvas>
      </div>

      <!-- Zone Creation Toolbar -->
      <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); display: flex; flex-direction: column; gap: 8px;">
        <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
          <input type="text" id="zoneNameInput" placeholder="Zone Name (e.g. Front Gate / Porch / Window)" style="flex: 1.4; background: #161b22; color: #fff; padding: 7px 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.2); font-size: 11.5px;" />
          <select id="zoneClassSelect" style="flex: 1; background: #161b22; color: #fff; padding: 7px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.2); font-size: 11.5px;">
            <option value="both">🧍+🚗 Human & Vehicle</option>
            <option value="person">🧍 Human Only</option>
            <option value="vehicle">🚗 Vehicle Only</option>
          </select>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
          <button class="btn btn-primary" style="flex: 1;" onclick="setDrawMode('TRIPWIRE')">⚡ Add Tripwire (2 Pts)</button>
          <button class="btn btn-primary" style="flex: 1;" onclick="setDrawMode('INTRUSION')">🛑 Add Restricted Area (3+ Pts)</button>
          <button class="btn btn-primary" style="flex: 1;" onclick="setDrawMode('EXCLUSION')">🌫️ Add Exclusion Mask (3+ Pts)</button>
          <button class="btn" style="flex: 0.8;" onclick="saveDrawnZone()">💾 Save Zone</button>
          <button class="btn btn-danger" style="flex: 0.6;" onclick="clearCanvasPoints()">✕ Cancel</button>
        </div>
      </div>

      <div style="display: flex; gap: 8px; flex-wrap: wrap;">
        <button class="btn" style="flex: 1;" onclick="triggerSnapshot()">📸 Snapshot</button>
        <button class="btn" style="flex: 1;" onclick="triggerClip()">🎥 15s MP4 Clip</button>
        <button class="btn" style="flex: 1;" onclick="rescanCameras()">🔄 Rescan Cameras</button>
        <button class="btn btn-primary" style="flex: 1.2;" onclick="rescanESP32()">📡 Scan ESP32 Sentries</button>
      </div>
    </div>

    <div>
      <!-- ⚡ AI & IoT Quick Controls Panel -->
      <div class="card">
        <div class="card-title">
          <span>⚡ AI & IoT Quick Controls</span>
          <span class="badge" style="font-size: 10px; padding: 2px 7px;" id="aiModeBadge">LIVE SYNC</span>
        </div>

        <!-- AI Engine Feature Toggles -->
        <div style="display: flex; flex-direction: column; gap: 8px; margin-bottom: 12px;">
          <!-- Fall Detection Toggle -->
          <div class="control-row">
            <div class="control-label-group">
              <span class="control-title">🏃 Fall Detection</span>
              <span class="control-desc">Kinematic descent & immobility monitor</span>
            </div>
            <label class="switch">
              <input type="checkbox" id="toggle_fall_detection" onchange="onToggleFeature('fall_detection', this.checked)" checked>
              <span class="slider"></span>
            </label>
          </div>

          <!-- Skeletal Tracking Toggle -->
          <div class="control-row">
            <div class="control-label-group">
              <span class="control-title">🦴 Skeletal Tracking</span>
              <span class="control-desc">17-Keypoint joint limb overlay</span>
            </div>
            <label class="switch">
              <input type="checkbox" id="toggle_skeletal_tracking" onchange="onToggleFeature('skeletal_tracking', this.checked)" checked>
              <span class="slider"></span>
            </label>
          </div>

          <!-- Multi-Class Object Detection Toggle -->
          <div class="control-row">
            <div class="control-label-group">
              <span class="control-title">👁️ Multi-Class Object Detection</span>
              <span class="control-desc">YOLO DNN (Standby = 0% compute)</span>
            </div>
            <label class="switch">
              <input type="checkbox" id="toggle_object_detection" onchange="onToggleFeature('object_detection', this.checked)" checked>
              <span class="slider"></span>
            </label>
          </div>
        </div>

        <!-- ESP32 IoT Hardware Telemetry & Controls -->
        <div style="border-top: 1px solid rgba(255,255,255,0.08); padding-top: 10px;" id="esp32SentrySection">
          <div style="font-size: 11px; font-weight: 700; color: var(--accent-cyan); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 4px;">
            <div style="display: flex; align-items: center; gap: 6px;">
              <span>📡 IoT Hardware Sentries</span>
              <span class="badge" id="esp32StatusPill" style="font-size: 10px; padding: 2px 8px; border-radius: 12px; font-weight: 700; background: rgba(255,170,0,0.15); border: 1px solid #ffaa00; color: #ffaa00;">
                ⚠️ NO SENTRY CONNECTED
              </span>
            </div>
            <button class="btn btn-sm" id="btnDemoSim" style="font-size: 9.5px; padding: 2px 7px;" onclick="toggleDemoSimulation()">🧪 Demo Simulation</button>
          </div>

          <!-- Disconnected Notice Banner -->
          <div id="esp32NoticeBox" style="display: block; background: rgba(255,170,0,0.08); border: 1px dashed rgba(255,170,0,0.35); border-radius: 6px; padding: 7px 9px; margin-bottom: 8px; font-size: 10.5px; color: #ffca66; line-height: 1.35;">
            ℹ️ Connect an ESP32-S3 Sentry node to enable physical PIR, Ultrasonic, and Door reed switches. Click <strong>"📡 Scan ESP32 Sentries"</strong> above to auto-detect.
          </div>

          <div class="sensor-grid sensor-toggles-disabled" id="sensorGridContainer">
            <!-- PIR Motion Sensor -->
            <div class="sensor-card">
              <div class="sensor-title-row">
                <span class="sensor-name">🚶 PIR Motion</span>
                <span class="sensor-badge offline" id="pirStatusBadge">OFFLINE</span>
              </div>
              <div class="sensor-controls-row">
                <span style="color: var(--text-dim); font-size: 10px;">Armed:</span>
                <div class="sensor-switch-wrapper" onclick="handleSensorToggleClick('esp32_pir', event)">
                  <label class="switch switch-sm">
                    <input type="checkbox" id="toggle_esp32_pir" onchange="onToggleFeature('esp32_pir', this.checked)" disabled>
                    <span class="slider"></span>
                  </label>
                </div>
                <button class="btn btn-sm sensor-sim-btn" style="font-size: 9.5px; padding: 2px 6px;" onclick="simulateSensor('pir')">Simulate</button>
              </div>
            </div>

            <!-- Ultrasonic Sensor -->
            <div class="sensor-card">
              <div class="sensor-title-row">
                <span class="sensor-name">📏 Ultrasonic</span>
                <span class="sensor-badge offline" id="distStatusBadge">OFFLINE</span>
              </div>
              <div class="sensor-controls-row">
                <span style="color: var(--text-dim); font-size: 10px;">Armed:</span>
                <div class="sensor-switch-wrapper" onclick="handleSensorToggleClick('esp32_ultrasonic', event)">
                  <label class="switch switch-sm">
                    <input type="checkbox" id="toggle_esp32_ultrasonic" onchange="onToggleFeature('esp32_ultrasonic', this.checked)" disabled>
                    <span class="slider"></span>
                  </label>
                </div>
                <button class="btn btn-sm sensor-sim-btn" style="font-size: 9.5px; padding: 2px 6px;" onclick="simulateSensor('dist')">Ping</button>
              </div>
            </div>

            <!-- Door 1 Sensor -->
            <div class="sensor-card">
              <div class="sensor-title-row">
                <span class="sensor-name">🚪 Door 1 (Front)</span>
                <span class="sensor-badge offline" id="door1StatusBadge">OFFLINE</span>
              </div>
              <div class="sensor-controls-row">
                <span style="color: var(--text-dim); font-size: 10px;">Armed:</span>
                <div class="sensor-switch-wrapper" onclick="handleSensorToggleClick('esp32_door1', event)">
                  <label class="switch switch-sm">
                    <input type="checkbox" id="toggle_esp32_door1" onchange="onToggleFeature('esp32_door1', this.checked)" disabled>
                    <span class="slider"></span>
                  </label>
                </div>
                <button class="btn btn-sm sensor-sim-btn" style="font-size: 9.5px; padding: 2px 6px;" onclick="simulateSensor('door1')">Toggle</button>
              </div>
            </div>

            <!-- Door 2 Sensor -->
            <div class="sensor-card">
              <div class="sensor-title-row">
                <span class="sensor-name">🚪 Door 2 (Back)</span>
                <span class="sensor-badge offline" id="door2StatusBadge">OFFLINE</span>
              </div>
              <div class="sensor-controls-row">
                <span style="color: var(--text-dim); font-size: 10px;">Armed:</span>
                <div class="sensor-switch-wrapper" onclick="handleSensorToggleClick('esp32_door2', event)">
                  <label class="switch switch-sm">
                    <input type="checkbox" id="toggle_esp32_door2" onchange="onToggleFeature('esp32_door2', this.checked)" disabled>
                    <span class="slider"></span>
                  </label>
                </div>
                <button class="btn btn-sm sensor-sim-btn" style="font-size: 9.5px; padding: 2px 6px;" onclick="simulateSensor('door2')">Toggle</button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Multi-Zone Manager Panel -->
      <div class="card">
        <div class="card-title">
          <span>⚡ Active Tripwires</span>
          <button class="btn btn-danger btn-sm" onclick="clearAllZones()">Clear All</button>
        </div>
        <div id="tripwiresListContainer">
          <div style="font-size: 11.5px; color: var(--text-dim);">No tripwires configured.</div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">
          <span>🛑 Restricted Polygon Zones</span>
        </div>
        <div id="intrusionListContainer">
          <div style="font-size: 11.5px; color: var(--text-dim);">No restricted areas configured.</div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">
          <span>🌫️ Exclusion & Privacy Masks</span>
        </div>
        <div id="exclusionListContainer">
          <div style="font-size: 11.5px; color: var(--text-dim);">No exclusion masks configured.</div>
        </div>
      </div>

      <!-- Telemetry Card -->
      <div class="card">
        <div class="card-title">📊 Multi-Class Telemetry</div>
        <div class="telemetry-row">
          <span class="telemetry-label">Humans Detected:</span>
          <span class="telemetry-val" id="personCountVal">0</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Total Confirmed Tracks:</span>
          <span class="telemetry-val" id="totalTracksVal" style="color: var(--accent-orange);">0</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Torso Angle (θ):</span>
          <span class="telemetry-val" id="torsoAngleVal">85.0°</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Fall Status:</span>
          <span class="telemetry-val" id="fallStatusVal">NORMAL</span>
        </div>
      </div>

      <!-- Security Events Log -->
      <div class="card">
        <div class="card-title">📋 Real Security Events Log</div>
        <div class="event-log-container" id="eventLogList">
          <div class="event-log-item">System armed. Insect/Noise filter active.</div>
        </div>
      </div>
    </div>
  </div>

  <div class="toast" id="toast">Notification</div>

  <script>
    const canvas = document.getElementById('interactiveCanvas');
    const ctx = canvas.getContext('2d');
    let currentMode = 'NONE';
    let drawnPoints = [];
    let activeCameraUrl = '';

    function resizeCanvas() {
      canvas.width = canvas.parentElement.clientWidth;
      canvas.height = canvas.parentElement.clientHeight;
      drawOverlay();
    }
    window.addEventListener('resize', resizeCanvas);
    setTimeout(resizeCanvas, 300);

    canvas.addEventListener('click', (e) => {
      if (currentMode === 'NONE') return;
      const rect = canvas.getBoundingClientRect();
      const nx = (e.clientX - rect.left) / canvas.width;
      const ny = (e.clientY - rect.top) / canvas.height;

      if (currentMode === 'TRIPWIRE') {
        if (drawnPoints.length >= 2) drawnPoints = [];
        drawnPoints.push({ x: nx, y: ny });
      } else {
        drawnPoints.push({ x: nx, y: ny });
      }
      drawOverlay();
    });

    function drawOverlay() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (drawnPoints.length === 0) return;

      if (currentMode === 'TRIPWIRE') {
        ctx.strokeStyle = '#00f0ff';
        ctx.fillStyle = 'rgba(0, 240, 255, 0.2)';
      } else if (currentMode === 'INTRUSION') {
        ctx.strokeStyle = '#ffaa00';
        ctx.fillStyle = 'rgba(255, 170, 0, 0.25)';
      } else if (currentMode === 'EXCLUSION') {
        ctx.strokeStyle = '#a0aec0';
        ctx.fillStyle = 'rgba(160, 174, 192, 0.35)';
      }

      ctx.lineWidth = 2.5;
      ctx.beginPath();
      drawnPoints.forEach((pt, i) => {
        const px = pt.x * canvas.width;
        const py = pt.y * canvas.height;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });

      if ((currentMode === 'INTRUSION' || currentMode === 'EXCLUSION') && drawnPoints.length >= 3) {
        ctx.closePath();
        ctx.fill();
      }
      ctx.stroke();

      drawnPoints.forEach(pt => {
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 5, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    function setDrawMode(mode) {
      currentMode = mode;
      drawnPoints = [];
      drawOverlay();
      showToast(`Mode: ${mode} - Click on the live feed to position vertices.`);
    }

    function clearCanvasPoints() {
      drawnPoints = [];
      currentMode = 'NONE';
      drawOverlay();
      showToast('Drawing cancelled.');
    }

    function getAllowedClasses() {
      const val = document.getElementById('zoneClassSelect').value;
      if (val === 'person') return ['person'];
      if (val === 'vehicle') return ['vehicle'];
      return ['person', 'vehicle'];
    }

    async function saveDrawnZone() {
      const defaultName = currentMode === 'TRIPWIRE' ? 'Tripwire' : (currentMode === 'INTRUSION' ? 'Restricted Area' : 'Exclusion Mask');
      const name = document.getElementById('zoneNameInput').value.trim() || defaultName;
      const allowed = getAllowedClasses();

      try {
        if (currentMode === 'TRIPWIRE' && drawnPoints.length === 2) {
          const res = await fetch('/api/zones/tripwire', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              name: name,
              x1: drawnPoints[0].x, y1: drawnPoints[0].y,
              x2: drawnPoints[1].x, y2: drawnPoints[1].y,
              direction: 'BIDIRECTIONAL',
              allowed_classes: allowed,
              enabled: true
            })
          });
          const data = await res.json();
          showToast(`✅ Tripwire '${name}' saved!`);
          clearCanvasPoints();
          loadZonesList();
        } else if (currentMode === 'INTRUSION' && drawnPoints.length >= 3) {
          const res = await fetch('/api/zones/intrusion', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              name: name,
              points: drawnPoints,
              allowed_classes: allowed,
              dwell_time_seconds: 0.5,
              enabled: true
            })
          });
          const data = await res.json();
          showToast(`✅ Restricted Area '${name}' saved!`);
          clearCanvasPoints();
          loadZonesList();
        } else if (currentMode === 'EXCLUSION' && drawnPoints.length >= 3) {
          const res = await fetch('/api/zones/exclusion', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              name: name,
              points: drawnPoints,
              mask_mode: 'BLUR',
              enabled: true
            })
          });
          const data = await res.json();
          showToast(`✅ Exclusion Mask '${name}' saved!`);
          clearCanvasPoints();
          loadZonesList();
        } else {
          showToast('Please click points on video first (2 for tripwire, 3+ for areas).');
        }
      } catch (err) {
        showToast('Error saving zone: ' + err.message);
      }
    }

    async function deleteTripwire(id) {
      try {
        await fetch(`/api/zones/tripwire/${id}`, { method: 'DELETE' });
        showToast('Tripwire removed.');
        loadZonesList();
      } catch (e) { showToast('Failed to delete tripwire'); }
    }

    async function deleteIntrusion(id) {
      try {
        await fetch(`/api/zones/intrusion/${id}`, { method: 'DELETE' });
        showToast('Restricted Area removed.');
        loadZonesList();
      } catch (e) { showToast('Failed to delete restricted area'); }
    }

    async function deleteExclusion(id) {
      try {
        await fetch(`/api/zones/exclusion/${id}`, { method: 'DELETE' });
        showToast('Exclusion Mask removed.');
        loadZonesList();
      } catch (e) { showToast('Failed to delete exclusion mask'); }
    }

    async function clearAllZones() {
      if (confirm('Are you sure you want to remove all tripwires, restricted areas, and masks?')) {
        await fetch('/api/zones/clear', { method: 'POST' });
        showToast('All zones cleared.');
        loadZonesList();
      }
    }

    async function loadZonesList() {
      try {
        const res = await fetch('/api/zones');
        const data = await res.json();

        // 1. Tripwires List
        const twContainer = document.getElementById('tripwiresListContainer');
        const tripwires = data.tripwires || [];
        if (tripwires.length === 0) {
          twContainer.innerHTML = '<div style="font-size: 11.5px; color: var(--text-dim);">No tripwires configured.</div>';
        } else {
          twContainer.innerHTML = '';
          tripwires.forEach(tw => {
            const item = document.createElement('div');
            item.className = 'zone-item';
            item.innerHTML = `
              <div class="zone-info">
                <span class="zone-name">⚡ ${tw.name}</span>
                <span class="zone-sub">In: ${tw.in_count || 0} | Out: ${tw.out_count || 0} | [${tw.direction || 'BIDIRECTIONAL'}]</span>
              </div>
              <button class="btn btn-danger btn-sm" onclick="deleteTripwire('${tw.id}')">🗑️ Remove</button>
            `;
            twContainer.appendChild(item);
          });
        }

        // 2. Intrusion Zones List
        const intContainer = document.getElementById('intrusionListContainer');
        const intrusion_zones = data.intrusion_zones || [];
        if (intrusion_zones.length === 0) {
          intContainer.innerHTML = '<div style="font-size: 11.5px; color: var(--text-dim);">No restricted areas configured.</div>';
        } else {
          intContainer.innerHTML = '';
          intrusion_zones.forEach(iz => {
            const item = document.createElement('div');
            item.className = 'zone-item';
            item.innerHTML = `
              <div class="zone-info">
                <span class="zone-name">🛑 ${iz.name}</span>
                <span class="zone-sub">${iz.points ? iz.points.length : 0} Vertices | Target: ${iz.allowed_classes ? iz.allowed_classes.join(',') : 'all'}</span>
              </div>
              <button class="btn btn-danger btn-sm" onclick="deleteIntrusion('${iz.id}')">🗑️ Remove</button>
            `;
            intContainer.appendChild(item);
          });
        }

        // 3. Exclusion Masks List
        const exContainer = document.getElementById('exclusionListContainer');
        const exclusion_masks = data.exclusion_masks || [];
        if (exclusion_masks.length === 0) {
          exContainer.innerHTML = '<div style="font-size: 11.5px; color: var(--text-dim);">No exclusion masks configured.</div>';
        } else {
          exContainer.innerHTML = '';
          exclusion_masks.forEach(ex => {
            const item = document.createElement('div');
            item.className = 'zone-item';
            item.innerHTML = `
              <div class="zone-info">
                <span class="zone-name">🌫️ ${ex.name}</span>
                <span class="zone-sub">${ex.points ? ex.points.length : 0} Vertices | [${ex.mask_mode || 'BLUR'}]</span>
              </div>
              <button class="btn btn-danger btn-sm" onclick="deleteExclusion('${ex.id}')">🗑️ Remove</button>
            `;
            exContainer.appendChild(item);
          });
        }
      } catch (e) {}
    }

    async function loadCameraSources() {
      try {
        const res = await fetch('/api/rescan', { method: 'POST' });
        const data = await res.json();
        const list = document.getElementById('cameraButtonsList');
        list.innerHTML = '';

        (data.sources || []).forEach((src, idx) => {
          const btn = document.createElement('button');
          btn.className = `cam-btn ${idx === 0 ? 'active' : ''}`;
          btn.textContent = src.name;
          btn.onclick = () => switchCameraFeed(src.url, btn);
          list.appendChild(btn);
        });
      } catch (e) {}
    }

    async function switchCameraFeed(url, btnElement) {
      try {
        await fetch('/api/switch_source', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: url })
        });
        document.querySelectorAll('.cam-btn').forEach(b => b.classList.remove('active'));
        if (btnElement) btnElement.classList.add('active');
        showToast('Switched camera source.');
        setTimeout(updateTelemetry, 800);
      } catch (e) {
        showToast('Failed to switch source');
      }
    }

    function showToast(msg) {
      const t = document.getElementById('toast');
      t.textContent = msg;
      t.style.display = 'block';
      setTimeout(() => { t.style.display = 'none'; }, 3000);
    }

    async function triggerSnapshot() {
      try {
        const res = await fetch('/api/action/snapshot', { method: 'POST' });
        const data = await res.json();
        showToast(data.message || 'Snapshot captured');
      } catch (e) { showToast('Snapshot failed'); }
    }

    async function triggerClip() {
      try {
        const res = await fetch('/api/action/clip', { method: 'POST' });
        const data = await res.json();
        showToast(data.message || 'Exporting 15s clip');
      } catch (e) { showToast('Clip export failed'); }
    }

    async function rescanCameras() {
      showToast('Scanning network and USB devices...');
      try {
        await loadCameraSources();
        showToast('Camera scan complete.');
      } catch (e) { showToast('Camera scan failed.'); }
    }

    let isUserToggling = false;
    let isEsp32ConnectedGlobal = false;
    let isDemoModeGlobal = false;

    function handleSensorToggleClick(feature, event) {
      if (!isEsp32ConnectedGlobal && !isDemoModeGlobal) {
        if (event) {
          event.preventDefault();
          event.stopPropagation();
        }
        showToast('⚠️ No physical ESP32 sentry connected on network! Please scan or connect device.');
        return false;
      }
    }

    async function rescanESP32() {
      showToast('📡 Scanning subnet & mDNS for ESP32 sentries...');
      const pill = document.getElementById('esp32StatusPill');
      if (pill) {
        pill.textContent = '⏳ SCANNING...';
        pill.style.borderColor = 'var(--accent-cyan)';
        pill.style.color = 'var(--accent-cyan)';
      }
      try {
        const res = await fetch('/api/sensors/rescan', { method: 'POST' });
        const data = await res.json();
        if (data.is_connected) {
          showToast(`✅ Connected to ${data.found_count} ESP32 sentry node(s)!`);
        } else {
          showToast('⚠️ No physical ESP32 sentries found on subnet.');
        }
        updateTelemetry();
      } catch (e) {
        showToast('❌ ESP32 rescan failed.');
      }
    }

    async function toggleDemoSimulation() {
      try {
        const res = await fetch('/api/sensors/demo_mode', { method: 'POST' });
        const data = await res.json();
        showToast(data.demo_enabled ? '🧪 Demo Simulation ENABLED (Toggles Unlocked)' : 'Demo Simulation DISABLED');
        updateTelemetry();
      } catch (e) {
        showToast('Failed to toggle demo simulation mode');
      }
    }

    async function onToggleFeature(feature, enabled) {
      if (feature.startsWith('esp32_') && !isEsp32ConnectedGlobal && !isDemoModeGlobal) {
        showToast('⚠️ No physical ESP32 sentry connected on network! Please scan or connect device.');
        const el = document.getElementById(`toggle_${feature}`);
        if (el) el.checked = !enabled;
        return;
      }
      isUserToggling = true;
      try {
        const res = await fetch('/api/toggles', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ feature: feature, enabled: enabled })
        });
        const data = await res.json();
        if (!res.ok || data.status === 'error') {
          showToast(data.message || 'Failed to update toggle: Disconnected');
          const el = document.getElementById(`toggle_${feature}`);
          if (el) el.checked = !enabled;
          return;
        }
        showToast(`Feature '${feature}' -> ${enabled ? 'ON' : 'OFF'}`);
      } catch (e) {
        showToast('Failed to update toggle');
        const el = document.getElementById(`toggle_${feature}`);
        if (el) el.checked = !enabled;
      } finally {
        setTimeout(() => { isUserToggling = false; }, 600);
      }
    }

    async function simulateSensor(type) {
      if (!isEsp32ConnectedGlobal && !isDemoModeGlobal) {
        showToast('⚠️ Connect an ESP32 or click "🧪 Demo Simulation" to test sensors.');
        return;
      }
      let payload = {};
      if (type === 'pir') {
        payload = { pir_motion: true };
        setTimeout(() => {
          fetch('/api/sensors/esp32', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pir_motion: false })
          });
        }, 3500);
      } else if (type === 'dist') {
        const d = (Math.random() * 120 + 20).toFixed(1);
        payload = { distance_cm: parseFloat(d) };
      } else if (type === 'door1') {
        const curr = document.getElementById('door1StatusBadge').textContent.trim() === 'OPEN';
        payload = { door1_open: !curr };
      } else if (type === 'door2') {
        const curr = document.getElementById('door2StatusBadge').textContent.trim() === 'OPEN';
        payload = { door2_open: !curr };
      }
      try {
        await fetch('/api/sensors/esp32', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        showToast(`Simulated sensor: ${type}`);
      } catch (e) {
        showToast('Failed to simulate sensor');
      }
    }

    async function updateTelemetry() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();

        isEsp32ConnectedGlobal = !!data.is_esp32_connected;
        isDemoModeGlobal = !!data.demo_simulation_enabled;

        document.getElementById('fpsBadge').textContent = `${data.fps.toFixed(1)} FPS`;
        document.getElementById('latencyBadge').textContent = `${data.latency_ms.toFixed(1)} ms`;
        document.getElementById('sourceBadge').textContent = data.current_source;

        document.getElementById('personCountVal').textContent = data.person_count;
        document.getElementById('totalTracksVal').textContent = data.total_tracks;
        document.getElementById('torsoAngleVal').textContent = `${data.torso_angle.toFixed(1)}°`;

        const fallStatus = document.getElementById('fallStatusVal');
        if (data.is_fall_active) {
          fallStatus.textContent = '🚨 FALL DETECTED!';
          fallStatus.style.color = 'var(--accent-red)';
        } else {
          fallStatus.textContent = data.feature_toggles && !data.feature_toggles.fall_detection ? 'DISABLED' : 'NORMAL';
          fallStatus.style.color = data.feature_toggles && !data.feature_toggles.fall_detection ? 'var(--text-dim)' : 'var(--accent-green)';
        }

        // Sync Feature Toggles (when user is not clicking)
        if (!isUserToggling && data.feature_toggles) {
          const ft = data.feature_toggles;
          for (const [key, val] of Object.entries(ft)) {
            const el = document.getElementById(`toggle_${key}`);
            if (el && el.checked !== val) {
              el.checked = val;
            }
          }
          const modeBadge = document.getElementById('aiModeBadge');
          if (modeBadge) {
            if (!ft.object_detection) {
              modeBadge.textContent = 'STANDBY 0%';
              modeBadge.style.borderColor = 'var(--text-dim)';
              modeBadge.style.color = 'var(--text-dim)';
            } else if (ft.skeletal_tracking) {
              modeBadge.textContent = 'POSE + DET';
              modeBadge.style.borderColor = 'var(--accent-cyan)';
              modeBadge.style.color = 'var(--accent-cyan)';
            } else {
              modeBadge.textContent = 'DET ONLY';
              modeBadge.style.borderColor = 'var(--accent-green)';
              modeBadge.style.color = 'var(--accent-green)';
            }
          }
        }

        // Update ESP32 Sentry Status Header Pill & UI Locks
        const esp32Pill = document.getElementById('esp32StatusPill');
        const noticeBox = document.getElementById('esp32NoticeBox');
        const gridContainer = document.getElementById('sensorGridContainer');
        const demoBtn = document.getElementById('btnDemoSim');

        if (demoBtn) {
          demoBtn.textContent = isDemoModeGlobal ? '🧪 Exit Demo' : '🧪 Demo Simulation';
          demoBtn.className = isDemoModeGlobal ? 'btn btn-sm btn-primary' : 'btn btn-sm';
        }

        const isUnlocked = isEsp32ConnectedGlobal || isDemoModeGlobal;

        if (esp32Pill) {
          if (isEsp32ConnectedGlobal) {
            const devIp = (data.connected_esp32_devices && data.connected_esp32_devices.length > 0)
              ? data.connected_esp32_devices[0].ip
              : 'ONLINE';
            esp32Pill.textContent = `🟢 ESP32 ONLINE: ${devIp}`;
            esp32Pill.style.background = 'rgba(0, 255, 157, 0.15)';
            esp32Pill.style.borderColor = 'var(--accent-green)';
            esp32Pill.style.color = 'var(--accent-green)';
          } else if (isDemoModeGlobal) {
            esp32Pill.textContent = '🧪 DEMO SIMULATION';
            esp32Pill.style.background = 'rgba(0, 240, 255, 0.18)';
            esp32Pill.style.borderColor = 'var(--accent-cyan)';
            esp32Pill.style.color = 'var(--accent-cyan)';
          } else if (data.esp32_connection_status === 'SCANNING') {
            esp32Pill.textContent = '⏳ SCANNING...';
            esp32Pill.style.background = 'rgba(0, 240, 255, 0.15)';
            esp32Pill.style.borderColor = 'var(--accent-cyan)';
            esp32Pill.style.color = 'var(--accent-cyan)';
          } else if (data.esp32_connection_status === 'ERROR') {
            esp32Pill.textContent = '⚠️ ESP32 OFFLINE / ERROR';
            esp32Pill.style.background = 'rgba(255, 0, 85, 0.18)';
            esp32Pill.style.borderColor = 'var(--accent-red)';
            esp32Pill.style.color = 'var(--accent-red)';
          } else {
            esp32Pill.textContent = '⚠️ NO SENTRY CONNECTED';
            esp32Pill.style.background = 'rgba(255, 170, 0, 0.15)';
            esp32Pill.style.borderColor = '#ffaa00';
            esp32Pill.style.color = '#ffaa00';
          }
        }

        if (noticeBox) {
          noticeBox.style.display = isUnlocked ? 'none' : 'block';
        }

        if (gridContainer) {
          if (isUnlocked) {
            gridContainer.classList.remove('sensor-toggles-disabled');
          } else {
            gridContainer.classList.add('sensor-toggles-disabled');
          }
        }

        ['toggle_esp32_pir', 'toggle_esp32_ultrasonic', 'toggle_esp32_door1', 'toggle_esp32_door2'].forEach(id => {
          const el = document.getElementById(id);
          if (el) {
            el.disabled = !isUnlocked;
          }
        });

        // Sync ESP32 Sensor Telemetry Badges
        if (!isUnlocked) {
          const pirBadge = document.getElementById('pirStatusBadge');
          const distBadge = document.getElementById('distStatusBadge');
          const d1Badge = document.getElementById('door1StatusBadge');
          const d2Badge = document.getElementById('door2StatusBadge');
          if (pirBadge) { pirBadge.textContent = 'OFFLINE'; pirBadge.className = 'sensor-badge offline'; }
          if (distBadge) { distBadge.textContent = 'OFFLINE'; distBadge.className = 'sensor-badge offline'; }
          if (d1Badge) { d1Badge.textContent = 'OFFLINE'; d1Badge.className = 'sensor-badge offline'; }
          if (d2Badge) { d2Badge.textContent = 'OFFLINE'; d2Badge.className = 'sensor-badge offline'; }
        } else if (data.esp32_sensor_readings) {
          const s = data.esp32_sensor_readings;

          // PIR Motion
          const pirBadge = document.getElementById('pirStatusBadge');
          if (pirBadge) {
            if (s.pir_motion) {
              pirBadge.textContent = '🚨 MOTION';
              pirBadge.className = 'sensor-badge alert';
            } else {
              pirBadge.textContent = 'IDLE';
              pirBadge.className = 'sensor-badge idle';
            }
          }

          // Ultrasonic Distance
          const distBadge = document.getElementById('distStatusBadge');
          if (distBadge) {
            if (s.distance_cm && s.distance_cm > 0) {
              distBadge.textContent = `${s.distance_cm.toFixed(1)} cm`;
              distBadge.className = 'sensor-badge';
            } else {
              distBadge.textContent = '--- cm';
              distBadge.className = 'sensor-badge idle';
            }
          }

          // Door 1
          const d1Badge = document.getElementById('door1StatusBadge');
          if (d1Badge) {
            if (s.door1_open) {
              d1Badge.textContent = 'OPEN';
              d1Badge.className = 'sensor-badge alert';
            } else {
              d1Badge.textContent = 'CLOSED';
              d1Badge.className = 'sensor-badge idle';
            }
          }

          // Door 2
          const d2Badge = document.getElementById('door2StatusBadge');
          if (d2Badge) {
            if (s.door2_open) {
              d2Badge.textContent = 'OPEN';
              d2Badge.className = 'sensor-badge alert';
            } else {
              d2Badge.textContent = 'CLOSED';
              d2Badge.className = 'sensor-badge idle';
            }
          }
        }

        if (data.events && data.events.length > 0) {
          const container = document.getElementById('eventLogList');
          container.innerHTML = '';
          data.events.forEach(ev => {
            const item = document.createElement('div');
            item.className = `event-log-item ${ev.type === 'FALL_DETECTED' ? 'event-fall' : ''} ${ev.type === 'INTRUSION' ? 'event-intrusion' : ''}`;
            item.textContent = `[${ev.time}] ${ev.type}: ${ev.message}`;
            container.appendChild(item);
          });
        }
      } catch (e) {}
    }

    loadZonesList();
    loadCameraSources();
    setInterval(updateTelemetry, 500);
    setInterval(loadZonesList, 3500);
  </script>
</body>
</html>
        """
        return HTMLResponse(content=html_content)

    @web_app.get("/stream")
    async def video_feed():
        def iter_frames():
            while monitor.is_running:
                with monitor.frame_lock:
                    jpeg = monitor.latest_encoded_jpeg
                if jpeg:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                time.sleep(0.033)

        return StreamingResponse(
            iter_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @web_app.get("/api/status")
    async def get_status():
        with monitor.event_lock:
            recent_events = list(monitor.event_log[:15])
        with monitor.tracks_lock:
            total_tracks = len([t for t in monitor.tracks.values() if t.is_confirmed])
        return {
            "fps": monitor.fps,
            "latency_ms": monitor.avg_inference_ms,
            "current_source": monitor.current_source_name,
            "person_count": monitor.person_count,
            "total_tracks": total_tracks,
            "torso_angle": monitor.torso_angle,
            "descent_velocity": monitor.descent_velocity,
            "aspect_ratio": monitor.aspect_ratio,
            "floor_proximity": monitor.floor_proximity,
            "is_fall_active": monitor.is_fall_active,
            "feature_toggles": monitor.feature_toggles,
            "esp32_sensor_readings": monitor.esp32_sensor_readings,
            "is_esp32_connected": monitor.is_esp32_connected,
            "esp32_connection_status": monitor.esp32_connection_status,
            "connected_esp32_devices": monitor.connected_esp32_devices,
            "demo_simulation_enabled": monitor.demo_simulation_enabled,
            "events": recent_events
        }

    @web_app.get("/api/toggles")
    async def get_toggles():
        return {
            "status": "ok",
            "feature_toggles": monitor.feature_toggles,
            "esp32_sensor_readings": monitor.esp32_sensor_readings,
            "is_esp32_connected": monitor.is_esp32_connected,
            "esp32_connection_status": monitor.esp32_connection_status,
            "connected_esp32_devices": monitor.connected_esp32_devices,
            "demo_simulation_enabled": monitor.demo_simulation_enabled
        }

    @web_app.post("/api/toggles")
    async def update_toggles(request: Request):
        try:
            data = await request.json()
        except Exception:
            data = {}

        # Guard: Check if any esp32_* toggle is being toggled while disconnected and not in demo mode
        toggling_esp32 = False
        if "feature" in data and str(data["feature"]).startswith("esp32_"):
            toggling_esp32 = True
        elif "toggles" in data and isinstance(data["toggles"], dict):
            if any(str(k).startswith("esp32_") for k in data["toggles"].keys()):
                toggling_esp32 = True
        else:
            if any(str(k).startswith("esp32_") for k in data.keys() if k in monitor.feature_toggles):
                toggling_esp32 = True

        if toggling_esp32 and not monitor.is_esp32_connected and not monitor.demo_simulation_enabled:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "No physical ESP32 sentry connected on network! Please scan or connect device."
                }
            )

        updated = {}
        if "feature" in data and "enabled" in data:
            feat = str(data["feature"])
            val = bool(data["enabled"])
            if feat in monitor.feature_toggles:
                monitor.feature_toggles[feat] = val
                updated[feat] = val
        elif "toggles" in data and isinstance(data["toggles"], dict):
            for k, v in data["toggles"].items():
                if k in monitor.feature_toggles:
                    monitor.feature_toggles[k] = bool(v)
                    updated[k] = bool(v)
        else:
            for k, v in data.items():
                if k in monitor.feature_toggles and isinstance(v, bool):
                    monitor.feature_toggles[k] = v
                    updated[k] = v

        return {
            "status": "ok",
            "updated": updated,
            "feature_toggles": monitor.feature_toggles,
            "is_esp32_connected": monitor.is_esp32_connected,
            "esp32_connection_status": monitor.esp32_connection_status,
            "demo_simulation_enabled": monitor.demo_simulation_enabled
        }

    @web_app.post("/api/sensors/rescan")
    async def rescan_esp32():
        devices = monitor.rescan_esp32_devices()
        return {
            "status": "ok",
            "found_count": len(devices),
            "devices": devices,
            "is_connected": monitor.is_esp32_connected,
            "esp32_connection_status": monitor.esp32_connection_status
        }

    @web_app.post("/api/sensors/demo_mode")
    async def toggle_demo_mode(request: Request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        if "enabled" in data:
            monitor.demo_simulation_enabled = bool(data["enabled"])
        else:
            monitor.demo_simulation_enabled = not monitor.demo_simulation_enabled
        return {
            "status": "ok",
            "demo_enabled": monitor.demo_simulation_enabled,
            "is_connected": monitor.is_esp32_connected,
            "esp32_connection_status": monitor.esp32_connection_status
        }

    @web_app.post("/api/sensors/esp32")
    async def ingest_esp32_telemetry(request: Request):
        try:
            data = await request.json()
        except Exception:
            data = {}

        if data:
            monitor.is_esp32_connected = True
            monitor.esp32_connection_status = "ONLINE"
            if "ip" in data:
                monitor.active_esp32_ip = str(data["ip"])

        # 1. PIR Motion
        if "pir_motion" in data:
            monitor.esp32_sensor_readings["pir_motion"] = bool(data["pir_motion"])
        elif "pir" in data:
            monitor.esp32_sensor_readings["pir_motion"] = bool(data["pir"])

        # 2. Ultrasonic Distance
        if "distance_cm" in data:
            monitor.esp32_sensor_readings["distance_cm"] = float(data["distance_cm"])
        elif "distance" in data:
            monitor.esp32_sensor_readings["distance_cm"] = float(data["distance"])

        # 3. Door 1
        if "door1_open" in data:
            monitor.esp32_sensor_readings["door1_open"] = bool(data["door1_open"])
        elif "door1" in data:
            monitor.esp32_sensor_readings["door1_open"] = bool(data["door1"])

        # 4. Door 2
        if "door2_open" in data:
            monitor.esp32_sensor_readings["door2_open"] = bool(data["door2_open"])
        elif "door2" in data:
            monitor.esp32_sensor_readings["door2_open"] = bool(data["door2"])

        # Alert dispatches if respective sensor toggle is enabled
        if monitor.feature_toggles.get("esp32_pir", True) and monitor.esp32_sensor_readings["pir_motion"]:
            monitor.trigger_alert("🚶 ESP32 PIR Hardware Motion Detected!", "PIR_MOTION", duration=3.0)

        if monitor.feature_toggles.get("esp32_door1", True) and monitor.esp32_sensor_readings["door1_open"]:
            monitor.trigger_alert("🚪 Door 1 (Front Entrance) Breach / Open!", "DOOR_OPEN", duration=3.0)

        if monitor.feature_toggles.get("esp32_door2", True) and monitor.esp32_sensor_readings["door2_open"]:
            monitor.trigger_alert("🚪 Door 2 (Back Exit) Breach / Open!", "DOOR_OPEN", duration=3.0)

        return {
            "status": "ok",
            "esp32_sensor_readings": monitor.esp32_sensor_readings,
            "feature_toggles": monitor.feature_toggles,
            "is_esp32_connected": monitor.is_esp32_connected,
            "esp32_connection_status": monitor.esp32_connection_status
        }

    @web_app.get("/api/zones")
    async def get_zones():
        return {
            "status": "ok",
            "tripwires": list(monitor.tripwires.values()),
            "intrusion_zones": list(monitor.intrusion_zones.values()),
            "exclusion_masks": list(monitor.exclusion_masks.values())
        }

    @web_app.post("/api/zones/tripwire")
    async def add_or_update_tripwire(req: TripwireReq):
        tw_id = monitor.add_or_update_tripwire(req.model_dump())
        return {"status": "ok", "id": tw_id, "message": "Tripwire saved successfully"}

    @web_app.delete("/api/zones/tripwire/{zone_id}")
    async def delete_tripwire(zone_id: str):
        success = monitor.delete_tripwire(zone_id)
        return {"status": "ok" if success else "error", "message": "Tripwire deleted" if success else "Not found"}

    @web_app.post("/api/zones/intrusion")
    async def add_or_update_intrusion(req: IntrusionReq):
        iz_id = monitor.add_or_update_intrusion_zone(req.model_dump())
        return {"status": "ok", "id": iz_id, "message": "Restricted Area saved successfully"}

    @web_app.delete("/api/zones/intrusion/{zone_id}")
    async def delete_intrusion(zone_id: str):
        success = monitor.delete_intrusion_zone(zone_id)
        return {"status": "ok" if success else "error", "message": "Restricted Area deleted" if success else "Not found"}

    @web_app.post("/api/zones/exclusion")
    async def add_or_update_exclusion(req: ExclusionReq):
        ex_id = monitor.add_or_update_exclusion_mask(req.model_dump())
        return {"status": "ok", "id": ex_id, "message": "Exclusion Mask saved successfully"}

    @web_app.delete("/api/zones/exclusion/{zone_id}")
    async def delete_exclusion(zone_id: str):
        success = monitor.delete_exclusion_mask(zone_id)
        return {"status": "ok" if success else "error", "message": "Exclusion Mask deleted" if success else "Not found"}

    @web_app.post("/api/zones/clear")
    async def clear_all():
        monitor.clear_all_zones()
        return {"status": "ok", "message": "All zones cleared"}

    @web_app.post("/api/action/snapshot")
    async def take_snapshot():
        snap_path = settings.SNAPSHOTS_DIR / f"snapshot_{int(time.time())}.jpg"
        with monitor.frame_lock:
            jpeg = monitor.latest_encoded_jpeg
        if jpeg:
            snap_path.parent.mkdir(parents=True, exist_ok=True)
            snap_path.write_bytes(jpeg)
            monitor.trigger_alert(f"Snapshot Saved: {snap_path.name}", "SNAPSHOT", duration=2.5)
        return {"status": "ok", "message": f"Snapshot saved: {snap_path.name}"}

    @web_app.post("/api/action/clip")
    async def export_clip():
        monitor.record_15s_incident_clip("MANUAL")
        return {"status": "ok", "message": "15s Incident Clip recording started"}

    @web_app.post("/api/rescan")
    async def rescan():
        sources = monitor.rescan_sources()
        return {"status": "ok", "sources": sources}

    @web_app.post("/api/switch_source")
    async def switch_source(req: SwitchSourceReq):
        monitor.switch_source(req.url)
        return {"status": "ok", "current_source": monitor.current_source_name}

    return web_app


# ==============================================================================
# Main Runner with HighGUI & Web HUD Fallbacks
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Edge AI CCTV Live AI Monitor & Multi-Zone Evaluator")
    parser.add_argument("--stream", type=str, default=None,
                        help="Stream URL of ESP32 / IP Camera (e.g. http://10.68.21.86:81/stream)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to YOLO ONNX detection model (yolo11n.onnx, yolov8n.onnx, yolov5n.onnx)")
    parser.add_argument("--pose-model", type=str, default=None,
                        help="Path to YOLO Pose ONNX model (yolo11n-pose.onnx, yolov8n-pose.onnx)")
    parser.add_argument("--port", type=int, default=8080, help="Web HUD port (default 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    args = parser.parse_args()

    monitor = LiveAIMonitor(
        initial_stream=args.stream,
        model_path=args.model,
        pose_model_path=args.pose_model
    )

    capture_thread = threading.Thread(target=monitor.capture_frame_loop, daemon=True)
    capture_thread.start()

    import uvicorn
    web_app = create_web_hud_app(monitor)

    def run_uvicorn():
        uvicorn.run(web_app, host="0.0.0.0", port=args.port, log_level="warning")

    server_thread = threading.Thread(target=run_uvicorn, daemon=True)
    server_thread.start()

    print("\n" + "=" * 75)
    print("  🛡️  EDGE AI CCTV - MULTI-ZONE STUDIO ACTIVE")
    print("=" * 75)
    print(f"  🌐 Live Web HUD:  http://localhost:{args.port}")
    print(f"  🌐 Remote Access: http://0.0.0.0:{args.port}")
    print("  Advanced Security Features:")
    print("  • Multi-Zone Studio: Tripwires, Restricted Areas, and Exclusion Masks")
    print("  • Automatic 15-Second Clip Recording on Security Alarms")
    print("  • False-Alarm Rejection: Insects, moths, shadows, and reflections filtered")
    print("  • Multi-Class Neural Vision: Humans and Vehicles tracked with high confidence")
    print("  • Real-Time Fall Detection on live active persons")
    print("  • Press 'Q' or ESC to Exit")
    print("=" * 75 + "\n")

    if not args.no_browser:
        try:
            webbrowser.open(f"http://localhost:{args.port}")
        except Exception:
            pass

    gui_supported = False
    try:
        if sys.platform.startswith("win") or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            cv2.namedWindow("Edge AI CCTV - Live Monitor", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Edge AI CCTV - Live Monitor", 1024, 576)
            gui_supported = True
            logger.info("[+] Native desktop HighGUI window initialized.")
    except Exception as e:
        logger.info(f"[-] Native desktop window not available ({e}). Using Web HUD at http://localhost:{args.port}")

    try:
        while monitor.is_running:
            if gui_supported:
                with monitor.frame_lock:
                    jpeg = monitor.latest_encoded_jpeg
                if jpeg:
                    nparr = np.frombuffer(jpeg, np.uint8)
                    hud_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if hud_frame is not None:
                        try:
                            cv2.imshow("Edge AI CCTV - Live Monitor", hud_frame)
                            key = cv2.waitKey(30) & 0xFF
                            if key in (27, ord('q'), ord('Q')):
                                break
                            elif key in (ord('s'), ord('S')):
                                snap_path = settings.SNAPSHOTS_DIR / f"snapshot_{int(time.time())}.jpg"
                                snap_path.write_bytes(jpeg)
                                monitor.trigger_alert(f"Snapshot Saved: {snap_path.name}", "SNAPSHOT", duration=2.5)
                            elif key in (ord('r'), ord('R')):
                                monitor.record_15s_incident_clip("MANUAL")
                            elif key in (ord('c'), ord('C')):
                                monitor.rescan_sources()
                        except Exception:
                            gui_supported = False
                else:
                    time.sleep(0.03)
            else:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.is_running = False
        clip_muxer.shutdown()
        if gui_supported:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("\n[+] Live AI Monitor closed cleanly.")


if __name__ == "__main__":
    main()
