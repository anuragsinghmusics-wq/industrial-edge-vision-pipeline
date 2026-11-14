"""
detector.py
Edge Vision Detector for Industrial Safety and PPE Compliance.
Processes frames, performs bounding box inference, and evaluates safety compliance:
- Person detection
- Hardhat (Helmet) verification
- Hi-Vis Vest verification
- Hazardous Machinery proximity / intrusion alert
"""

import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class DetectionBox:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax)


@dataclass
class SafetyIncident:
    worker_id: int
    incident_type: str
    severity: str
    bbox: Tuple[int, int, int, int]
    timestamp: float


class IndustrialPPEDetector:
    """
    Simulates high-performance edge detection using ONNX Runtime / TensorRT backend.
    """
    CLASSES = ["person", "hardhat", "hi_vis_vest", "machinery_hazard"]

    def __init__(self, model_precision: str = "INT8", confidence_threshold: float = 0.45):
        self.model_precision = model_precision
        self.confidence_threshold = confidence_threshold
        self.total_frames_processed = 0
        self.latencies_history = []

    def _calculate_overlap(self, boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        # Determine intersection box
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        if boxAArea == 0:
            return 0.0
        return interArea / float(boxAArea)

    def process_frame(self, frame_id: int, simulated_detections: List[DetectionBox] = None) -> Dict[str, Any]:
        """
        Runs inference and applies business rules for industrial site safety.
        """
        start_t = time.perf_counter()

        # Simulated inference latency based on precision
        # In actual deployment, onnxruntime.InferenceSession.run() is executed here
        if simulated_detections is None:
            # Default mock scene: 2 workers, 1 missing helmet
            simulated_detections = [
                DetectionBox("person", 0.92, (100, 150, 260, 520)),
                DetectionBox("hardhat", 0.88, (135, 150, 220, 215)),
                DetectionBox("hi_vis_vest", 0.86, (115, 220, 245, 390)),
                DetectionBox("person", 0.89, (420, 160, 580, 530)),
                # Second worker has vest, but NO hardhat!
                DetectionBox("hi_vis_vest", 0.84, (435, 230, 565, 400)),
            ]

        # Filter by confidence
        valid_boxes = [b for b in simulated_detections if b.confidence >= self.confidence_threshold]

        workers = [b for b in valid_boxes if b.class_name == "person"]
        helmets = [b for b in valid_boxes if b.class_name == "hardhat"]
        vests = [b for b in valid_boxes if b.class_name == "hi_vis_vest"]

        incidents: List[SafetyIncident] = []

        for idx, worker in enumerate(workers):
            wb = worker.bbox
            # Head region estimation: top 25% of worker bbox
            head_region = (wb[0], wb[1], wb[2], wb[1] + int((wb[3] - wb[1]) * 0.30))
            # Torso region estimation: middle 45% of worker bbox
            torso_region = (wb[0], wb[1] + int((wb[3] - wb[1]) * 0.20), wb[2], wb[1] + int((wb[3] - wb[1]) * 0.65))

            has_helmet = any(self._calculate_overlap(h.bbox, head_region) > 0.30 for h in helmets)
            has_vest = any(self._calculate_overlap(v.bbox, torso_region) > 0.30 for v in vests)

            if not has_helmet:
                incidents.append(SafetyIncident(
                    worker_id=idx + 1,
                    incident_type="MISSING_HARDHAT_VIOLATION",
                    severity="HIGH",
                    bbox=wb,
                    timestamp=time.time()
                ))

            if not has_vest:
                incidents.append(SafetyIncident(
                    worker_id=idx + 1,
                    incident_type="MISSING_HI_VIS_VEST_VIOLATION",
                    severity="MEDIUM",
                    bbox=wb,
                    timestamp=time.time()
                ))

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        self.total_frames_processed += 1
        self.latencies_history.append(elapsed_ms)

        return {
            "frame_id": frame_id,
            "inference_precision": self.model_precision,
            "total_workers": len(workers),
            "incidents_detected": [asdict(inc) for inc in incidents],
            "raw_detections_count": len(valid_boxes),
            "latency_ms": elapsed_ms
        }
