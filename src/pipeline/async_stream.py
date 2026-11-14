"""
async_stream.py
Asynchronous RTSP Video Ingestion Pipeline with Decoupled Buffer.
Prevents frame backlog and latency drift under high-frequency factory video streams.
"""

import time
import queue
import threading
from typing import Optional, Callable
from collections import deque


class AsyncRTSPStreamWorker:
    """
    Decoupled Producer-Consumer stream worker.
    The camera stream reader executes on an independent I/O thread,
    pushing to a fixed bounded ring-buffer (maxlen=2) with drop-oldest policy.
    This guarantees zero latency accumulation even if camera frame rate exceeds inference throughput.
    """

    def __init__(self, stream_url: str = "rtsp://mock-factory-cam-01:554/live", max_queue_size: int = 2):
        self.stream_url = stream_url
        self.max_queue_size = max_queue_size
        self.frame_buffer = deque(maxlen=max_queue_size)
        self.is_running = False
        self.capture_thread: Optional[threading.Thread] = None
        self.total_frames_received = 0
        self.total_frames_dropped = 0
        self.lock = threading.Lock()

    def start(self):
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def stop(self):
        self.is_running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)

    def _capture_loop(self):
        frame_idx = 0
        while self.is_running:
            # Simulate 25 FPS RTSP stream arrival (40ms interval)
            time.sleep(0.040)
            frame_data = {
                "frame_id": frame_idx,
                "timestamp": time.time(),
                "resolution": (1280, 720, 3)
            }

            with self.lock:
                if len(self.frame_buffer) >= self.max_queue_size:
                    self.total_frames_dropped += 1
                self.frame_buffer.append(frame_data)
                self.total_frames_received += 1

            frame_idx += 1

    def get_latest_frame(self) -> Optional[dict]:
        with self.lock:
            if self.frame_buffer:
                return self.frame_buffer.pop()
        return None

    def get_stream_stats(self) -> dict:
        with self.lock:
            drop_rate = (self.total_frames_dropped / max(1, self.total_frames_received)) * 100.0
            return {
                "stream_url": self.stream_url,
                "frames_received": self.total_frames_received,
                "frames_dropped": self.total_frames_dropped,
                "drop_rate_pct": round(drop_rate, 2),
                "buffer_status": "REAL_TIME_HEALTHY" if drop_rate < 15.0 else "BACKPRESSURE_DETECTED"
            }
