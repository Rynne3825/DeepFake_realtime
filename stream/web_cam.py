# =============================================================================
# File: stream/web_cam.py
# Mục đích: Bắt luồng khung hình từ Webcam vật lý bằng OpenCV.
#
# GIẢI THÍCH CHO BÁO CÁO:
#   Webcam truyền hình ảnh liên tục (30-60 khung hình/giây). Module này
#   sử dụng ĐA LUỒNG (Multi-threading) để đọc khung hình trong một
#   Thread riêng biệt, tránh làm đơ giao diện người dùng (UI).
#
#   Vấn đề nếu không dùng đa luồng:
#   - Camera.read() là hàm CHẶN (blocking) - nó đợi cho đến khi có frame mới.
#   - Nếu chạy trong Thread chính (cùng với UI), giao diện sẽ bị đông cứng
#     trong lúc chờ camera trả frame.
#
#   Giải pháp:
#   - Tạo một Thread riêng liên tục đọc frame từ camera.
#   - Thread UI chỉ cần lấy frame MỚI NHẤT (latest frame) để hiển thị.
# =============================================================================

import cv2
import numpy as np
import time
import threading
from typing import Optional, Tuple


class WebcamCapture:
    """
    Bộ bắt luồng Webcam sử dụng đa luồng (Threaded Webcam Capture).

    Cách dùng:
        cam = WebcamCapture(device_index=0)
        cam.start()
        while cam.is_running:
            ret, frame = cam.read()
            if ret:
                # Xử lý frame ở đây
                pass
        cam.release()
    """

    def __init__(self, device_index: int = 0):
        """
        Khởi tạo bộ bắt webcam.

        Tham số:
            device_index: Chỉ số camera (0 = camera mặc định, 1 = camera phụ).
        """
        self.device_index = device_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running: bool = False

        # Frame mới nhất được lưu ở đây (chia sẻ giữa 2 threads)
        self._current_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()  # Khóa đồng bộ khi đọc/ghi frame
        self._thread: Optional[threading.Thread] = None

    def start(self, width: int = 960, height: int = 540, fps: int = 30) -> bool:
        """
        Mở camera và bắt đầu đọc frame trong Thread riêng.

        Tham số:
            width: Chiều rộng khung hình mong muốn (pixel).
            height: Chiều cao khung hình mong muốn (pixel).
            fps: Số khung hình mỗi giây (FPS) mong muốn.

        Trả về:
            True nếu mở camera thành công, False nếu thất bại.
        """
        try:
            # Thử mở camera bằng DirectShow (Windows) trước, rồi fallback
            self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.device_index)

            if not self.cap or not self.cap.isOpened():
                print("[WEBCAM] Không thể mở camera!")
                return False

            # Cấu hình độ phân giải và FPS
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            # Ưu tiên độ trễ thấp cho chế độ live nếu backend camera hỗ trợ.
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.is_running = True

            # Bắt đầu Thread đọc frame liên tục
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

            return True

        except Exception as e:
            print(f"[WEBCAM] Lỗi khởi động: {e}")
            if self.cap:
                self.cap.release()
            return False

    def _capture_loop(self):
        """
        Vòng lặp đọc frame liên tục (chạy trong Thread riêng).

        GIẢI THÍCH:
            Thread này chạy độc lập, liên tục gọi cap.read() để lấy frame
            mới nhất từ camera và lưu vào self._current_frame.
            Thread UI sẽ đọc self._current_frame bất cứ lúc nào cần.
        """
        while self.is_running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                with self._frame_lock:
                    self._current_frame = frame
            else:
                # Tránh vòng lặp bận nếu webcam tạm thời chưa trả frame.
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Đọc frame mới nhất từ camera.

        Trả về:
            Tuple (success, frame):
            - success: True nếu có frame, False nếu không.
            - frame: Khung hình BGR (np.ndarray) hoặc None.
        """
        with self._frame_lock:
            if self._current_frame is not None:
                return True, self._current_frame.copy()
        return False, None

    def release(self):
        """Dừng camera và giải phóng tài nguyên."""
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._current_frame = None
