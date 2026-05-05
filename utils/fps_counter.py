# =============================================================================
# File: utils/fps_counter.py
# Mục đích: Đo lường số khung hình được xử lý mỗi giây (FPS - Frames Per Second).
#
# GIẢI THÍCH CHO BÁO CÁO:
#   FPS là chỉ số quan trọng nhất để đánh giá hiệu năng của hệ thống
#   thời gian thực. Con người cảm nhận video mượt mà khi FPS >= 25.
#   - FPS < 15: Giật lag rõ rệt, không đạt yêu cầu thời gian thực.
#   - FPS 15-25: Chấp nhận được cho demo.
#   - FPS >= 30: Mượt mà, đạt chuẩn thời gian thực.
#
#   Class này sử dụng kỹ thuật "Cửa sổ trượt" (Sliding Window) để tính
#   FPS trung bình trong 30 khung hình gần nhất, giúp giá trị hiển thị
#   ổn định hơn so với việc tính FPS tức thời (tức thời sẽ nhảy số liên tục).
# =============================================================================

import time
from collections import deque


class FPSCounter:
    """
    Bộ đếm FPS sử dụng kỹ thuật Cửa sổ trượt (Sliding Window).
    
    Cách dùng:
        fps_counter = FPSCounter()
        while True:
            fps_counter.tick()            # Gọi mỗi khi xử lý xong 1 frame
            print(fps_counter.get_fps())  # Lấy giá trị FPS hiện tại
    """

    def __init__(self, window_size: int = 30):
        """
        Khởi tạo bộ đếm FPS.
        
        Tham số:
            window_size (int): Số lượng khung hình dùng để tính trung bình.
                               Mặc định là 30 (tính FPS trung bình trên 30 frame gần nhất).
        """
        # deque (double-ended queue) tự động loại bỏ phần tử cũ nhất khi vượt maxlen
        self._timestamps = deque(maxlen=window_size)

    def tick(self):
        """
        Đánh dấu thời điểm hoàn thành xử lý 1 khung hình.
        Gọi hàm này sau mỗi lần xử lý xong 1 frame từ webcam.
        """
        self._timestamps.append(time.perf_counter())

    def get_fps(self) -> float:
        """
        Tính và trả về FPS trung bình dựa trên cửa sổ trượt.
        
        Trả về:
            float: Số khung hình trên giây (FPS). Trả về 0.0 nếu chưa đủ dữ liệu.
        
        Công thức:
            FPS = (Số frame - 1) / (Thời gian frame cuối - Thời gian frame đầu)
        """
        if len(self._timestamps) < 2:
            return 0.0

        # Khoảng thời gian từ frame đầu đến frame cuối trong cửa sổ
        time_span = self._timestamps[-1] - self._timestamps[0]

        if time_span <= 0:
            return 0.0

        # Số khoảng cách giữa các frame = số frame - 1
        return (len(self._timestamps) - 1) / time_span
