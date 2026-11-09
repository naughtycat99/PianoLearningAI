"""
Module cấu hình trung tâm (Configuration & Constants) cho hệ thống Piano Learning AI.
Tập trung toàn bộ các siêu tham số (hyperparameters), ngưỡng thuật toán (thresholds),
ánh xạ khớp xương MediaPipe và bảng màu giao diện nhằm đảm bảo chuẩn Clean Code (PEP 8).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class MediaPipeConfig:
    """Cấu hình cho mô hình nhận diện bàn tay Google MediaPipe Hands."""
    static_image_mode: bool = False
    max_num_hands: int = 4
    model_complexity: int = 1
    min_detection_confidence: float = 0.3
    min_tracking_confidence: float = 0.3


@dataclass(frozen=True)
class SignalProcessingConfig:
    """Các ngưỡng toán học cho thuật toán Xử lý tín hiệu số (DSP) và DTW."""
    min_frames_required: int = 10
    min_intro_frames: int = 30
    smoothing_window_size: int = 10
    intro_velocity_threshold: float = 0.005
    active_motion_threshold: float = 0.01
    segment_motion_threshold: float = 0.03
    onset_peak_height: float = 0.015
    onset_peak_distance: int = 10
    chord_cluster_frame_gap: int = 12
    event_window_half_size: int = 10
    error_score_threshold: float = 70.0
    weak_finger_threshold: float = 75.0
    default_fps: float = 30.0
    epsilon: float = 1e-6


# Danh sách chỉ số (Landmark IDs) của 5 đầu ngón tay theo chuẩn MediaPipe
FINGER_TIPS: List[int] = [4, 8, 12, 16, 20]

# Tên hiển thị chuẩn của 5 ngón tay
FINGER_NAMES: List[str] = [
    "Ngon cai",
    "Ngon tro",
    "Ngon giua",
    "Ngon ap ut",
    "Ngon ut",
]

# Tên tiếng Việt có dấu phục vụ hiển thị lên biểu đồ và báo cáo GUI
FINGER_DISPLAY_NAMES: Dict[str, str] = {
    "Ngon cai": "Ngón cái",
    "Ngon tro": "Ngón trỏ",
    "Ngon giua": "Ngón giữa",
    "Ngon ap ut": "Ngón áp út",
    "Ngon ut": "Ngón út",
}

# Ánh xạ tên ngón tay sang chỉ số khớp đầu ngón (Landmark Tip ID)
FINGER_TIPS_MAP: Dict[str, int] = dict(zip(FINGER_NAMES, FINGER_TIPS))

# Các cặp kết nối xương bàn tay (Skeleton Connections) để vẽ lên khung hình
MEDIAPIPE_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Danh sách 22 nốt nhạc chuẩn (từ C3 đến C6) dùng cho nhận diện bàn phím
DEFAULT_NOTES_LIST: List[str] = [
    "Đô (C3)", "Rê (D3)", "Mi (E3)", "Fa (F3)", "Sol (G3)", "La (A3)", "Si (B3)",
    "Đô (C4 - Đô Trung)", "Rê (D4)", "Mi (E4)", "Fa (F4)", "Sol (G4)", "La (A4)", "Si (B4)",
    "Đô (C5)", "Rê (D5)", "Mi (E5)", "Fa (F5)", "Sol (G5)", "La (A5)", "Si (B5)",
    "Đô (C6)",
]

# Khởi tạo đối tượng cấu hình mặc định
MP_CONFIG = MediaPipeConfig()
DSP_CONFIG = SignalProcessingConfig()
