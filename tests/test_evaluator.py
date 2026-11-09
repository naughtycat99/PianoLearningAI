"""
Bộ kiểm thử tự động (Unit Tests) cho các thuật toán Xử lý tín hiệu số (DSP),
Chuẩn hóa không gian Z-Score, Tương quan chéo (Cross-Correlation) và DTW.
"""

import os
import sys
import unittest
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from biomechanics_evaluator import (  # noqa: E402
    compute_joint_angle_deg,
    score_finger_biomechanics,
)
from evaluator import (  # noqa: E402
    calculate_finger_score,
    extract_normalized_features,
    find_start_index,
)
from sheet_music_evaluator import (  # noqa: E402
    normalize_note_code,
    note_code_to_semitone,
)
from timeline_evaluator import (  # noqa: E402
    dtw_align_sequences,
    score_aligned_finger_segment,
)


class TestPianoAIAlgorithms(unittest.TestCase):
    """Kiểm thử tính đúng đắn của các thuật toán cốt lõi trong Piano Learning AI."""

    def test_find_start_index_auto_intro_cropping(self) -> None:
        """Kiểm tra thuật toán đạo hàm tự động cắt bỏ đoạn Intro tĩnh ở đầu video."""
        # 40 khung hình đầu đứng yên (giá trị 1.0), từ khung hình 40 bắt đầu dao động mạnh
        intro_static = np.ones(40)
        active_play = 1.0 + 0.2 * np.sin(np.linspace(0, 10 * np.pi, 60))
        signal = np.concatenate([intro_static, active_play])

        start_idx = find_start_index(signal)
        self.assertGreaterEqual(start_idx, 30)
        self.assertLessEqual(start_idx, 45)

    def test_calculate_finger_score_identical_trajectories(self) -> None:
        """Hai quỹ đạo giống hệt nhau (kể cả khác tỷ lệ biên độ) phải đạt điểm ~100."""
        t = np.linspace(0, 4 * np.pi, 120)
        # Khoảng cách từ đầu ngón đến cổ tay biến thiên tuần hoàn theo nhịp gõ phím
        expert_traj = np.column_stack([1.0 + 0.3 * np.sin(t), 1.0 + 0.3 * np.sin(t)])
        # Học viên có góc máy xa hơn (tỷ lệ 0.5x) và thêm 20 frame ngồi chờ tĩnh ở đầu
        intro_padding = np.full((20, 2), 0.5)
        user_traj = np.vstack([intro_padding, 0.5 * expert_traj])

        score = calculate_finger_score(expert_traj, user_traj)
        self.assertGreaterEqual(score, 90.0)

    def test_dtw_align_sequences(self) -> None:
        """Kiểm tra thuật toán Quy hoạch động DTW căn chỉnh 2 chuỗi khác độ dài."""
        t_exp = np.linspace(0, 2 * np.pi, 50)
        t_usr = np.linspace(0, 2 * np.pi, 65)
        exp_seq = np.column_stack([np.sin(t_exp), np.cos(t_exp)])
        usr_seq = np.column_stack([np.sin(t_usr), np.cos(t_usr)])

        alignment = dtw_align_sequences(exp_seq, usr_seq)
        self.assertEqual(len(alignment), 65)
        self.assertEqual(alignment[0], 0)
        self.assertEqual(alignment[64], 49)

    def test_score_aligned_finger_segment(self) -> None:
        """Kiểm tra hàm chấm điểm phân đoạn ngắn quanh thời điểm nhấn phím (Note Onset)."""
        t = np.linspace(0, np.pi, 20)
        seg_exp = np.column_stack([0.2 * np.sin(t), 0.2 * np.cos(t)])
        seg_usr_good = np.column_stack([0.2 * np.sin(t), 0.2 * np.cos(t)])
        seg_usr_static = np.zeros((20, 2))

        good_score = score_aligned_finger_segment(seg_exp, seg_usr_good)
        bad_score = score_aligned_finger_segment(seg_exp, seg_usr_static)

        self.assertAlmostEqual(good_score, 100.0, places=1)
        self.assertLess(bad_score, 70.0)

    def test_extract_normalized_features_interpolation(self) -> None:
        """Kiểm tra khả năng nội suy tuyến tính khi có khung hình mất dấu bàn tay."""
        fake_hand = [{"id": i, "x": 0.3 + i * 0.01, "y": 0.5 + i * 0.01, "z": 0.0} for i in range(21)]
        frames = [
            {"frame_index": 0, "timestamp": 0.0, "hands": [fake_hand]},
            {"frame_index": 1, "timestamp": 0.033, "hands": []},  # Mất dấu tay
            {"frame_index": 2, "timestamp": 0.066, "hands": [fake_hand]},
        ]
        feats = extract_normalized_features(frames)
        self.assertEqual(feats["Left"]["Ngon tro"].shape, (3, 2))
        # Khung hình ở giữa (index 1) phải được bù dữ liệu khác 0
        self.assertNotEqual(feats["Left"]["Ngon tro"][1, 0], 0.0)

    def test_biomechanics_joint_angle_and_standalone_scoring(self) -> None:
        """Kiểm tra thuật toán tính góc khớp 3D và chấm điểm độc lập (không cần video mẫu)."""
        p_base = {"x": 0.0, "y": 0.0, "z": 0.0}
        p_mid = {"x": 1.0, "y": 0.0, "z": 0.0}
        p_tip_90 = {"x": 1.0, "y": 1.0, "z": 0.0}  # Góc vuông 90 độ
        angle_90 = compute_joint_angle_deg(p_base, p_mid, p_tip_90)
        self.assertAlmostEqual(angle_90, 90.0, places=2)

        # Chuỗi góc khớp chuẩn công thái học (142 độ) và cổ tay êm phải đạt điểm cao >= 90
        good_angles = [142.0] * 30
        good_tips = [[0.2 + 0.02 * np.sin(i), 0.3] for i in range(30)]
        good_wrist = [0.5] * 30
        standalone_score = score_finger_biomechanics(good_angles, good_tips, good_wrist)
        self.assertGreaterEqual(standalone_score, 90.0)

    def test_sheet_music_note_normalization_and_semitone(self) -> None:
        """Kiểm tra chuyển đổi ký hiệu nốt trên Sheet nhạc và tính khoảng cách bán cung (MIDI)."""
        self.assertEqual(normalize_note_code("Đô (C4 - Đô Trung)"), "C4")
        self.assertEqual(normalize_note_code("Sol (G4)"), "G4")
        # Khoảng cách từ C4 đến G4 là đúng 7 bán cung (Quãng 5 đúng)
        self.assertEqual(note_code_to_semitone("G4") - note_code_to_semitone("C4"), 7)


if __name__ == "__main__":
    unittest.main()
