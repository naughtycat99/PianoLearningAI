"""
Module đánh giá chi tiết từng sự kiện gõ phím (Note Onset) trên dòng thời gian (Timeline).
Sử dụng thuật toán Dynamic Time Warping (DTW) có cửa sổ ràng buộc (Sakoe-Chiba Band)
kết hợp phát hiện đỉnh vận tốc (SciPy Peak Detection) để định vị chính xác thời điểm mắc lỗi.
"""

from typing import Any, Dict, List
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from config import DSP_CONFIG, FINGER_NAMES


def dtw_align_sequences(
    expert_feats: np.ndarray, user_feats: np.ndarray
) -> Dict[int, int]:
    """
    Căn chỉnh phi tuyến tính hai chuỗi thời gian đa chiều (Chuyên gia và Học viên)
    bằng thuật toán Quy hoạch động Dynamic Time Warping (DTW).

    Args:
        expert_feats: Ma trận đặc trưng chuyển động của chuyên gia (N x D).
        user_feats: Ma trận đặc trưng chuyển động của học viên (M x D).

    Returns:
        Từ điển ánh xạ chỉ số khung hình của học viên sang khung hình chuyên gia tương ứng.
    """
    n_full = len(expert_feats)
    m_full = len(user_feats)
    stride = 4 if (n_full > 400 and m_full > 400) else 1

    exp_sub = expert_feats[::stride]
    usr_sub = user_feats[::stride]
    n = len(exp_sub)
    m = len(usr_sub)
    window = max(30, abs(n - m) + 20)

    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0
    parent_matrix = np.zeros((n + 1, m + 1, 2), dtype=int)

    for i in range(1, n + 1):
        left_bound = max(1, i - window)
        right_bound = min(m, i + window)
        for j in range(left_bound, right_bound + 1):
            cost = float(np.linalg.norm(exp_sub[i - 1] - usr_sub[j - 1]))
            choices = [
                (dtw_matrix[i - 1, j], (i - 1, j)),
                (dtw_matrix[i, j - 1], (i, j - 1)),
                (dtw_matrix[i - 1, j - 1], (i - 1, j - 1)),
            ]
            min_val, parent = min(choices, key=lambda x: x[0])
            dtw_matrix[i, j] = cost + min_val
            parent_matrix[i, j] = parent

    path = []
    curr = (n, m)
    while curr != (0, 0):
        path.append(curr)
        curr = tuple(parent_matrix[curr[0], curr[1]])
        if curr == (0, 0) or dtw_matrix[curr[0], curr[1]] == np.inf:
            break
    path.reverse()

    alignment: Dict[int, int] = {}
    for p in path:
        user_idx = (p[1] - 1) * stride
        expert_idx = (p[0] - 1) * stride
        if user_idx >= 0 and expert_idx >= 0:
            alignment[min(m_full - 1, user_idx)] = min(n_full - 1, expert_idx)
    alignment[0] = 0
    alignment[m_full - 1] = n_full - 1

    user_indices = sorted(list(alignment.keys()))
    if len(user_indices) > 1:
        user_all = np.arange(m_full)
        expert_values = [alignment[u] for u in user_indices]
        interp_func = interp1d(
            user_indices, expert_values, kind="linear", fill_value="extrapolate"
        )
        interpolated_experts = interp_func(user_all)
        for f in range(m_full):
            alignment[f] = max(
                0, min(n_full - 1, int(round(float(interpolated_experts[f]))))
            )

    return alignment


def get_hand_features(
    features: Dict[str, Dict[str, np.ndarray]]
) -> np.ndarray:
    """Gộp đặc trưng tọa độ của cả 10 ngón tay thành một ma trận phẳng (N x 20)."""
    full_feats = []
    for hand_side in ["Left", "Right"]:
        for name in FINGER_NAMES:
            full_feats.append(features[hand_side][name])
    return np.hstack(full_feats)


def score_aligned_finger_segment(
    expert_seg: np.ndarray, user_seg: np.ndarray
) -> float:
    """
    Chấm điểm một phân đoạn cử động ngón tay ngắn quanh thời điểm gõ phím
    kết hợp khoảng cách Euclid chuẩn hóa Z-Score (60%) và tương quan Pearson (40%).
    """
    if len(expert_seg) < 3 or len(user_seg) < 3:
        return 100.0

    expert_range = np.max(expert_seg, axis=0) - np.min(expert_seg, axis=0)
    user_range = np.max(user_seg, axis=0) - np.min(user_seg, axis=0)

    expert_moving = bool(np.max(expert_range) > DSP_CONFIG.segment_motion_threshold)
    user_moving = bool(np.max(user_range) > DSP_CONFIG.segment_motion_threshold)

    if not expert_moving and not user_moving:
        return 100.0
    if expert_moving != user_moving:
        return 62.0

    expert_mean = np.mean(expert_seg, axis=0)
    expert_std = np.std(expert_seg, axis=0) + DSP_CONFIG.epsilon
    user_mean = np.mean(user_seg, axis=0)
    user_std = np.std(user_seg, axis=0) + DSP_CONFIG.epsilon

    expert_norm = (expert_seg - expert_mean) / expert_std
    user_norm = (user_seg - user_mean) / user_std

    dists = np.sqrt(np.sum((expert_norm - user_norm) ** 2, axis=1))
    avg_dist = float(np.mean(dists))

    dist_score = max(0.0, (3.2 - avg_dist) / 3.2 * 100.0)

    corr_x_full = np.correlate(expert_norm[:, 0], user_norm[:, 0], mode="full") / len(user_norm)
    corr_y_full = np.correlate(expert_norm[:, 1], user_norm[:, 1], mode="full") / len(user_norm)
    corr_x = float(np.max(corr_x_full))
    corr_y = float(np.max(corr_y_full))

    if np.isnan(corr_x):
        corr_x = 1.0
    if np.isnan(corr_y):
        corr_y = 1.0

    corr_score = min(1.0, max(-1.0, (corr_x + corr_y) / 2.0))
    corr_score_mapped = (corr_score + 1.0) / 2.0 * 100.0

    final_score = 0.45 * dist_score + 0.55 * corr_score_mapped
    return float(max(0.0, min(100.0, final_score)))


def generate_timeline_report(
    expert_data: List[Dict[str, Any]],
    user_data: List[Dict[str, Any]],
    expert_fps: float = 30.0,
    user_fps: float = 30.0,
) -> List[Dict[str, Any]]:
    """
    Tạo danh sách báo cáo từng sự kiện nhấn phím trên Timeline, kèm theo tọa độ sóng
    (waveform) để giao diện GUI vẽ biểu đồ đối chiếu quỹ đạo Chuyên gia vs. Học viên.
    """
    from evaluator import extract_normalized_features

    expert_feats = extract_normalized_features(expert_data)
    user_feats = extract_normalized_features(user_data)

    expert_flat = get_hand_features(expert_feats)
    user_flat = get_hand_features(user_feats)

    alignment = dtw_align_sequences(expert_flat, user_flat)
    num_user_frames = len(user_data)

    # 1. Phát hiện Note Onset (Thời điểm gõ phím) qua đỉnh vận tốc tức thời
    all_events: List[int] = []
    for hand_side in ["Left", "Right"]:
        for name in FINGER_NAMES:
            traj = user_feats[hand_side][name]
            if len(traj) < 2:
                continue
            vel = np.sqrt(np.sum(np.diff(traj, axis=0) ** 2, axis=1))
            vel_smooth = np.convolve(vel, np.ones(5) / 5.0, mode="same")
            peaks, _ = find_peaks(
                vel_smooth,
                height=DSP_CONFIG.onset_peak_height,
                distance=DSP_CONFIG.onset_peak_distance,
            )
            for p in peaks:
                all_events.append(int(p))

    # 2. Gom cụm các sự kiện xảy ra đồng thời (Hợp âm / Chords)
    all_events.sort()
    clustered_events: List[int] = []
    if all_events:
        curr = [all_events[0]]
        for e in all_events[1:]:
            if e - curr[-1] < DSP_CONFIG.chord_cluster_frame_gap:
                curr.append(e)
            else:
                clustered_events.append(int(np.mean(curr)))
                curr = [e]
        clustered_events.append(int(np.mean(curr)))

    timeline: List[Dict[str, Any]] = []
    half_win = DSP_CONFIG.event_window_half_size

    for event_id, event_f in enumerate(clustered_events, start=1):
        start_f = max(0, event_f - half_win)
        end_f = min(num_user_frames, event_f + half_win)

        finger_scores: Dict[tuple, float] = {}
        finger_waves: Dict[tuple, tuple] = {}

        for hand_side in ["Left", "Right"]:
            for name in FINGER_NAMES:
                user_seg = user_feats[hand_side][name][start_f:end_f]
                expert_seg = np.array(
                    [
                        expert_feats[hand_side][name][alignment.get(f, min(f, len(expert_data) - 1))]
                        for f in range(start_f, end_f)
                    ]
                )
                score = score_aligned_finger_segment(expert_seg, user_seg)
                finger_scores[(hand_side, name)] = score

                # Lưu biên độ khoảng cách 2D để vẽ biểu đồ sóng trên GUI
                exp_mag = np.sqrt(np.sum(expert_seg ** 2, axis=1)).tolist()
                usr_mag = np.sqrt(np.sum(user_seg ** 2, axis=1)).tolist()
                finger_waves[(hand_side, name)] = (
                    [round(float(v), 4) for v in exp_mag],
                    [round(float(v), 4) for v in usr_mag],
                )

        worst_key = min(finger_scores, key=finger_scores.get)
        avg_event_score = round(float(np.mean(list(finger_scores.values()))), 2)
        exp_wave, usr_wave = finger_waves[worst_key]

        if avg_event_score < DSP_CONFIG.error_score_threshold:
            status = "ERROR"
            hand_viet = "Tay trái" if worst_key[0] == "Left" else "Tay phải"
            finger_viet = worst_key[1].lower()
            advice = (
                f"Lỗi ở {hand_viet.lower()} ({finger_viet}). "
                "Cần chú ý cử động đúng nhịp và khớp biên độ với chuyên gia."
            )
        else:
            status = "OK"
            advice = "Đạt yêu cầu! Tư thế và nhịp độ ngón tay rất tốt."

        timeline.append(
            {
                "event_id": event_id,
                "second": round(event_f / user_fps, 1),
                "status": status,
                "score": avg_event_score,
                "error_hand": worst_key[0],
                "error_finger": worst_key[1],
                "expert_frame": int(alignment.get(event_f, 0)),
                "user_frame": int(event_f),
                "advice": advice,
                "expert_wave": exp_wave,
                "user_wave": usr_wave,
            }
        )

    return timeline
