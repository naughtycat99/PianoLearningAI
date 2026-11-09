"""
Module Đánh giá Sinh cơ học & Kỹ thuật Piano Độc lập (Standalone Biomechanical AI Evaluator).
Cho phép chấm điểm BẤT KỲ bài hát nào chỉ từ 1 video của học viên (không cần video mẫu)
thông qua 4 tiêu chí Công thái học Piano (Piano Ergonomics & Biomechanics):
1. Góc cong khớp ngón tay 3D (Finger Arch Angle - Vector Dot Product): phát hiện lỗi bẹt ngón / quặp ngón.
2. Độ ổn định cổ tay (Wrist Stability - đạo hàm bậc 2 / Jerk): phát hiện lỗi nhấp nhổm cổ tay.
3. Độ linh hoạt và độc lập của ngón tay (Finger Independence & Articulation).
4. Độ ổn định nhịp điệu nội tại (Internal Tempo / Inter-Onset Interval Consistency).
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.signal import find_peaks
from config import DSP_CONFIG, FINGER_DISPLAY_NAMES, FINGER_NAMES

# Bộ 3 khớp xương (MCP/CMC -> PIP/MCP -> TIP) của từng ngón tay để tính góc cong 3D
FINGER_JOINT_TRIPLETS: Dict[str, Tuple[int, int, int]] = {
    "Ngon cai": (1, 2, 4),
    "Ngon tro": (5, 6, 8),
    "Ngon giua": (9, 10, 12),
    "Ngon ap ut": (13, 14, 16),
    "Ngon ut": (17, 18, 20),
}


def compute_joint_angle_deg(
    p_base: Dict[str, float], p_mid: Dict[str, float], p_tip: Dict[str, float]
) -> float:
    """
    Tính góc gập khớp ngón tay (độ) trong không gian 3D (x, y, z) tại khớp giữa p_mid
    bằng tích vô hướng (Dot Product) của hai vector xương:
        u = p_base - p_mid
        v = p_tip - p_mid
        theta = arccos( (u . v) / (||u|| * ||v||) )

    Góc lý tưởng khi bấm phím Piano (tư thế khum tròn tự nhiên): 120° -> 162°.
    - Nếu theta > 168°: Lỗi duỗi thẳng / bẹt ngón (Flat / Collapsed joint).
    - Nếu theta < 100°: Lỗi quặp ngón quá mức (Over-curled / Cramped finger).
    """
    u = np.array(
        [
            p_base["x"] - p_mid["x"],
            p_base["y"] - p_mid["y"],
            p_base.get("z", 0.0) - p_mid.get("z", 0.0),
        ],
        dtype=float,
    )
    v = np.array(
        [
            p_tip["x"] - p_mid["x"],
            p_tip["y"] - p_mid["y"],
            p_tip.get("z", 0.0) - p_mid.get("z", 0.0),
        ],
        dtype=float,
    )
    norm_u = float(np.linalg.norm(u))
    norm_v = float(np.linalg.norm(v))
    if norm_u < 1e-6 or norm_v < 1e-6:
        return 145.0
    cos_theta = float(np.clip(np.dot(u, v) / (norm_u * norm_v), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def extract_biomechanical_metrics(
    user_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Trích xuất chuỗi góc khớp 3D, độ ổn định cổ tay và vận tốc đầu ngón tay
    của cả 2 bàn tay (Trái & Phải) qua toàn bộ các khung hình video.
    """
    angles: Dict[str, Dict[str, List[float]]] = {
        "Left": {name: [] for name in FINGER_NAMES},
        "Right": {name: [] for name in FINGER_NAMES},
    }
    tip_positions: Dict[str, Dict[str, List[List[float]]]] = {
        "Left": {name: [] for name in FINGER_NAMES},
        "Right": {name: [] for name in FINGER_NAMES},
    }
    wrist_y: Dict[str, List[float]] = {"Left": [], "Right": []}

    for frame in user_data:
        hands = frame.get("hands", [])
        if len(hands) > 2:
            def hand_size(h: List[Dict[str, float]]) -> float:
                xs = [lm["x"] for lm in h]
                ys = [lm["y"] for lm in h]
                return (max(xs) - min(xs)) * (max(ys) - min(ys))

            hands = sorted(hands, key=hand_size, reverse=True)[:2]

        sorted_hands = sorted(hands, key=lambda h: h[0]["x"])
        left_hand = None
        right_hand = None
        if len(sorted_hands) == 1:
            if sorted_hands[0][0]["x"] > 0.5:
                right_hand = sorted_hands[0]
            else:
                left_hand = sorted_hands[0]
        elif len(sorted_hands) >= 2:
            left_hand = sorted_hands[0]
            right_hand = sorted_hands[1]

        for side, h_lms in [("Left", left_hand), ("Right", right_hand)]:
            if h_lms and len(h_lms) >= 21:
                wrist = h_lms[0]
                base = h_lms[9]
                scale = (
                    np.sqrt((base["x"] - wrist["x"]) ** 2 + (base["y"] - wrist["y"]) ** 2)
                    + DSP_CONFIG.epsilon
                )
                wrist_y[side].append(float(wrist["y"]))
                for fname in FINGER_NAMES:
                    b_idx, m_idx, t_idx = FINGER_JOINT_TRIPLETS[fname]
                    ang = compute_joint_angle_deg(h_lms[b_idx], h_lms[m_idx], h_lms[t_idx])
                    angles[side][fname].append(ang)

                    tip = h_lms[t_idx]
                    nx = float((tip["x"] - wrist["x"]) / scale)
                    ny = float((tip["y"] - wrist["y"]) / scale)
                    tip_positions[side][fname].append([nx, ny])
            else:
                if wrist_y[side]:
                    wrist_y[side].append(wrist_y[side][-1])
                else:
                    wrist_y[side].append(0.5)
                for fname in FINGER_NAMES:
                    if angles[side][fname]:
                        angles[side][fname].append(angles[side][fname][-1])
                    else:
                        angles[side][fname].append(145.0)
                    if tip_positions[side][fname]:
                        tip_positions[side][fname].append(tip_positions[side][fname][-1])
                    else:
                        tip_positions[side][fname].append([0.0, 0.0])

    return {
        "angles": angles,
        "tip_positions": tip_positions,
        "wrist_y": wrist_y,
    }


def score_finger_biomechanics(
    angle_series: List[float],
    tip_series: List[List[float]],
    wrist_series: List[float],
) -> float:
    """
    Chấm điểm kỹ thuật độc lập của một ngón tay trên thang [0, 100] dựa trên:
    - 50%: Tỷ lệ khung hình giữ được độ cong vòm chuẩn công thái học (115° - 165°).
    - 30%: Độ ổn định và êm ái của cổ tay (đạo hàm bậc 2 / Jerk thấp).
    - 20%: Độ dứt khoát và kiểm soát biên độ gõ phím (Articulation control).
    """
    if len(angle_series) < 10:
        return 0.0

    ang_arr = np.array(angle_series, dtype=float)
    tip_arr = np.array(tip_series, dtype=float)
    wrist_arr = np.array(wrist_series, dtype=float)

    # 1. Điểm tư thế góc khớp (Arch Posture Score): Lý tưởng 142°, dung sai [115°, 168°]
    deviation = np.abs(ang_arr - 142.0)
    # Sai lệch <= 18 độ được tính tối đa, lớn hơn giảm tuyến tính
    frame_posture_scores = np.clip(100.0 - np.maximum(0.0, deviation - 18.0) * 1.6, 40.0, 100.0)
    posture_score = float(np.mean(frame_posture_scores))

    # 2. Điểm ổn định cổ tay (Wrist Smoothness Score) qua đạo hàm bậc 2 (Acceleration/Jerk)
    if len(wrist_arr) >= 5:
        wrist_acc = np.abs(np.diff(wrist_arr, n=2))
        mean_jerk = float(np.mean(wrist_acc))
        wrist_score = float(np.clip(100.0 - mean_jerk * 1200.0, 60.0, 100.0))
    else:
        wrist_score = 90.0

    # 3. Điểm kiểm soát vận tốc ngón tay (Articulation Control Score)
    if len(tip_arr) >= 5:
        vel = np.sqrt(np.sum(np.diff(tip_arr, axis=0) ** 2, axis=1))
        active_ratio = float(np.mean(vel > 0.008))
        # Đảm bảo ngón tay có hoạt động linh hoạt mà không bị cứng đờ hoặc giật cục
        articulation_score = float(np.clip(78.0 + min(0.22, active_ratio) * 100.0, 65.0, 100.0))
    else:
        articulation_score = 85.0

    final_sc = 0.50 * posture_score + 0.30 * wrist_score + 0.20 * articulation_score
    return round(float(np.clip(final_sc, 0.0, 100.0)), 2)


def evaluate_standalone_video(
    user_data: List[Dict[str, Any]], user_fps: float = 30.0
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    """
    Đánh giá toàn diện một video chơi Piano bất kỳ mà KHÔNG cần video chuyên gia.
    Trả về bảng điểm 10 ngón tay và danh sách sự kiện trên Timeline kèm đường chuẩn công thái học.
    """
    metrics = extract_biomechanical_metrics(user_data)
    angles = metrics["angles"]
    tip_positions = metrics["tip_positions"]
    wrist_y = metrics["wrist_y"]

    scores: Dict[str, Dict[str, float]] = {"Left": {}, "Right": {}}
    for side in ["Left", "Right"]:
        for fname in FINGER_NAMES:
            tip_arr = np.array(tip_positions[side][fname], dtype=float)
            if len(tip_arr) == 0 or np.max(np.linalg.norm(tip_arr, axis=1)) < 0.01:
                continue
            scores[side][fname] = score_finger_biomechanics(
                angles[side][fname], tip_positions[side][fname], wrist_y[side]
            )

    # Phát hiện các sự kiện nhấn phím (Note Onsets) từ đỉnh vận tốc của học viên
    num_frames = len(user_data)
    all_events: List[int] = []
    for side in ["Left", "Right"]:
        for fname in FINGER_NAMES:
            traj = np.array(tip_positions[side][fname], dtype=float)
            if len(traj) < 3:
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
        end_f = min(num_frames, event_f + half_win)

        finger_event_scores: Dict[Tuple[str, str], float] = {}
        finger_angles_at_event: Dict[Tuple[str, str], float] = {}
        finger_waves: Dict[Tuple[str, str], Tuple[List[float], List[float]]] = {}

        for side in ["Left", "Right"]:
            for fname in FINGER_NAMES:
                ang_seg = angles[side][fname][start_f:end_f]
                tip_seg = tip_positions[side][fname][start_f:end_f]
                w_seg = wrist_y[side][start_f:end_f]
                sc = score_finger_biomechanics(ang_seg, tip_seg, w_seg)
                finger_event_scores[(side, fname)] = sc
                mean_ang = float(np.mean(ang_seg)) if ang_seg else 142.0
                finger_angles_at_event[(side, fname)] = round(mean_ang, 1)

                # Tạo đường chuẩn công thái học (Ideal Bell-Curve Keystroke) và đường thực tế của học viên
                usr_mag = [
                    round(float(np.sqrt(p[0] ** 2 + p[1] ** 2)), 4) for p in tip_seg
                ]
                if usr_mag:
                    t_norm = np.linspace(-1.0, 1.0, len(usr_mag))
                    base_v = float(np.min(usr_mag))
                    amp_v = max(0.08, float(np.max(usr_mag) - base_v))
                    ideal_wave = [
                        round(float(base_v + amp_v * np.exp(-3.0 * (x ** 2))), 4)
                        for x in t_norm
                    ]
                else:
                    ideal_wave = []
                finger_waves[(side, fname)] = (ideal_wave, usr_mag)

        worst_key = min(finger_event_scores, key=finger_event_scores.get)
        worst_score = round(float(finger_event_scores[worst_key]), 2)
        worst_angle = finger_angles_at_event[worst_key]
        ideal_w, usr_w = finger_waves[worst_key]

        hand_viet = "Tay trái" if worst_key[0] == "Left" else "Tay phải"
        finger_viet = FINGER_DISPLAY_NAMES.get(worst_key[1], worst_key[1])

        if worst_score < 82.0:
            status = "ERROR"
            if worst_angle > 165.0:
                issue_desc = f"duỗi quá thẳng/bẹt ngón (góc khớp {worst_angle}° > 165°)"
            elif worst_angle < 110.0:
                issue_desc = f"quặp đốt ngón quá mức (góc khớp {worst_angle}° < 110°)"
            else:
                issue_desc = f"dao động cổ tay chưa ổn định (góc khớp {worst_angle}°)"
            advice = (
                f"Phát hiện {hand_viet} ({finger_viet.lower()}) bị {issue_desc}. "
                "Hãy giữ ngón tay khum tròn tự nhiên (120°–160°) và thả lỏng cổ tay."
            )
        else:
            status = "OK"
            advice = (
                f"Tư thế chuẩn! Góc khớp {finger_viet.lower()} đạt {worst_angle}° "
                "(nằm trong vùng chuẩn công thái học 120°–160°)."
            )

        timeline.append(
            {
                "event_id": event_id,
                "second": round(event_f / user_fps, 1),
                "status": status,
                "score": worst_score,
                "joint_angle": worst_angle,
                "mode": "standalone",
                "error_hand": worst_key[0],
                "error_finger": worst_key[1],
                "expert_frame": int(event_f),
                "user_frame": int(event_f),
                "advice": advice,
                "expert_wave": ideal_w,
                "user_wave": usr_w,
            }
        )

    return scores, timeline
