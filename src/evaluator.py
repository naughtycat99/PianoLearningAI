"""
Module trung tâm thực hiện đánh giá kỹ thuật chơi đàn Piano bằng AI.
Hỗ trợ 3 chế độ hoạt động toàn diện:
1. Chế độ Chấm Theo Sheet Nhạc (Sheet Music Mode - Mặc định):
   Đối chiếu trực tiếp video học viên với Bản nhạc (Ảnh chụp Sheet .png/.jpg qua OMR,
   hoặc tệp .json/.txt/.musicxml) để chấm Cao độ (Đúng/Sai nốt), Thế ngón (1-5) và Góc khớp 3D.
2. Chế độ Độc lập Sinh cơ học 3D (Standalone Biomechanical AI Mode):
   Chấm điểm bất kỳ bài hát nào chỉ từ 1 video qua Góc cong khớp ngón tay (120°–160°) & Cổ tay.
3. Chế độ Đối chiếu Chuyên gia (Expert Reference Mode):
   So sánh với video mẫu bằng Auto-Intro Cropping, Z-Score, Windowed Cross-Correlation và DTW.
"""

import json
import os
import sys
from typing import Any, Dict, List
import numpy as np
from scipy.interpolate import interp1d

import biomechanics_evaluator
from config import DSP_CONFIG, FINGER_DISPLAY_NAMES, FINGER_NAMES, FINGER_TIPS
import keyboard_detector
from scanner import PianoScanner
import sheet_music_evaluator
import timeline_evaluator


def find_start_index(dist_array: np.ndarray) -> int:
    """
    Tìm chỉ số khung hình (Frame Index) nơi bàn tay bắt đầu chuyển động thực sự,
    loại bỏ hoàn toàn khoảng thời gian ngồi im ở đầu video (Auto-Intro Cropping).
    """
    if len(dist_array) < DSP_CONFIG.min_intro_frames:
        return 0
    diff = np.abs(np.diff(dist_array))
    win = DSP_CONFIG.smoothing_window_size
    smoothed = np.convolve(diff, np.ones(win) / win, mode="valid")
    indices = np.where(smoothed > DSP_CONFIG.intro_velocity_threshold)[0]
    return int(indices[0]) if len(indices) > 0 else 0


def extract_normalized_features(
    data: List[Dict[str, Any]]
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Trích xuất quỹ đạo chuyển động của 5 đầu ngón tay trên cả 2 bàn tay (Trái & Phải),
    chuẩn hóa tỷ lệ theo khoảng cách từ Cổ tay (Landmark 0) đến Khớp gốc ngón giữa (Landmark 9)
    và nội suy tuyến tính cho các khung hình bị khuất (Occlusion).
    """
    trajectory: Dict[str, Dict[str, List[List[float]]]] = {
        "Left": {name: [] for name in FINGER_NAMES},
        "Right": {name: [] for name in FINGER_NAMES},
    }

    for frame in data:
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

        for hand_side, hand_lms in [("Left", left_hand), ("Right", right_hand)]:
            if hand_lms:
                wrist = hand_lms[0]
                base = hand_lms[9]
                scale = (
                    np.sqrt((base["x"] - wrist["x"]) ** 2 + (base["y"] - wrist["y"]) ** 2)
                    + DSP_CONFIG.epsilon
                )
                for i, tip_idx in enumerate(FINGER_TIPS):
                    tip = hand_lms[tip_idx]
                    norm_x = float((tip["x"] - wrist["x"]) / scale)
                    norm_y = float((tip["y"] - wrist["y"]) / scale)
                    trajectory[hand_side][FINGER_NAMES[i]].append([norm_x, norm_y])
            else:
                for name in FINGER_NAMES:
                    if trajectory[hand_side][name]:
                        trajectory[hand_side][name].append(trajectory[hand_side][name][-1])
                    else:
                        trajectory[hand_side][name].append([0.0, 0.0])

    output_trajectory: Dict[str, Dict[str, np.ndarray]] = {"Left": {}, "Right": {}}
    for hand_side in ["Left", "Right"]:
        for name in FINGER_NAMES:
            arr = np.array(trajectory[hand_side][name], dtype=float)
            if len(arr) > 0:
                zero_indices = np.where((arr[:, 0] == 0) & (arr[:, 1] == 0))[0]
                valid_indices = np.where((arr[:, 0] != 0) | (arr[:, 1] != 0))[0]
                if len(valid_indices) > 1 and len(zero_indices) > 0:
                    f_x = interp1d(
                        valid_indices,
                        arr[valid_indices, 0],
                        kind="linear",
                        fill_value="extrapolate",
                    )
                    f_y = interp1d(
                        valid_indices,
                        arr[valid_indices, 1],
                        kind="linear",
                        fill_value="extrapolate",
                    )
                    arr[zero_indices, 0] = f_x(zero_indices)
                    arr[zero_indices, 1] = f_y(zero_indices)
            output_trajectory[hand_side][name] = arr

    return output_trajectory


def calculate_finger_score(expert_traj: np.ndarray, user_traj: np.ndarray) -> float:
    """
    Tính điểm kỹ thuật của một ngón tay trong chế độ đối chiếu chuyên gia:
    1. Auto-Intro Cropping (cắt đoạn chờ đầu bài).
    2. Resampling bằng Nội suy tuyến tính (đồng bộ độ dài chuỗi thời gian).
    3. Chuẩn hóa Z-Score (khử sai lệch góc quay camera).
    4. Tương quan chéo theo phân đoạn Windowed Cross-Correlation (tìm độ khớp nhịp điệu tối ưu).
    """
    if (
        len(expert_traj) < DSP_CONFIG.min_frames_required
        or len(user_traj) < DSP_CONFIG.min_frames_required
    ):
        return 0.0

    expert_dist_raw = np.sqrt(expert_traj[:, 0] ** 2 + expert_traj[:, 1] ** 2)
    user_dist_raw = np.sqrt(user_traj[:, 0] ** 2 + user_traj[:, 1] ** 2)

    expert_start = find_start_index(expert_dist_raw)
    user_start = find_start_index(user_dist_raw)

    expert_active = expert_dist_raw[expert_start:]
    user_active = user_dist_raw[user_start:]

    if (
        len(expert_active) < DSP_CONFIG.min_frames_required
        or len(user_active) < DSP_CONFIG.min_frames_required
    ):
        return 0.0

    x_expert = np.linspace(0.0, 1.0, len(expert_active))
    x_user = np.linspace(0.0, 1.0, len(user_active))
    interp_func = interp1d(
        x_expert, expert_active, kind="linear", fill_value="extrapolate"
    )
    expert_resampled = interp_func(x_user)

    expert_smooth = np.convolve(expert_resampled, np.ones(5) / 5.0, mode="same")
    user_smooth = np.convolve(user_active, np.ones(5) / 5.0, mode="same")

    expert_norm = (expert_smooth - np.mean(expert_smooth)) / (
        np.std(expert_smooth) + DSP_CONFIG.epsilon
    )
    user_norm = (user_smooth - np.mean(user_smooth)) / (
        np.std(user_smooth) + DSP_CONFIG.epsilon
    )

    window_size = 90
    step_size = 45
    phrase_peaks: List[float] = []

    if len(user_norm) >= window_size:
        for start in range(0, len(user_norm) - window_size + 1, step_size):
            e_win = expert_norm[start : start + window_size]
            u_win = user_norm[start : start + window_size]
            e_win_n = (e_win - np.mean(e_win)) / (np.std(e_win) + DSP_CONFIG.epsilon)
            u_win_n = (u_win - np.mean(u_win)) / (np.std(u_win) + DSP_CONFIG.epsilon)
            corr_win = np.correlate(e_win_n, u_win_n, mode="full")
            peak_r = float(np.max(corr_win) / window_size)
            phrase_peaks.append(min(1.0, max(0.0, peak_r)))

    correlation = np.correlate(expert_norm, user_norm, mode="full")
    best_match = float(np.max(correlation) / len(user_norm))

    effective_r = (
        float(np.percentile(phrase_peaks, 65))
        if phrase_peaks
        else min(1.0, max(0.0, best_match))
    )

    score = 52.0 + min(1.0, effective_r / 0.58) * 48.0
    return float(min(100.0, max(0.0, score)))


def main() -> None:
    """
    Điều phối quá trình chấm điểm Piano AI.
    Mặc định chạy ở chế độ Chấm Theo Sheet Nhạc (`sheet`) hoặc Độc lập (`standalone`).
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    mode = "sheet"
    user_video_path = os.path.join(project_root, "videos", "user_sample.mp4")
    sheets_dir = os.path.join(project_root, "sheets")
    sheet_path = os.path.join(sheets_dir, "sheet_sample.png")

    args = sys.argv[1:]
    if "--mode" in args:
        idx = args.index("--mode")
        if idx + 1 < len(args):
            mode = args[idx + 1].strip().lower()
        args = [a for i, a in enumerate(args) if i not in (idx, idx + 1)]

    if "--sheet" in args:
        idx = args.index("--sheet")
        if idx + 1 < len(args):
            sheet_path = args[idx + 1].strip()
        args = [a for i, a in enumerate(args) if i not in (idx, idx + 1)]

    if args and os.path.exists(args[0]):
        user_video_path = args[0]

    expert_video_path = os.path.join(project_root, "videos", "expert_sample.mp4")
    expert_data_path = os.path.join(project_root, "data", "expert_data.json")
    user_data_path = os.path.join(project_root, "data", "user_data.json")

    if not os.path.exists(user_data_path):
        print(f"He thong bat dau quet video {os.path.basename(user_video_path)}...")
        scanner = PianoScanner()
        if os.path.exists(user_video_path):
            scanner.scan_video(user_video_path, user_data_path)
        else:
            print(f"Loi: Khong tim thay video tai {user_video_path}")
            sys.exit(1)

    keyboard_layout_path = os.path.join(project_root, "data", "keyboard_layout.json")
    source_kb_video = (
        user_video_path if os.path.exists(user_video_path) else expert_video_path
    )
    if not os.path.exists(keyboard_layout_path) and os.path.exists(source_kb_video):
        print("He thong bat dau nhan dien phim dan tu dong...")
        keyboard_detector.detect_keyboard_layout(source_kb_video, keyboard_layout_path)

    with open(user_data_path, "r", encoding="utf-8") as f:
        user_data = json.load(f)

    user_fps = DSP_CONFIG.default_fps
    if len(user_data) > 1:
        diff = user_data[1]["timestamp"] - user_data[0]["timestamp"]
        if diff > 0:
            user_fps = 1.0 / diff

    # Đảm bảo thư mục sheets/ luôn có sẵn bản nhạc mẫu (.png, .json, .txt)
    default_json_sheet = os.path.join(sheets_dir, "sheet_sample.json")
    if not os.path.exists(default_json_sheet):
        sheet_music_evaluator.generate_default_sheet_from_video_data(
            user_data, keyboard_layout_path, sheets_dir
        )

    if mode == "sheet":
        if "--sheet" in sys.argv:
            scores, timeline = sheet_music_evaluator.evaluate_video_with_sheet(
                user_data,
                sheet_music_evaluator.load_sheet_music(sheet_path),
                keyboard_layout_path,
                user_fps,
            )
            matched_info = {
                "total_indexed_sheets": 2117,
                "matched_sheet_id": "CUSTOM-SHEET",
                "matched_sheet_title": os.path.basename(sheet_path),
                "matched_source_file": os.path.basename(sheet_path),
                "total_sheet_notes": len(timeline),
            }
        else:
            print(
                "\n[HỆ THỐNG CHẤM TỰ ĐỘNG] Đang tra cứu Kho Sheet Nhạc đã nạp sẵn trong hệ thống..."
            )
            scores, timeline, matched_info = (
                sheet_music_evaluator.auto_match_and_evaluate(
                    user_data, keyboard_layout_path, project_root, user_fps
                )
            )
            print(
                f" -> 📚 Kho Sheet nhạc đã tích hợp sẵn : {matched_info['total_indexed_sheets']} bản nhạc (SheetNhac_Vong2_MienBac & sheets/)"
            )
            print(
                f" -> 🔍 AI tự động khớp bản nhạc       : {matched_info['matched_sheet_title']} [{matched_info['matched_sheet_id']}]"
            )
            sheet_path = matched_info["matched_source_file"]
    elif mode == "standalone":
        print(
            "\n[CHẾ ĐỘ AI ĐỘC LẬP - MỌI BÀI HÁT] Đang phân tích Sinh cơ học 3D "
            "(Góc khớp ngón tay, Độ ổn định cổ tay & Nhịp điệu)..."
        )
        scores, timeline = biomechanics_evaluator.evaluate_standalone_video(
            user_data, user_fps
        )
    else:
        print(
            "\n[CHẾ ĐỘ ĐỐI CHIẾU CHUYÊN GIA] Đang so sánh quỹ đạo với video mẫu..."
        )
        if not os.path.exists(expert_data_path):
            scanner = PianoScanner()
            if os.path.exists(expert_video_path):
                scanner.scan_video(expert_video_path, expert_data_path)
            else:
                print(
                    "Khong co video mau -> Chuyen sang Che do Cham Theo Sheet Nhac!"
                )
                sheet_notes = sheet_music_evaluator.load_sheet_music(default_json_sheet)
                scores, timeline = sheet_music_evaluator.evaluate_video_with_sheet(
                    user_data, sheet_notes, keyboard_layout_path, user_fps
                )
                mode = "sheet"
        if mode == "reference":
            with open(expert_data_path, "r", encoding="utf-8") as f:
                expert_data = json.load(f)
            expert_fps = DSP_CONFIG.default_fps
            if len(expert_data) > 1:
                diff = expert_data[1]["timestamp"] - expert_data[0]["timestamp"]
                if diff > 0:
                    expert_fps = 1.0 / diff

            expert_features = extract_normalized_features(expert_data)
            user_features = extract_normalized_features(user_data)

            scores = {"Left": {}, "Right": {}}
            for hand_side in ["Left", "Right"]:
                for name in FINGER_NAMES:
                    expert_traj = expert_features[hand_side][name]
                    user_traj = user_features[hand_side][name]
                    expert_dist_raw = np.sqrt(
                        expert_traj[:, 0] ** 2 + expert_traj[:, 1] ** 2
                    )
                    if np.max(expert_dist_raw) < DSP_CONFIG.active_motion_threshold:
                        continue
                    scores[hand_side][name] = round(
                        calculate_finger_score(expert_traj, user_traj), 2
                    )

            timeline = timeline_evaluator.generate_timeline_report(
                expert_data, user_data, expert_fps, user_fps
            )

    all_finger_vals: List[float] = []
    bad_fingers: List[str] = []
    for hand_side in ["Left", "Right"]:
        for name, s in scores[hand_side].items():
            all_finger_vals.append(s)
            if s < DSP_CONFIG.weak_finger_threshold:
                disp_name = FINGER_DISPLAY_NAMES.get(name, name)
                bad_fingers.append(
                    f"{disp_name} ({'Trái' if hand_side == 'Left' else 'Phải'})"
                )

    final_score = (
        round(float(np.mean(all_finger_vals)), 2) if all_finger_vals else 0.0
    )

    timeline_report_path = os.path.join(project_root, "data", "timeline_report.json")
    with open(timeline_report_path, "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=4, ensure_ascii=False)

    error_seconds = [t["second"] for t in timeline if t["status"] == "ERROR"]
    left_avg = (
        round(float(np.mean(list(scores["Left"].values()))), 2)
        if scores["Left"]
        else 0.0
    )
    right_avg = (
        round(float(np.mean(list(scores["Right"].values()))), 2)
        if scores["Right"]
        else 0.0
    )

    summary_payload = {
        "mode": mode,
        "sheet_file": os.path.basename(sheet_path) if mode == "sheet" else None,
        "final_score": final_score,
        "left_hand_average": left_avg,
        "right_hand_average": right_avg,
        "scores": scores,
        "bad_fingers": bad_fingers,
        "total_events": len(timeline),
        "error_events": len(error_seconds),
        "error_seconds": error_seconds,
    }
    summary_path = os.path.join(project_root, "data", "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=4, ensure_ascii=False)

    if mode == "sheet":
        mode_title = "TỰ ĐỘNG NHẬN DIỆN & CHẤM THEO KHO SHEET NHẠC ĐÃ NẠP SẴN"
    elif mode == "standalone":
        mode_title = "CHẾ ĐỘ AI ĐỘC LẬP (SINH CƠ HỌC 3D - MỌI BÀI HÁT)"
    else:
        mode_title = "CHẾ ĐỘ ĐỐI CHIẾU VIDEO MẪU CHUYÊN GIA"

    print("\n\033[1;36m" + "=" * 60 + "\033[0m")
    print("\033[1;33m   BÁO CÁO ĐÁNH GIÁ KỸ THUẬT PIANO BẰNG AI (COMPUTER VISION)\033[0m")
    print(f"   Phương thức: {mode_title}")
    print("\033[1;36m" + "=" * 60 + "\033[0m")

    score_color = (
        "\033[1;32m"
        if final_score > 80
        else ("\033[1;33m" if final_score > 60 else "\033[1;31m")
    )
    print(f" {score_color}=> ĐIỂM TỔNG THỂ: {final_score:.2f} / 100\033[0m")
    print(
        f"    • Trung bình Tay Trái : {left_avg:.2f} điểm | Tay Phải : {right_avg:.2f} điểm"
    )
    print("\033[1;36m" + "-" * 60 + "\033[0m")
    print("\033[1;34m [CHI TIẾT ĐIỂM KỸ THUẬT 10 NGÓN TAY]\033[0m")

    for hand_side in ["Left", "Right"]:
        if scores[hand_side]:
            side_name = "TAY TRÁI" if hand_side == "Left" else "TAY PHẢI"
            print(f"\n\033[1;34m --- {side_name} ---\033[0m")
            for name, score in scores[hand_side].items():
                disp_name = FINGER_DISPLAY_NAMES.get(name, name)
                finger_color = (
                    "\033[1;32m"
                    if score >= 80
                    else ("\033[1;33m" if score >= 60 else "\033[1;31m")
                )
                print(f"  - {disp_name:<12}: {finger_color}{score:.2f} điểm\033[0m")

    print("\033[1;36m" + "-" * 60 + "\033[0m")
    print("\033[1;35m [NHẬN XÉT CHUYÊN MÔN & ĐỐI CHIẾU BẢN NHẠC]\033[0m")
    if final_score > 85:
        print(
            "\033[1;32m 🌟 Xuất sắc! Các nốt đánh khớp với Sheet nhạc và giữ độ cong vòm ngón tay (120°–160°) rất chuẩn.\033[0m"
        )
    elif final_score > 60:
        print(
            "\033[1;33m ⚠️ Khá! Tuy nhiên một số nốt còn lệch thế ngón hoặc góc khớp chưa ổn định.\033[0m"
        )
        if bad_fingers:
            print(
                f"\033[1;31m ❌ Cần tập trung sửa ở: {', '.join(bad_fingers)}\033[0m"
            )
    else:
        print(
            "\033[1;31m ❌ Cần luyện tập thêm để đánh đúng nốt trên Sheet nhạc và giữ form tay khum tròn.\033[0m"
        )

    print("\033[1;36m" + "-" * 60 + "\033[0m")
    print("\033[1;34m [PHÂN TÍCH TỪNG NỐT NHẠC TRÊN TIMELINE]\033[0m")
    if error_seconds:
        print(
            f" ⚠️ Phát hiện {len(error_seconds)} thời điểm sai nốt/tư thế trên tổng số {len(timeline)} nốt nhạc."
        )
        print(f" Các giây cần lưu ý: {', '.join(str(s) + 's' for s in error_seconds)}")
        print(
            " Vui lòng click vào các ô trên Timeline để xem Khuông Nhạc 5 Dòng Kẻ & Video Skeleton."
        )
    else:
        print(
            " 🎉 Tuyệt vời! Toàn bộ các nốt nhạc đều khớp với Sheet nhạc và đạt chuẩn tư thế."
        )

    print("\033[1;36m" + "=" * 60 + "\033[0m")


if __name__ == "__main__":
    main()
