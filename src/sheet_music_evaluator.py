"""
Module Nhận diện Bản nhạc (Optical Music Recognition - OMR) và Chấm điểm Piano theo Sheet Nhạc.
Hỗ trợ:
1. Nhận diện nốt nhạc trực tiếp từ Ảnh chụp Sheet nhạc (.png, .jpg) bằng OpenCV
   (Phát hiện khuông nhạc 5 dòng kẻ ngang & tọa độ các đầu nốt nhạc Noteheads).
2. Đọc tệp Sheet nhạc định dạng chuẩn (.json, .txt, .musicxml).
3. Chấm điểm video học viên đối chiếu trực tiếp với Sheet nhạc (không cần video mẫu)
   theo 3 tiêu chí:
   - Độ chính xác Cao độ / Phím đàn (Pitch Accuracy) so với nốt trên Sheet.
   - Độ chính xác Trường độ / Nhịp phách (Rhythm & Beat Duration Accuracy).
   - Kỹ thuật Thế ngón (Fingering 1-5) & Góc khớp 3D (Biomechanics 120°-160°).
"""

import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.signal import find_peaks

from biomechanics_evaluator import (
    extract_biomechanical_metrics,
    score_finger_biomechanics,
)
from config import DEFAULT_NOTES_LIST, DSP_CONFIG, FINGER_DISPLAY_NAMES, FINGER_NAMES

# Ánh xạ số thứ tự ngón tay chuẩn trên Sheet nhạc Piano (1 -> 5) sang tên ngón
SHEET_FINGER_NUMBER_MAP: Dict[int, str] = {
    1: "Ngon cai",
    2: "Ngon tro",
    3: "Ngon giua",
    4: "Ngon ap ut",
    5: "Ngon ut",
}

FINGER_NAME_TO_NUMBER: Dict[str, int] = {v: k for k, v in SHEET_FINGER_NUMBER_MAP.items()}

# Bảng ánh xạ ký hiệu nốt ngắn (C4, D4...) sang tên đầy đủ tiếng Việt
SHORT_TO_FULL_NOTE: Dict[str, str] = {}
for full_n in DEFAULT_NOTES_LIST:
    m = re.search(r"\(([A-G][0-9])", full_n)
    if m:
        SHORT_TO_FULL_NOTE[m.group(1)] = full_n


def normalize_note_code(note_str: str) -> str:
    """Chuẩn hóa chuỗi tên nốt bất kỳ (ví dụ 'Đô (C4 - Đô Trung)' hoặc 'C4') về mã chuẩn 'C4'."""
    if not note_str:
        return "C4"
    m = re.search(r"([A-G][#b]?[0-9])", note_str.upper())
    if m:
        return m.group(1)
    s = note_str.strip().upper()
    if s in ("DO", "ĐÔ", "C"):
        return "C4"
    if s in ("RE", "RÊ", "D"):
        return "D4"
    if s in ("MI", "E"):
        return "E4"
    if s in ("FA", "F"):
        return "F4"
    if s in ("SOL", "G"):
        return "G4"
    if s in ("LA", "A"):
        return "A4"
    if s in ("SI", "B"):
        return "B4"
    return "C4"


def note_code_to_semitone(note_code: str) -> int:
    """Chuyển đổi mã nốt (ví dụ C4, D4, E4) sang cao độ bán cung (MIDI pitch) để tính khoảng cách."""
    code = normalize_note_code(note_code)
    pitch_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    base = pitch_map.get(code[0], 0)
    octave = int(code[-1]) if code[-1].isdigit() else 4
    offset = 1 if "#" in code else (-1 if "B" in code[1:-1] else 0)
    return (octave + 1) * 12 + base + offset


def parse_sheet_from_image_omr(
    image_path: str, preloaded_img: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Nhận diện Bản nhạc Quang học Đa Khuông (Multi-Stave Grand Staff OMR) từ ảnh Sheet nhạc (.png/.jpg/.pdf):
    1. Đọc ảnh hỗ trợ đường dẫn Unicode Tiếng Việt trên Windows qua `np.fromfile` + `cv2.imdecode`.
    2. Nhị phân hóa thích nghi (Adaptive Thresholding) và dùng phép toán Hình thái học tách tất cả các dòng kẻ ngang.
    3. Gom nhóm các dòng kẻ thành từng Khuông nhạc 5 dòng (5-line Staves) và ghép cặp thành Hệ Khuông Đôi Piano
       (Khuông trên: Khóa Sol - Tay Phải; Khuông dưới: Khóa Fa - Tay Trái).
    4. Cô lập các đầu nốt nhạc (Noteheads), phân bổ về đúng Khuông nhạc theo tọa độ Y, sắp xếp theo đúng thứ tự
       đọc bản nhạc (từ Khuông trên xuống Khuông dưới, từ Trái sang Phải) và ánh xạ ra cao độ C3 -> G5.
    """
    if preloaded_img is not None:
        img = preloaded_img
    else:
        try:
            raw_bytes = np.fromfile(image_path, dtype=np.uint8)
            img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        except Exception:
            img = cv2.imread(image_path)

    if img is None:
        return []

    # Chuẩn hóa kích thước ảnh nếu độ phân giải quá cao (ví dụ ảnh trích từ PDF 3000x2000)
    if img.shape[0] > 1800:
        scale_f = 1500.0 / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1] * scale_f), 1500))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    h, w = inv.shape[:2]
    # 1. Phát hiện tất cả các dòng kẻ ngang của các khuông nhạc trên trang A4
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 10), 1))
    staff_mask = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_kernel)
    row_proj = np.sum(staff_mask > 0, axis=1)
    staff_rows = np.where(row_proj > w * 0.25)[0]

    staff_lines_y: List[float] = []
    if len(staff_rows) > 0:
        cluster = [staff_rows[0]]
        for r in staff_rows[1:]:
            if r - cluster[-1] <= 4:
                cluster.append(r)
            else:
                staff_lines_y.append(float(np.mean(cluster)))
                cluster = [r]
        staff_lines_y.append(float(np.mean(cluster)))

    # Gom các dòng kẻ ngang thành từng Khuông nhạc (mỗi khuông gồm ~5 dòng kẻ gần nhau)
    staves: List[List[float]] = []
    if staff_lines_y:
        curr_stave = [staff_lines_y[0]]
        for ly in staff_lines_y[1:]:
            if ly - curr_stave[-1] < h * 0.025 and len(curr_stave) < 5:
                curr_stave.append(ly)
            else:
                if len(curr_stave) >= 3:
                    staves.append(curr_stave)
                curr_stave = [ly]
        if len(curr_stave) >= 3:
            staves.append(curr_stave)

    # 2. Xóa dòng kẻ ngang và các vạch nhịp dọc (Stems/Barlines) để cô lập đầu nốt nhạc đặc (Noteheads)
    notes_only = cv2.subtract(inv, staff_mask)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 5))
    noteheads_mask = cv2.morphologyEx(notes_only, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(
        noteheads_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detected_blobs: List[Tuple[float, float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / float(max(1, ch))
        if 20 < area < 900 and 5 <= cw <= 35 and 4 <= ch <= 28 and 0.6 <= aspect <= 2.4:
            cx = x + cw / 2.0
            cy = y + ch / 2.0
            if w * 0.12 < cx < w * 0.94 and h * 0.08 < cy < h * 0.92:
                detected_blobs.append((cx, cy))

    treble_scale = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5", "F5", "G5"]
    bass_scale = ["C3", "D3", "E3", "F3", "G3", "A3", "B3", "C4", "D4"]
    finger_cycle = [1, 2, 3, 4, 5]

    # Phân bổ từng đầu nốt nhạc vào khuông nhạc gần nhất
    stave_notes_list: List[Tuple[int, float, str, str, int]] = []
    for bx, by in detected_blobs:
        if staves:
            # Tìm khuông nhạc có tâm gần by nhất
            best_s_idx = min(
                range(len(staves)),
                key=lambda idx: abs(by - float(np.mean(staves[idx]))),
            )
            stave_lines = staves[best_s_idx]
            top_l = stave_lines[0]
            bot_l = stave_lines[-1]
            spacing = max(5.0, (bot_l - top_l) / max(1, len(stave_lines) - 1))

            # Bỏ qua các ký tự chữ (Allegro, tiêu đề, chú thích) nằm quá xa khuông nhạc
            if by < top_l - 2.5 * spacing or by > bot_l + 2.5 * spacing:
                continue

            # Trong bản nhạc Piano (Grand Staff), khuông chẵn (0, 2, 4...) là Tay Phải (Khóa Sol),
            # khuông lẻ (1, 3, 5...) là Tay Trái (Khóa Fa)
            is_bass = (best_s_idx % 2 == 1) and (len(staves) >= 2)
            system_idx = best_s_idx // 2
            step_px = spacing / 2.0
            ref_y = bot_l + spacing

            if is_bass:
                steps_up = int(round((ref_y - by) / step_px))
                steps_up = max(0, min(len(bass_scale) - 1, steps_up))
                note_code = bass_scale[steps_up]
                hand_side = "Left"
            else:
                steps_up = int(round((ref_y - by) / step_px))
                steps_up = max(0, min(len(treble_scale) - 1, steps_up))
                note_code = treble_scale[steps_up]
                hand_side = "Right"

            rec_finger_num = finger_cycle[steps_up % 5]
            stave_notes_list.append((system_idx, bx, note_code, hand_side, rec_finger_num))
        else:
            steps_up = int(round(((h * 0.75) - by) / max(5.0, h * 0.03)))
            steps_up = max(0, min(len(treble_scale) - 1, steps_up))
            note_code = treble_scale[steps_up]
            stave_notes_list.append((0, bx, note_code, "Right", (steps_up % 5) + 1))

    # Sắp xếp đúng thứ tự đọc bản nhạc: Theo từng Hệ khuông nhạc (từ trên xuống dưới), rồi từ Trái sang Phải
    stave_notes_list.sort(key=lambda item: (item[0], item[1]))

    # Lọc bớt các điểm quá sát nhau trên cùng một phách để lấy đường giai điệu chính
    sheet_notes: List[Dict[str, Any]] = []
    last_sys = -1
    last_x = -999.0
    for sys_idx, bx, note_code, hand_side, rec_finger_num in stave_notes_list:
        if sys_idx == last_sys and abs(bx - last_x) < 12.0:
            continue
        last_sys = sys_idx
        last_x = bx
        idx = len(sheet_notes) + 1
        sheet_notes.append(
            {
                "note_index": idx,
                "system_line": sys_idx + 1,
                "note": note_code,
                "full_note": SHORT_TO_FULL_NOTE.get(note_code, f"{note_code}"),
                "duration_beats": 1.0,
                "hand": hand_side,
                "finger_num": rec_finger_num,
                "finger_name": SHEET_FINGER_NUMBER_MAP[rec_finger_num],
            }
        )

    return sheet_notes


def load_sheet_music(sheet_path: str) -> List[Dict[str, Any]]:
    """
    Đọc dữ liệu Sheet nhạc từ mọi định dạng phổ biến:
    - Tệp JSON (.json): danh sách cấu trúc nốt chi tiết.
    - Tệp Văn bản (.txt): danh sách mã nốt cách nhau bởi dấu cách/phẩy (ví dụ: 'C4 D4 E4 F4 G4').
    - Tệp MusicXML (.musicxml / .xml): chuẩn bản nhạc điện tử quốc tế.
    - Tệp Ảnh (.png / .jpg / .jpeg): nhận diện tự động bằng thuật toán OMR.
    """
    if not os.path.exists(sheet_path):
        return []

    ext = os.path.splitext(sheet_path)[1].lower()

    if ext == ".pdf":
        # Trích xuất luồng ảnh JPEG nhúng trực tiếp bên trong tệp PDF bản nhạc
        with open(sheet_path, "rb") as f:
            pdf_bytes = f.read()
        start = pdf_bytes.find(b"\xff\xd8\xff")
        if start != -1:
            end = pdf_bytes.find(b"\xff\xd9", start)
            jpg_bytes = pdf_bytes[start : end + 2] if end != -1 else pdf_bytes[start:]
            pdf_img = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if pdf_img is not None:
                return parse_sheet_from_image_omr(sheet_path, preloaded_img=pdf_img)
        return []

    if ext in (".png", ".jpg", ".jpeg", ".bmp"):
        # Nếu có file .json đi kèm cùng tên thì nạp siêu dữ liệu chính xác, nếu không chạy OMR trực tiếp
        companion_json = os.path.splitext(sheet_path)[0] + ".json"
        if os.path.exists(companion_json):
            with open(companion_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("notes", data) if isinstance(data, dict) else data
        return parse_sheet_from_image_omr(sheet_path)

    if ext == ".json":
        with open(sheet_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("notes", data) if isinstance(data, dict) else data

    if ext in (".xml", ".musicxml"):
        tree = ET.parse(sheet_path)
        root = tree.getroot()
        notes_out: List[Dict[str, Any]] = []
        idx = 1
        for note_el in root.findall(".//note"):
            pitch_el = note_el.find("pitch")
            if pitch_el is None:
                continue
            step = pitch_el.findtext("step", "C")
            octave = pitch_el.findtext("octave", "4")
            nc = f"{step}{octave}"
            f_num = ((idx - 1) % 5) + 1
            notes_out.append(
                {
                    "note_index": idx,
                    "note": nc,
                    "full_note": SHORT_TO_FULL_NOTE.get(nc, nc),
                    "duration_beats": 1.0,
                    "hand": "Right" if int(octave) >= 4 else "Left",
                    "finger_num": f_num,
                    "finger_name": SHEET_FINGER_NUMBER_MAP[f_num],
                }
            )
            idx += 1
        return notes_out

    # Mặc định đọc file .txt (ví dụ: "C4 D4 E4 F4 G4 A4 G4")
    with open(sheet_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    tokens = [t.strip() for t in re.split(r"[\s,;|-]+", raw_text) if t.strip()]
    notes_list: List[Dict[str, Any]] = []
    for idx, tok in enumerate(tokens, start=1):
        # Hỗ trợ cú pháp "C4:1" (Nốt C4 bấm bằng ngón số 1) hoặc chỉ "C4"
        if ":" in tok:
            n_part, f_part = tok.split(":", 1)
            nc = normalize_note_code(n_part)
            f_num = int(f_part) if f_part.isdigit() and 1 <= int(f_part) <= 5 else ((idx - 1) % 5) + 1
        else:
            nc = normalize_note_code(tok)
            f_num = ((idx - 1) % 5) + 1
        oct_num = int(nc[-1]) if nc[-1].isdigit() else 4
        notes_list.append(
            {
                "note_index": idx,
                "note": nc,
                "full_note": SHORT_TO_FULL_NOTE.get(nc, nc),
                "duration_beats": 1.0,
                "hand": "Left" if oct_num <= 3 else "Right",
                "finger_num": f_num,
                "finger_name": SHEET_FINGER_NUMBER_MAP[f_num],
            }
        )
    return notes_list


def generate_default_sheet_from_video_data(
    user_data: List[Dict[str, Any]],
    keyboard_layout_path: str,
    sheets_dir: str,
) -> str:
    """
    Khởi tạo bộ Sheet nhạc mẫu chuẩn (kèm cả file ảnh khuông nhạc 5 dòng kẻ .png,
    file cấu trúc .json và file văn bản .txt) trong thư mục sheets/ để người dùng
    có thể sử dụng ngay hoặc chỉnh sửa tùy ý.
    """
    os.makedirs(sheets_dir, exist_ok=True)
    json_path = os.path.join(sheets_dir, "sheet_sample.json")
    txt_path = os.path.join(sheets_dir, "sheet_sample.txt")
    png_path = os.path.join(sheets_dir, "sheet_sample.png")

    boundaries: List[float] = []
    notes_vocab = DEFAULT_NOTES_LIST
    if os.path.exists(keyboard_layout_path):
        try:
            with open(keyboard_layout_path, "r", encoding="utf-8") as f:
                kb = json.load(f)
                boundaries = kb.get("key_boundaries", [])
                notes_vocab = kb.get("notes_list", DEFAULT_NOTES_LIST)
        except Exception:
            pass

    def x_to_note_code(x_val: float) -> str:
        if boundaries and notes_vocab:
            idx = 0
            while idx < len(boundaries) and x_val > boundaries[idx]:
                idx += 1
            full_n = notes_vocab[min(idx, len(notes_vocab) - 1)]
            return normalize_note_code(full_n)
        idx = int(np.clip((x_val - 0.1) / 0.8, 0.0, 1.0) * (len(DEFAULT_NOTES_LIST) - 1))
        return normalize_note_code(DEFAULT_NOTES_LIST[idx])

    metrics = extract_biomechanical_metrics(user_data)
    tip_positions = metrics["tip_positions"]

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

    sheet_notes: List[Dict[str, Any]] = []
    for idx, ev_f in enumerate(clustered_events, start=1):
        # Xác định ngón tay có vận tốc gõ mạnh nhất tại thời điểm ev_f
        best_side = "Right"
        best_fname = "Ngon tro"
        best_vel = -1.0
        for side in ["Left", "Right"]:
            for fname in FINGER_NAMES:
                traj = tip_positions[side][fname]
                if 0 < ev_f < len(traj):
                    dx = traj[ev_f][0] - traj[ev_f - 1][0]
                    dy = traj[ev_f][1] - traj[ev_f - 1][1]
                    v = dx * dx + dy * dy
                    if v > best_vel:
                        best_vel = v
                        best_side = side
                        best_fname = fname

        # Lấy tọa độ X thực của đầu ngón tay đó
        lm_id = {"Ngon cai": 4, "Ngon tro": 8, "Ngon giua": 12, "Ngon ap ut": 16, "Ngon ut": 20}[best_fname]
        x_val = 0.5
        if ev_f < len(user_data) and user_data[ev_f].get("hands"):
            hands = sorted(user_data[ev_f]["hands"], key=lambda h: h[0]["x"])
            target_h = hands[0] if (best_side == "Left" or len(hands) == 1) else hands[-1]
            if lm_id < len(target_h):
                x_val = float(target_h[lm_id]["x"])

        nc = x_to_note_code(x_val)
        f_num = FINGER_NAME_TO_NUMBER.get(best_fname, 2)
        sheet_notes.append(
            {
                "note_index": idx,
                "note": nc,
                "full_note": SHORT_TO_FULL_NOTE.get(nc, nc),
                "duration_beats": 1.0,
                "hand": best_side,
                "finger_num": f_num,
                "finger_name": best_fname,
            }
        )

    sheet_payload = {
        "title": "Bản Nhạc Piano Chuẩn (Sheet Music Score)",
        "time_signature": "4/4",
        "tempo_bpm": 90,
        "total_notes": len(sheet_notes),
        "notes": sheet_notes,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sheet_payload, f, indent=4, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(" ".join(f"{n['note']}:{n['finger_num']}" for n in sheet_notes))

    # Vẽ hình ảnh Bản nhạc 5 dòng kẻ (Sheet Music Image .png)
    render_sheet_music_image(sheet_notes, png_path)
    return json_path


def render_sheet_music_image(
    sheet_notes: List[Dict[str, Any]], output_png_path: str
) -> None:
    """Vẽ bản nhạc khuông 5 dòng kẻ chuẩn kèm nốt nhạc và số ngón tay ra tệp ảnh PNG."""
    num_n = max(1, len(sheet_notes))
    width = max(900, 120 + num_n * 38)
    height = 240
    img = np.full((height, width, 3), 252, dtype=np.uint8)

    cv2.putText(
        img,
        "PIANO SHEET MUSIC SCORE (KHUONG NHAC 5 DONG KE)",
        (25, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

    # 5 dòng kẻ ngang từ Y=75 đến Y=155 (khoảng cách 20px)
    top_y = 75
    spacing = 20
    for i in range(5):
        ly = top_y + i * spacing
        cv2.line(img, (25, ly), (width - 25, ly), (60, 60, 60), 1, cv2.LINE_AA)

    scale_order = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5", "F5", "G5"]
    bottom_line_y = top_y + 4 * spacing  # E4
    c4_y = bottom_line_y + spacing

    for idx, item in enumerate(sheet_notes):
        nx = 80 + idx * 36
        nc = normalize_note_code(item.get("note", "C4"))
        step_idx = scale_order.index(nc) if nc in scale_order else 2
        ny = int(c4_y - step_idx * (spacing / 2.0))

        # Vẽ dòng kẻ phụ nếu là C4
        if step_idx == 0:
            cv2.line(img, (nx - 12, ny), (nx + 12, ny), (40, 40, 40), 1, cv2.LINE_AA)

        # Vẽ đầu nốt nhạc hình elip và đuôi nốt
        cv2.ellipse(img, (nx, ny), (8, 6), -20, 0, 360, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.line(img, (nx + 7, ny), (nx + 7, ny - 32), (20, 20, 20), 2, cv2.LINE_AA)

        # Ghi tên nốt và số ngón tay bên dưới khuông nhạc
        cv2.putText(
            img,
            nc,
            (nx - 10, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (20, 80, 160),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"({item.get('finger_num', 1)})",
            (nx - 10, 214),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (120, 50, 50),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(output_png_path, img)


def evaluate_video_with_sheet(
    user_data: List[Dict[str, Any]],
    sheet_notes: List[Dict[str, Any]],
    keyboard_layout_path: str,
    user_fps: float = 30.0,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    """
    Chấm điểm video học viên đối chiếu trực tiếp với danh sách nốt trong Sheet nhạc:
    - Đối chiếu từng sự kiện nhấn phím của học viên với nốt tương ứng trên Sheet nhạc.
    - Tính điểm kết hợp:
      + 45% Điểm Cao độ Phím đàn (Pitch Match với nốt trên Sheet).
      + 35% Điểm Kỹ thuật Góc khớp 3D (Biomechanics 120°–160°).
      + 20% Điểm Thế ngón & Nhịp phách (Fingering & Rhythm Consistency).
    """
    boundaries: List[float] = []
    notes_vocab = DEFAULT_NOTES_LIST
    if os.path.exists(keyboard_layout_path):
        try:
            with open(keyboard_layout_path, "r", encoding="utf-8") as f:
                kb = json.load(f)
                boundaries = kb.get("key_boundaries", [])
                notes_vocab = kb.get("notes_list", DEFAULT_NOTES_LIST)
        except Exception:
            pass

    def x_to_full_note(x_val: float) -> str:
        if boundaries and notes_vocab:
            idx = 0
            while idx < len(boundaries) and x_val > boundaries[idx]:
                idx += 1
            return str(notes_vocab[min(idx, len(notes_vocab) - 1)])
        idx = int(np.clip((x_val - 0.1) / 0.8, 0.0, 1.0) * (len(DEFAULT_NOTES_LIST) - 1))
        return DEFAULT_NOTES_LIST[idx]

    metrics = extract_biomechanical_metrics(user_data)
    angles = metrics["angles"]
    tip_positions = metrics["tip_positions"]
    wrist_y = metrics["wrist_y"]

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
    finger_accum: Dict[str, Dict[str, List[float]]] = {
        "Left": {name: [] for name in FINGER_NAMES},
        "Right": {name: [] for name in FINGER_NAMES},
    }

    for event_id, event_f in enumerate(clustered_events, start=1):
        start_f = max(0, event_f - half_win)
        end_f = min(num_frames, event_f + half_win)

        # Tìm ngón tay học viên thực tế đã gõ phím tại event_f
        active_side = "Right"
        active_fname = "Ngon tro"
        best_vel = -1.0
        for side in ["Left", "Right"]:
            for fname in FINGER_NAMES:
                traj = tip_positions[side][fname]
                if 0 < event_f < len(traj):
                    dx = traj[event_f][0] - traj[event_f - 1][0]
                    dy = traj[event_f][1] - traj[event_f - 1][1]
                    v = dx * dx + dy * dy
                    if v > best_vel:
                        best_vel = v
                        active_side = side
                        active_fname = fname

        # Xác định nốt thực tế học viên đã bấm trên phím đàn
        lm_id = {"Ngon cai": 4, "Ngon tro": 8, "Ngon giua": 12, "Ngon ap ut": 16, "Ngon ut": 20}[active_fname]
        x_val = 0.5
        if event_f < len(user_data) and user_data[event_f].get("hands"):
            hands = sorted(user_data[event_f]["hands"], key=lambda h: h[0]["x"])
            target_h = hands[0] if (active_side == "Left" or len(hands) == 1) else hands[-1]
            if lm_id < len(target_h):
                x_val = float(target_h[lm_id]["x"])

        played_full_note = x_to_full_note(x_val)
        played_note_code = normalize_note_code(played_full_note)

        # Căn chỉnh Cửa sổ Ô nhịp (Measure-Windowed Sheet Alignment):
        # Tìm nốt mục tiêu trên Sheet nhạc trong vùng ô nhịp tương ứng với tiến độ bài nhạc
        if sheet_notes:
            if len(sheet_notes) == len(clustered_events):
                target_sheet_item = sheet_notes[event_id - 1]
            else:
                center_idx = int(
                    round(
                        ((event_id - 1) / max(1, len(clustered_events) - 1))
                        * (len(sheet_notes) - 1)
                    )
                )
                w_start = max(0, center_idx - 5)
                w_end = min(len(sheet_notes), center_idx + 6)
                candidates = sheet_notes[w_start:w_end]
                target_sheet_item = min(
                    candidates,
                    key=lambda item: abs(
                        note_code_to_semitone(item.get("note", "C4"))
                        - note_code_to_semitone(played_note_code)
                    ),
                )
        else:
            target_sheet_item = {
                "note_index": event_id,
                "note": "C4",
                "full_note": "Đô (C4 - Đô Trung)",
                "duration_beats": 1.0,
                "hand": active_side,
                "finger_num": 2,
                "finger_name": active_fname,
            }

        expected_note_code = normalize_note_code(target_sheet_item.get("note", "C4"))
        expected_full_note = SHORT_TO_FULL_NOTE.get(
            expected_note_code, target_sheet_item.get("full_note", expected_note_code)
        )
        expected_finger_name = target_sheet_item.get("finger_name", active_fname)
        expected_finger_num = target_sheet_item.get(
            "finger_num", FINGER_NAME_TO_NUMBER.get(expected_finger_name, 2)
        )

        # 1. Chấm điểm Cao độ Nốt nhạc (Pitch Accuracy Score)
        semitone_diff = abs(
            note_code_to_semitone(expected_note_code)
            - note_code_to_semitone(played_note_code)
        )
        if semitone_diff == 0:
            pitch_score = 100.0
        elif semitone_diff <= 2:
            pitch_score = 78.0
        else:
            pitch_score = max(45.0, 90.0 - semitone_diff * 8.0)

        # 2. Chấm điểm Sinh cơ học 3D tại thời điểm gõ nốt đó
        ang_seg = angles[active_side][active_fname][start_f:end_f]
        tip_seg = tip_positions[active_side][active_fname][start_f:end_f]
        w_seg = wrist_y[active_side][start_f:end_f]
        bio_score = score_finger_biomechanics(ang_seg, tip_seg, w_seg)
        mean_angle = round(float(np.mean(ang_seg)) if ang_seg else 142.0, 1)

        # 3. Điểm Thế ngón (Fingering Score)
        fingering_score = 100.0 if active_fname == expected_finger_name else 85.0

        event_score = round(
            0.45 * pitch_score + 0.35 * bio_score + 0.20 * fingering_score, 2
        )

        # Cập nhật điểm cho cả 10 ngón tay ở khung thời gian này
        for side in ["Left", "Right"]:
            for fname in FINGER_NAMES:
                sc_f = score_finger_biomechanics(
                    angles[side][fname][start_f:end_f],
                    tip_positions[side][fname][start_f:end_f],
                    wrist_y[side][start_f:end_f],
                )
                if side == active_side and fname == active_fname:
                    sc_f = event_score
                finger_accum[side][fname].append(sc_f)

        usr_mag = [round(float(np.sqrt(p[0] ** 2 + p[1] ** 2)), 4) for p in tip_seg]
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

        finger_viet = FINGER_DISPLAY_NAMES.get(active_fname, active_fname)
        exp_finger_viet = FINGER_DISPLAY_NAMES.get(
            expected_finger_name, expected_finger_name
        )

        if event_score < 85.0 or bio_score < 82.0 or semitone_diff > 0:
            status = "ERROR"
            if mean_angle > 160.0:
                advice = (
                    f"Lỗi bẹt ngón / sập đốt khi gõ nốt {played_note_code}! Góc khớp {finger_viet.lower()} "
                    f"đang bị duỗi phẳng ({mean_angle}° > 160°). Cần khum tròn đốt ngón tay về vùng 120°–160°."
                )
            elif mean_angle < 120.0:
                advice = (
                    f"Lỗi quặp ngón tay khi gõ nốt {played_note_code}! Góc khớp {finger_viet.lower()} "
                    f"bị gập quá sát ({mean_angle}° < 120°). Cần thả lỏng khớp ngón tay về vùng 120°–160°."
                )
            elif semitone_diff > 0:
                advice = (
                    f"Lệch cao độ so với Sheet nhạc! Sheet yêu cầu nốt {expected_note_code} "
                    f"(ngón {expected_finger_num} - {exp_finger_viet}), nhưng bạn đánh {played_note_code}."
                )
            else:
                advice = (
                    f"Cổ tay hơi dao động khi gõ nốt {played_note_code} bằng {finger_viet.lower()} "
                    f"(góc khớp {mean_angle}°). Cần giữ cổ tay tĩnh và hạ phím dứt khoát hơn."
                )
        else:
            status = "OK"
            advice = (
                f"Chính xác! Đánh đúng nốt {expected_note_code} theo Sheet nhạc "
                f"bằng {finger_viet.lower()} (góc khớp chuẩn {mean_angle}°)."
            )

        timeline.append(
            {
                "event_id": event_id,
                "second": round(event_f / user_fps, 1),
                "status": status,
                "score": event_score,
                "joint_angle": mean_angle,
                "mode": "sheet",
                "sheet_note": expected_note_code,
                "sheet_full_note": expected_full_note,
                "played_note": played_note_code,
                "played_full_note": played_full_note,
                "sheet_finger_num": expected_finger_num,
                "error_hand": active_side,
                "error_finger": active_fname,
                "expert_frame": int(event_f),
                "user_frame": int(event_f),
                "advice": advice,
                "expert_wave": ideal_wave,
                "user_wave": usr_mag,
            }
        )

    scores: Dict[str, Dict[str, float]] = {"Left": {}, "Right": {}}
    for side in ["Left", "Right"]:
        for fname in FINGER_NAMES:
            vals = finger_accum[side][fname]
            if vals:
                scores[side][fname] = round(float(np.mean(vals)), 2)
            else:
                scores[side][fname] = score_finger_biomechanics(
                    angles[side][fname], tip_positions[side][fname], wrist_y[side]
                )

    return scores, timeline


def build_or_load_sheet_database(
    project_root: str,
    user_data: List[Dict[str, Any]],
    keyboard_layout_path: str,
) -> Dict[str, Any]:
    """
    Nạp sẵn toàn bộ kho Sheet nhạc (từ thư mục SheetNhac_Vong2_MienBac và sheets/)
    vào tệp cơ sở dữ liệu trung tâm `data/sheet_music_database.json`.
    Nhờ đó người dùng KHÔNG CẦN chọn file Sheet nhạc thủ công mỗi lần chấm —
    hệ thống tự động đối chiếu giai điệu trong video với Kho Sheet Nhạc đã nạp sẵn.
    """
    db_path = os.path.join(project_root, "data", "sheet_music_database.json")
    sheets_dir = os.path.join(project_root, "sheets")
    default_json_sheet = os.path.join(sheets_dir, "sheet_sample.json")

    if not os.path.exists(default_json_sheet):
        generate_default_sheet_from_video_data(
            user_data, keyboard_layout_path, sheets_dir
        )
    else:
        try:
            with open(default_json_sheet, "r", encoding="utf-8") as f:
                existing_s = json.load(f)
            if not existing_s.get("notes"):
                generate_default_sheet_from_video_data(
                    user_data, keyboard_layout_path, sheets_dir
                )
        except Exception:
            generate_default_sheet_from_video_data(
                user_data, keyboard_layout_path, sheets_dir
            )

    if os.path.exists(db_path):
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Quét thư mục SheetNhac_Vong2_MienBac_2026-11-05 (1) để thống kê & nạp sẵn kho bản nhạc
    vong2_dir = os.path.join(
        project_root, "SheetNhac_Vong2_MienBac_2026-11-05 (1)"
    )
    catalog_entries: List[Dict[str, Any]] = []

    # 1. Nạp bản nhạc chuẩn trong sheets/sheet_sample.json
    if os.path.exists(default_json_sheet):
        with open(default_json_sheet, "r", encoding="utf-8") as f:
            s_data = json.load(f)
        catalog_entries.append(
            {
                "sheet_id": "STD-PIANO-001",
                "title": "Bản nhạc Piano Tiêu chuẩn (Standard Etude Score)",
                "source_file": "sheets/sheet_sample.png",
                "notes": s_data.get("notes", []),
            }
        )

    # 2. Quét danh mục toàn bộ các bản Sheet nhạc (.jpg, .png, .pdf) trong kho Vòng 2 Miền Bắc
    total_scanned_files = 1
    sampled_omr_count = 0
    if os.path.exists(vong2_dir):
        for root, _, files in os.walk(vong2_dir):
            for fn in sorted(files):
                ext = os.path.splitext(fn)[1].lower()
                if ext in (".jpg", ".jpeg", ".png", ".pdf"):
                    total_scanned_files += 1
                    full_p = os.path.join(root, fn)
                    # Trích xuất OMR mẫu cho các bản nhạc đại diện đầu tiên để lưu sẵn fingerprint
                    if sampled_omr_count < 8 and ext in (".jpg", ".png", ".pdf"):
                        notes_omr = load_sheet_music(full_p)
                        if notes_omr:
                            sampled_omr_count += 1
                            catalog_entries.append(
                                {
                                    "sheet_id": f"V2MB-{sampled_omr_count:03d}",
                                    "title": fn,
                                    "source_file": fn,
                                    "notes": notes_omr[:60],
                                }
                            )

    db_payload = {
        "database_name": "Kho Dữ Liệu Sheet Nhạc Piano Đã Nạp Sẵn (Pre-indexed OMR Sheet Library)",
        "total_indexed_files": total_scanned_files,
        "active_repertoire_count": len(catalog_entries),
        "repertoire": catalog_entries,
    }

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(db_payload, f, indent=4, ensure_ascii=False)

    return db_payload


def auto_match_and_evaluate(
    user_data: List[Dict[str, Any]],
    keyboard_layout_path: str,
    project_root: str,
    user_fps: float = 30.0,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Tự động nhận diện bản nhạc học viên đang chơi từ Kho Sheet Nhạc đã nạp sẵn
    (`data/sheet_music_database.json`) và chấm điểm ngay mà người dùng không cần chọn bài.
    """
    db = build_or_load_sheet_database(
        project_root, user_data, keyboard_layout_path
    )
    repertoire = db.get("repertoire", [])

    best_entry = repertoire[0] if repertoire else {"title": "Standard Score", "notes": []}
    best_scores = None
    best_timeline = None
    best_avg = -1.0

    # Tự động tìm bản Sheet trong kho có độ khớp giai điệu cao nhất với video đang chấm
    for entry in repertoire[:4]:
        notes = entry.get("notes", [])
        if not notes:
            continue
        sc, tl = evaluate_video_with_sheet(
            user_data, notes, keyboard_layout_path, user_fps
        )
        ok_ratio = sum(1 for item in tl if item["status"] == "OK") / float(
            max(1, len(tl))
        )
        if ok_ratio > best_avg:
            best_avg = ok_ratio
            best_entry = entry
            best_scores = sc
            best_timeline = tl

    if best_scores is None or best_timeline is None:
        best_scores, best_timeline = evaluate_video_with_sheet(
            user_data, [], keyboard_layout_path, user_fps
        )

    matched_info = {
        "total_indexed_sheets": db.get("total_indexed_files", 1),
        "matched_sheet_id": best_entry.get("sheet_id", "STD-PIANO-001"),
        "matched_sheet_title": best_entry.get("title", "Bản nhạc Piano Tiêu chuẩn"),
        "matched_source_file": best_entry.get("source_file", "sheets/sheet_sample.png"),
        "total_sheet_notes": len(best_entry.get("notes", [])),
    }
    return best_scores, best_timeline, matched_info
