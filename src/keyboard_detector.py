"""
Module nhận diện bố cục bàn phím Piano tự động từ khung hình video.
Ứng dụng Canny Edge Detection kết hợp Biến đổi Hough (Probabilistic Hough Transform)
và gom cụm 1D (Clustering) trên vùng bàn phím (ROI) để xác định ranh giới giữa các phím trắng.
"""

import json
import os
from typing import Dict, List, Union
import cv2
import numpy as np
from config import DEFAULT_NOTES_LIST


def detect_keyboard_layout(video_path: str, output_json_path: str) -> bool:
    """
    Phân tích các khung hình giữa video (tránh đoạn Intro tối ở đầu video)
    để tìm các vạch ngăn cách phím đàn theo phương thẳng đứng và lưu tọa độ
    chuẩn hóa [0, 1] vào tệp JSON.

    Args:
        video_path: Đường dẫn video đầu vào.
        output_json_path: Đường dẫn lưu tệp JSON bố cục bàn phím.

    Returns:
        True nếu lưu bố cục bàn phím thành công.
    """
    print(f"Bắt đầu nhận diện phím đàn từ video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Lấy mẫu 5 khung hình ở khoảng 1/3 thời lượng video (tránh màn hình đen đầu video)
    start_sample = max(0, total_frames // 3)
    if start_sample > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_sample)

    frames: List[np.ndarray] = []
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
    cap.release()

    if not frames:
        print("Lỗi: Không thể đọc video.")
        return False

    avg_frame = np.mean(frames, axis=0).astype(np.uint8)
    height, width = avg_frame.shape[:2]

    # Cắt vùng quan tâm (ROI) nửa dưới khung hình nơi đặt bàn phím Piano
    roi_gray = cv2.cvtColor(avg_frame[int(height * 0.35) :, :], cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.equalizeHist(roi_gray)
    blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)

    edges = cv2.Canny(blur, 30, 120, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=35, minLineLength=25, maxLineGap=12
    )

    vertical_x_coords: List[float] = []
    if lines is not None:
        for line in lines:
            x1, _, x2, _ = line[0]
            if abs(x1 - x2) < 18:
                vertical_x_coords.append((x1 + x2) / 2.0)

    normalized_x: List[float] = []
    if vertical_x_coords:
        vertical_x_coords.sort()
        merged_x: List[float] = []
        current_cluster: List[float] = [vertical_x_coords[0]]

        min_key_gap = max(10.0, width * 0.025)
        for x in vertical_x_coords[1:]:
            if x - current_cluster[-1] < min_key_gap:
                current_cluster.append(x)
            else:
                merged_x.append(float(np.mean(current_cluster)))
                current_cluster = [x]
        merged_x.append(float(np.mean(current_cluster)))

        normalized_x = [
            round(x / width, 4) for x in merged_x if 0.05 < (x / width) < 0.95
        ]

    # Nếu số vạch phát hiện được ít hơn 10 (do góc quay mờ), nội suy lưới phím chuẩn
    if len(normalized_x) < 10:
        num_keys = len(DEFAULT_NOTES_LIST)
        normalized_x = [
            round(float(x), 4) for x in np.linspace(0.08, 0.92, num_keys - 1)
        ]

    layout_data: Dict[str, Union[List[float], List[str]]] = {
        "key_boundaries": normalized_x,
        "notes_list": DEFAULT_NOTES_LIST,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(layout_data, f, indent=4, ensure_ascii=False)

    print(f"Đã lưu nhận diện {len(normalized_x) + 1} phím đàn vào: {output_json_path}")
    return True


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    video_path = os.path.join(project_root, "videos", "expert_sample.mp4")
    output_path = os.path.join(project_root, "data", "keyboard_layout.json")
    detect_keyboard_layout(video_path, output_path)
