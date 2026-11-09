"""
Module quét video và trích xuất tọa độ 3D của 21 khớp xương bàn tay (MediaPipe Hands).
Hỗ trợ nhận diện đồng thời hai bàn tay (Trái & Phải) và xuất dữ liệu ra định dạng JSON.
"""

import json
import os
from typing import Any, Dict, List, Optional
import cv2
import mediapipe as mp
from config import MP_CONFIG


class PianoScanner:
    """
    Lớp xử lý thị giác máy tính (Computer Vision) dùng Google MediaPipe Hands
    để trích xuất 21 điểm khớp xương (Landmarks) trên từng khung hình video.
    """

    def __init__(self) -> None:
        try:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
        except AttributeError:
            import mediapipe.python.solutions.hands as mp_hands
            import mediapipe.python.solutions.drawing_utils as mp_draw
            self.mp_hands = mp_hands
            self.mp_draw = mp_draw

        self.hands = self.mp_hands.Hands(
            static_image_mode=MP_CONFIG.static_image_mode,
            max_num_hands=MP_CONFIG.max_num_hands,
            model_complexity=MP_CONFIG.model_complexity,
            min_detection_confidence=MP_CONFIG.min_detection_confidence,
            min_tracking_confidence=MP_CONFIG.min_tracking_confidence,
        )

    def scan_video(
        self,
        video_path: str,
        output_json_path: Optional[str] = None,
        show_preview: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Quét toàn bộ khung hình của video đầu vào, trích xuất tọa độ 3D (x, y, z)
        của các bàn tay và lưu kết quả ra tệp JSON.

        Args:
            video_path: Đường dẫn tới tệp video (.mp4, .avi).
            output_json_path: Đường dẫn lưu tệp JSON kết quả (nếu có).
            show_preview: Có hiển thị cửa sổ xem trước quá trình quét hay không.

        Returns:
            Danh sách dữ liệu khung xương theo từng khung hình (frame).
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        all_frames_data: List[Dict[str, Any]] = []

        print(f"Dang quet video: {video_path}...")

        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            frame_data: Dict[str, Any] = {
                "frame_index": frame_count,
                "timestamp": frame_count / fps,
                "hands": [],
            }

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    landmarks = [
                        {"id": idx, "x": lm.x, "y": lm.y, "z": lm.z}
                        for idx, lm in enumerate(hand_lms.landmark)
                    ]
                    frame_data["hands"].append(landmarks)
                    if show_preview:
                        self.mp_draw.draw_landmarks(
                            frame, hand_lms, self.mp_hands.HAND_CONNECTIONS
                        )

            all_frames_data.append(frame_data)

            if show_preview:
                cv2.imshow("Piano AI Scanner", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1

        cap.release()
        if show_preview:
            cv2.destroyAllWindows()

        if output_json_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(all_frames_data, f, indent=4)
            print(f"Da luu du lieu quet vao: {output_json_path}")

        return all_frames_data


if __name__ == "__main__":
    scanner = PianoScanner()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    sample_video = os.path.join(project_root, "videos", "expert_sample.mp4")
    output_data = os.path.join(project_root, "data", "expert_data.json")

    if os.path.exists(sample_video):
        scanner.scan_video(sample_video, output_data)
    else:
        print(f"Khong tim thay video tai: {sample_video}")
        print(
            "Hay dam bao ban da de video vao thu muc PianoLearningAI/videos/ "
            "va doi ten thanh expert_sample.mp4"
        )
