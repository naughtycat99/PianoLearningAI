# Piano Learning AI — Hệ thống Đánh giá & Trực quan hóa Kỹ thuật Chơi Đàn bằng Computer Vision & DSP

**Sinh viên thực hiện:** Nguyễn Việt Anh — **Mã SV:** B17DCCN043  
**Học phần:** Lập trình với Python (`INT13162`)

---

## 1. Giới thiệu Dự án
**Piano Learning AI** là ứng dụng hỗ trợ người tự học Piano tại nhà thông qua công nghệ Thị giác máy tính (**Google MediaPipe Hands**, **OpenCV**) kết hợp với các thuật toán Xử lý tín hiệu số (**NumPy**, **SciPy**). Hệ thống tự động trích xuất quỹ đạo chuyển động 3D của 21 khớp xương trên cả hai bàn tay (Trái & Phải), nhận diện bố cục phím đàn, đồng bộ hóa nhịp điệu với video mẫu của Chuyên gia (bằng **Z-Score**, **Windowed Cross-Correlation** và **Dynamic Time Warping - DTW**) và trực quan hóa dữ liệu trên giao diện Dark Mode hiện đại.

## 2. Kiến trúc Thư mục Dự án
```text
PianoLearningAI/
├── gui.py                        # Giao diện đồ họa (Tkinter Dark Mode + Biểu đồ cột 10 ngón + Timeline + Video Loop + Biểu đồ sóng)
├── KhoiDong.bat                  # Script khởi động nhanh ứng dụng trên Windows
├── requirements.txt              # Danh sách các thư viện Python cần thiết
├── PianoLearningAI_Report.md     # Báo cáo học thuật & Tự đánh giá 5 tiêu chí (Markdown)
├── Bao_Cao_PianoLearningAI_HoanChinh.docx  # Báo cáo Word hoàn chỉnh kèm hình ảnh minh chứng
├── Bao_Cao_PianoLearningAI_VietAnh.docx    # Báo cáo học thuật chi tiết theo chuẩn PTIT
├── src/
│   ├── config.py                 # Cấu hình trung tâm (@dataclass) và các hằng số thuật toán (Clean Code PEP 8)
│   ├── scanner.py                # Module quét video và trích xuất 21 khớp xương tay (MediaPipe Hands)
│   ├── keyboard_detector.py      # Module nhận diện ranh giới phím đàn (Canny Edge + Hough Lines + ROI)
│   ├── evaluator.py              # Module chấm điểm 10 ngón tay (Auto-Cropping, Z-Score, Windowed Cross-Correlation)
│   └── timeline_evaluator.py     # Module căn chỉnh DTW, phát hiện Note Onset & trích xuất dữ liệu sóng (Waveform)
├── tests/
│   └── test_evaluator.py         # Bộ kiểm thử tự động (Unit Tests) cho các thuật toán DSP & DTW
├── videos/
│   ├── expert_sample.mp4         # Video mẫu của chuyên gia
│   └── user_sample.mp4           # Video luyện tập của học viên
└── data/
    ├── expert_data.json          # Dữ liệu khung xương chuyên gia (được cache tự động)
    ├── user_data.json            # Dữ liệu khung xương học viên
    ├── keyboard_layout.json      # Dữ liệu bố cục 22 phím đàn (C3 - C6)
    ├── evaluation_summary.json   # Tổng hợp điểm số 10 ngón tay và thống kê phục vụ vẽ biểu đồ
    └── timeline_report.json      # Báo cáo phân tích từng sự kiện gõ phím kèm biên độ sóng (Waveform)
```

## 3. Hướng dẫn Cài đặt & Sử dụng
1. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Khởi chạy giao diện đồ họa (Khuyên dùng):**
   ```bash
   python gui.py
   ```
   - Nhấn **"📂 CHỌN VIDEO"** để chọn video học viên bất kỳ (hoặc dùng mặc định `videos/user_sample.mp4`).
   - Nhấn **"▶ CHẠY CHẤM ĐIỂM"** để hệ thống phân tích và tự động vẽ **Biểu đồ cột 10 ngón tay**.
   - Nhấn vào bất kỳ ô nào trên **Timeline Sự Kiện Gõ Phím** để xem **Video Loop đối chiếu khung xương song song** và **Biểu đồ Sóng biên độ (Waveform)** kèm hướng dẫn sửa lỗi từng ngón.
   - Nhấn **"💾 XUẤT BÁO CÁO"** để lưu kết quả ra tệp `.txt` hoặc `.json`.
3. **Chạy bộ kiểm thử tự động (Unit Tests):**
   ```bash
   python tests/test_evaluator.py -v
   ```
