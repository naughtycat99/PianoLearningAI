"""
Giao diện Đồ họa Người dùng (GUI) và Trực quan hóa Dữ liệu cho hệ thống Piano Learning AI.
Tích hợp:
1. Bảng điều khiển đa luồng (Chọn video, Chạy AI chấm điểm, Reset cache học viên, Xuất báo cáo).
2. Biểu đồ Cột trực quan hóa điểm số 10 ngón tay (Tay Trái & Tay Phải).
3. Thanh Timeline tương tác từng sự kiện gõ phím (Note Onset) hiển thị tên nốt nhạc.
4. Khung Review đối chiếu Video Loop song song (kèm Skeleton Overlay) và Biểu đồ Sóng quỹ đạo (Waveform).
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from PIL import Image, ImageTk

# Thêm thư mục src vào sys.path để tái sử dụng module config
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import (  # noqa: E402
    DEFAULT_NOTES_LIST,
    FINGER_DISPLAY_NAMES,
    FINGER_NAMES,
    FINGER_TIPS_MAP,
    MEDIAPIPE_CONNECTIONS,
)


def strip_ansi(text: str) -> str:
    """Loại bỏ các ký tự điều khiển màu ANSI khỏi chuỗi văn bản."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


class PianoAIGUI:
    """Lớp giao diện chính của ứng dụng Piano Learning AI (Tkinter Dark Mode)."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(
            "Piano Learning AI — Hệ thống Đánh giá & Trực quan hóa Kỹ thuật Chơi Đàn"
        )
        self.root.geometry("1340x860")
        self.root.configure(bg="#0d1117")
        self.root.resizable(True, True)

        self.current_dir = CURRENT_DIR
        self.data_dir = os.path.join(self.current_dir, "data")
        self.videos_dir = os.path.join(self.current_dir, "videos")
        self.sheets_dir = os.path.join(self.current_dir, "sheets")
        self.selected_user_video = os.path.join(self.videos_dir, "user_sample.mp4")
        self.selected_sheet_path = os.path.join(self.sheets_dir, "sheet_sample.png")
        self.eval_mode = tk.StringVar(value="sheet")

        self.timeline_data: List[Dict[str, Any]] = []
        self.summary_data: Dict[str, Any] = {}
        self.selected_second: Optional[int] = None
        self.timeline_cells: List[tk.Widget] = []

        self.cap_expert: Optional[cv2.VideoCapture] = None
        self.cap_user: Optional[cv2.VideoCapture] = None
        self.loop_timer_id: Optional[str] = None
        self.is_playing_loop: bool = False
        self.is_video_paused: bool = False
        self.expert_raw_data: List[Dict[str, Any]] = []
        self.user_raw_data: List[Dict[str, Any]] = []
        self.curr_expert_note: str = "Không phát hiện"
        self.curr_user_note: str = "Không phát hiện"

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_header()
        self._build_main_panel()
        self._build_status_bar()

        # Tự động nạp kết quả sẵn có nếu đã từng chạy trước đó
        self.root.after(200, self.autoload_existing_results)

    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg="#161b22", pady=10)
        header.pack(fill=tk.X)

        title_frame = tk.Frame(header, bg="#161b22")
        title_frame.pack(side=tk.LEFT, padx=18)
        tk.Label(
            title_frame,
            text="🎹  PIANO LEARNING AI  —  Đánh Giá & Trực Quan Hóa Kỹ Thuật Piano",
            font=("Segoe UI", 15, "bold"),
            bg="#161b22",
            fg="#58a6ff",
        ).pack(anchor="w")
        self.video_lbl = tk.Label(
            title_frame,
            text=f"Video cần chấm: {os.path.basename(self.selected_user_video)}  |  📚 Kho Sheet Nhạc đã nạp sẵn trong hệ thống (Chỉ cần nạp Video để chấm)",
            font=("Segoe UI", 9),
            bg="#161b22",
            fg="#8b949e",
        )
        self.video_lbl.pack(anchor="w")

        btn_frame = tk.Frame(header, bg="#161b22")
        btn_frame.pack(side=tk.RIGHT, padx=16)

        self.choose_btn = tk.Button(
            btn_frame,
            text="📂  CHỌN VIDEO",
            font=("Segoe UI", 10, "bold"),
            bg="#1f6feb",
            fg="white",
            padx=14,
            pady=6,
            bd=0,
            cursor="hand2",
            activebackground="#388bfd",
            activeforeground="white",
            command=self.select_user_video,
        )
        self.choose_btn.pack(side=tk.LEFT, padx=6)

        self.run_btn = tk.Button(
            btn_frame,
            text="▶  CHẠY CHẤM ĐIỂM",
            font=("Segoe UI", 10, "bold"),
            bg="#238636",
            fg="white",
            padx=16,
            pady=6,
            bd=0,
            cursor="hand2",
            activebackground="#2ea043",
            activeforeground="white",
            command=self.start_evaluation,
        )
        self.run_btn.pack(side=tk.LEFT, padx=6)

        self.reset_btn = tk.Button(
            btn_frame,
            text="🔄  RESET HỌC VIÊN",
            font=("Segoe UI", 10, "bold"),
            bg="#b08800",
            fg="white",
            padx=14,
            pady=6,
            bd=0,
            cursor="hand2",
            activebackground="#d4a017",
            activeforeground="white",
            command=self.reset_data,
        )
        self.reset_btn.pack(side=tk.LEFT, padx=6)

        self.export_btn = tk.Button(
            btn_frame,
            text="💾  XUẤT BÁO CÁO",
            font=("Segoe UI", 10, "bold"),
            bg="#6e40c9",
            fg="white",
            padx=14,
            pady=6,
            bd=0,
            cursor="hand2",
            activebackground="#8957e5",
            activeforeground="white",
            command=self.export_report,
        )
        self.export_btn.pack(side=tk.LEFT, padx=6)

    def _build_main_panel(self) -> None:
        main = tk.Frame(self.root, bg="#0d1117")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 0))

        left_col = tk.Frame(main, bg="#0d1117")
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_col = tk.Frame(main, bg="#0d1117", width=480)
        right_col.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right_col.pack_propagate(False)

        self._build_result_panel(left_col)
        self._build_chart_panel(left_col)
        self._build_timeline_panel(left_col)
        self._build_review_panel(right_col)

    def _build_result_panel(self, parent: tk.Widget) -> None:
        frame = tk.LabelFrame(
            parent,
            text="  📊  Báo Cáo Phân Tích Chuyên Môn  ",
            font=("Segoe UI", 10, "bold"),
            bg="#161b22",
            fg="#8b949e",
            bd=1,
            relief=tk.SOLID,
            labelanchor="nw",
        )
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.output_text = scrolledtext.ScrolledText(
            frame,
            font=("Cascadia Code", 10),
            bg="#0d1117",
            fg="#c9d1d9",
            bd=0,
            insertbackground="#58a6ff",
            selectbackground="#1f6feb",
            wrap=tk.WORD,
            height=10,
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _build_chart_panel(self, parent: tk.Widget) -> None:
        chart_frame = tk.LabelFrame(
            parent,
            text="  📈  Biểu Đồ Điểm Số 10 Ngón Tay (Trực Quan Hóa Dữ Liệu)  ",
            font=("Segoe UI", 10, "bold"),
            bg="#161b22",
            fg="#8b949e",
            bd=1,
            relief=tk.SOLID,
            labelanchor="nw",
        )
        chart_frame.pack(fill=tk.X, pady=(0, 5))
        self.chart_canvas = tk.Canvas(
            chart_frame, bg="#0d1117", height=165, bd=0, highlightthickness=0
        )
        self.chart_canvas.pack(fill=tk.X, padx=6, pady=6)
        self.chart_canvas.bind("<Configure>", lambda e: self.draw_finger_bar_chart())

    def _build_timeline_panel(self, parent: tk.Widget) -> None:
        outer = tk.LabelFrame(
            parent,
            text="  🕐  Timeline Sự Kiện Gõ Phím  (Click ô bất kỳ để xem Video & Biểu đồ Sóng)  ",
            font=("Segoe UI", 10, "bold"),
            bg="#161b22",
            fg="#8b949e",
            bd=1,
            relief=tk.SOLID,
            labelanchor="nw",
        )
        outer.pack(fill=tk.X, pady=(0, 5))

        legend = tk.Frame(outer, bg="#161b22")
        legend.pack(anchor="w", padx=8, pady=(4, 0))
        for color, label in [
            ("#238636", "✔ Đạt (>=75đ)"),
            ("#da3633", "✘ Lỗi (<75đ)"),
            ("#30363d", "— Không tay"),
        ]:
            dot = tk.Label(
                legend, text="■ ", font=("Segoe UI", 9, "bold"), bg="#161b22", fg=color
            )
            dot.pack(side=tk.LEFT)
            tk.Label(
                legend,
                text=label + "   ",
                font=("Segoe UI", 9),
                bg="#161b22",
                fg="#8b949e",
            ).pack(side=tk.LEFT)

        canvas_frame = tk.Frame(outer, bg="#161b22")
        canvas_frame.pack(fill=tk.X, padx=8, pady=6)

        self.timeline_canvas = tk.Canvas(
            canvas_frame, bg="#0d1117", height=58, bd=0, highlightthickness=0
        )
        tl_scroll = tk.Scrollbar(
            canvas_frame, orient=tk.HORIZONTAL, command=self.timeline_canvas.xview
        )
        self.timeline_canvas.configure(xscrollcommand=tl_scroll.set)
        tl_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.timeline_canvas.pack(fill=tk.X)

        self.tl_inner = tk.Frame(self.timeline_canvas, bg="#0d1117")
        self.tl_inner_window = self.timeline_canvas.create_window(
            (0, 0), window=self.tl_inner, anchor="nw"
        )
        self.tl_inner.bind(
            "<Configure>",
            lambda e: self.timeline_canvas.configure(
                scrollregion=self.timeline_canvas.bbox("all")
            ),
        )

    def _build_review_panel(self, parent: tk.Widget) -> None:
        review = tk.LabelFrame(
            parent,
            text="  🔍  Review Chi Tiết & Biểu Đồ Sóng Quỹ Đạo  ",
            font=("Segoe UI", 10, "bold"),
            bg="#161b22",
            fg="#8b949e",
            bd=1,
            relief=tk.SOLID,
            labelanchor="nw",
        )
        review.pack(fill=tk.BOTH, expand=True)

        self.review_second_label = tk.Label(
            review,
            text="Chọn một ô trên Timeline để đối chiếu Video & Biểu đồ Sóng",
            font=("Segoe UI", 10, "bold"),
            bg="#161b22",
            fg="#58a6ff",
            wraplength=450,
            justify=tk.LEFT,
        )
        self.review_second_label.pack(padx=10, pady=(6, 2), anchor="w")

        self.review_advice_label = tk.Label(
            review,
            text="",
            font=("Segoe UI", 9, "italic"),
            bg="#161b22",
            fg="#f0883e",
            wraplength=450,
            justify=tk.LEFT,
        )
        self.review_advice_label.pack(padx=10, pady=(0, 4), anchor="w")

        imgs_frame = tk.Frame(review, bg="#161b22")
        imgs_frame.pack(fill=tk.X, padx=8, pady=2)

        user_frame_col = tk.Frame(imgs_frame, bg="#161b22")
        user_frame_col.pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Label(
            user_frame_col,
            text="👤 Học Viên (Skeleton)",
            font=("Segoe UI", 9, "bold"),
            bg="#161b22",
            fg="#da3633",
        ).pack()
        self.user_img_label = tk.Label(
            user_frame_col, bg="#0d1117", bd=1, relief=tk.SOLID
        )
        self.user_img_label.pack(padx=4, pady=2)

        expert_frame_col = tk.Frame(imgs_frame, bg="#161b22")
        expert_frame_col.pack(side=tk.RIGHT, expand=True, fill=tk.X)
        tk.Label(
            expert_frame_col,
            text="⭐ Chuyên Gia (Skeleton)",
            font=("Segoe UI", 9, "bold"),
            bg="#161b22",
            fg="#238636",
        ).pack()
        self.expert_img_label = tk.Label(
            expert_frame_col, bg="#0d1117", bd=1, relief=tk.SOLID
        )
        self.expert_img_label.pack(padx=4, pady=2)

        # Canvas vẽ biểu đồ sóng đối chiếu (Waveform Comparison)
        wave_lbl = tk.Label(
            review,
            text="📉 Biểu đồ Sóng Biên độ Ngón tay (Xanh lá: Chuyên gia | Đỏ: Học viên)",
            font=("Segoe UI", 8, "bold"),
            bg="#161b22",
            fg="#8b949e",
        )
        wave_lbl.pack(padx=10, pady=(4, 0), anchor="w")

        self.wave_canvas = tk.Canvas(
            review, bg="#0d1117", height=110, bd=0, highlightthickness=1, highlightbackground="#30363d"
        )
        self.wave_canvas.pack(fill=tk.X, padx=10, pady=4)

        self.review_detail_text = scrolledtext.ScrolledText(
            review,
            font=("Segoe UI", 9),
            bg="#0d1117",
            fg="#c9d1d9",
            bd=0,
            height=8,
            wrap=tk.WORD,
        )
        self.review_detail_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

    def _build_status_bar(self) -> None:
        self.status_bar = tk.Frame(self.root, bg="#161b22", pady=5)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_label = tk.Label(
            self.status_bar,
            text="Sẵn sàng...",
            font=("Segoe UI", 9),
            bg="#161b22",
            fg="#8b949e",
        )
        self.status_label.pack(side=tk.LEFT, padx=15)

    def select_sheet_music(self) -> None:
        """Cho phép người dùng nạp Ảnh chụp Sheet nhạc (.png, .jpg) hoặc tệp Sheet (.json, .txt, .musicxml)."""
        os.makedirs(self.sheets_dir, exist_ok=True)
        file_path = filedialog.askopenfilename(
            title="Chọn Ảnh/PDF Sheet Nhạc (.png/.jpg/.pdf) hoặc Tệp Bản Nhạc (.json/.txt/.musicxml)",
            initialdir=self.sheets_dir,
            filetypes=[
                ("Sheet Music Files", "*.png *.jpg *.jpeg *.pdf *.json *.txt *.musicxml *.xml"),
                ("All Files", "*.*"),
            ],
        )
        if file_path:
            self.selected_sheet_path = file_path
            self.eval_mode.set("sheet")
            self.video_lbl.config(
                text=f"Video: {os.path.basename(self.selected_user_video)}  |  Sheet nhạc: {os.path.basename(file_path)}"
            )
            self.status_label.config(
                text=f"Đã nạp Sheet nhạc: {os.path.basename(file_path)}. Nhấn '▶ CHẠY CHẤM ĐIỂM' để chấm theo Sheet!",
                fg="#58a6ff",
            )

    def select_user_video(self) -> None:
        """Cho phép người dùng chọn tệp video học viên bất kỳ từ máy tính."""
        file_path = filedialog.askopenfilename(
            title="Chọn Video Luyện Tập Của Học Viên",
            initialdir=self.videos_dir,
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")],
        )
        if file_path:
            self.selected_user_video = file_path
            target_default = os.path.join(self.videos_dir, "user_sample.mp4")
            try:
                if os.path.abspath(file_path) != os.path.abspath(target_default):
                    os.makedirs(self.videos_dir, exist_ok=True)
                    shutil.copy2(file_path, target_default)
                # Xóa cache dữ liệu học viên cũ để quét video mới
                user_json = os.path.join(self.data_dir, "user_data.json")
                if os.path.exists(user_json):
                    os.remove(user_json)
                self.user_raw_data = []
                self.video_lbl.config(
                    text=f"Video cần chấm: {os.path.basename(file_path)}  |  📚 Kho Sheet Nhạc đã nạp sẵn trong hệ thống"
                )
                self.status_label.config(
                    text=f"Đã chọn video mới: {os.path.basename(file_path)}. Nhấn 'CHẠY CHẤM ĐIỂM' để phân tích!",
                    fg="#58a6ff",
                )
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể nạp video: {e}")

    def export_report(self) -> None:
        """Xuất nội dung báo cáo chấm điểm và thống kê chi tiết ra tệp .txt."""
        content = self.output_text.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning(
                "Chưa có dữ liệu", "Vui lòng chạy chấm điểm trước khi xuất báo cáo!"
            )
            return

        save_path = filedialog.asksaveasfilename(
            title="Lưu Báo Cáo Đánh Giá Piano AI",
            defaultextension=".txt",
            initialfile="Bao_Cao_Ket_Qua_PianoAI.txt",
            filetypes=[("Text Files", "*.txt"), ("JSON Files", "*.json")],
        )
        if not save_path:
            return

        try:
            if save_path.endswith(".json") and self.summary_data:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "summary": self.summary_data,
                            "timeline": self.timeline_data,
                        },
                        f,
                        indent=4,
                        ensure_ascii=False,
                    )
            else:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(content)
            messagebox.showinfo(
                "Xuất báo cáo thành công", f"Đã lưu báo cáo tại:\n{save_path}"
            )
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu báo cáo: {e}")

    def autoload_existing_results(self) -> None:
        """Tự động hiển thị biểu đồ và timeline nếu trong thư mục data đã có kết quả."""
        timeline_path = os.path.join(self.data_dir, "timeline_report.json")
        summary_path = os.path.join(self.data_dir, "evaluation_summary.json")

        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    self.summary_data = json.load(f)
                self.draw_finger_bar_chart()
            except Exception:
                pass

        if os.path.exists(timeline_path):
            try:
                with open(timeline_path, "r", encoding="utf-8") as f:
                    self.timeline_data = json.load(f)
                self.build_timeline()
                if not self.output_text.get(1.0, tk.END).strip():
                    self._render_summary_to_console()
                # Tự động hiển thị sự kiện lỗi đầu tiên (nếu có) để người dùng thấy ngay
                first_err = next(
                    (t for t in self.timeline_data if t.get("status") == "ERROR"),
                    self.timeline_data[0] if self.timeline_data else None,
                )
                if first_err:
                    self.on_cell_click(first_err.get("event_id", 1))
            except Exception:
                pass

    def _render_summary_to_console(self) -> None:
        if not self.summary_data:
            return
        final_score = self.summary_data.get("final_score", 0.0)
        left_avg = self.summary_data.get("left_hand_average", 0.0)
        right_avg = self.summary_data.get("right_hand_average", 0.0)
        scores = self.summary_data.get("scores", {})
        bad_fingers = self.summary_data.get("bad_fingers", [])
        err_secs = self.summary_data.get("error_seconds", [])
        total_ev = self.summary_data.get("total_events", len(self.timeline_data))

        lines = [
            "=" * 54,
            "      BÁO CÁO PHÂN TÍCH CHI TIẾT CẢ HAI TAY (PIANO AI)",
            "=" * 54,
            f" => ĐIỂM TỔNG THỂ: {final_score:.2f} / 100",
            f"    • Trung bình Tay Trái : {left_avg:.2f} điểm | Tay Phải : {right_avg:.2f} điểm",
            "-" * 54,
            " [CHI TIẾT TỪNG TAY]",
        ]
        for hand_side in ["Left", "Right"]:
            if scores.get(hand_side):
                side_name = "TAY TRÁI" if hand_side == "Left" else "TAY PHẢI"
                lines.append(f"\n --- {side_name} ---")
                for name, sc in scores[hand_side].items():
                    disp = FINGER_DISPLAY_NAMES.get(name, name)
                    lines.append(f"  - {disp:<12}: {sc:.2f} điểm")
        lines.append("-" * 54)
        if bad_fingers:
            lines.append(f" ❌ Các ngón cần lưu ý sửa: {', '.join(bad_fingers)}")
        else:
            lines.append(" 🌟 Tất cả các ngón tay đều đạt ngưỡng kỹ thuật tốt (>= 70 điểm)!")
        lines.append(
            f" ⚠️ Phát hiện {len(err_secs)} điểm sai trên tổng số {total_ev} sự kiện gõ phím."
        )
        lines.append("=" * 54 + "\n")
        self.output_text.delete(1.0, tk.END)
        self.append_output("\n".join(lines))

    def draw_finger_bar_chart(self) -> None:
        """Vẽ biểu đồ cột trực quan hóa điểm số 10 ngón tay (5 Trái + 5 Phải) trên Canvas."""
        self.chart_canvas.delete("all")
        w = self.chart_canvas.winfo_width()
        h = self.chart_canvas.winfo_height()
        if w < 100:
            w = 780
        if h < 50:
            h = 165

        scores = self.summary_data.get("scores", {})
        if not scores or (not scores.get("Left") and not scores.get("Right")):
            self.chart_canvas.create_text(
                w // 2,
                h // 2,
                text="Chưa có dữ liệu biểu đồ — Hãy bấm '▶ CHẠY CHẤM ĐIỂM' để tạo biểu đồ 10 ngón tay",
                fill="#8b949e",
                font=("Segoe UI", 10, "italic"),
            )
            return

        # Vẽ đường tham chiếu 80 điểm (Tốt) và 60 điểm (Đạt)
        top_pad = 24
        bottom_pad = 36
        left_pad = 45
        right_pad = 150
        plot_w = max(200, w - left_pad - right_pad)
        plot_h = max(60, h - top_pad - bottom_pad)

        for ref_val, ref_color, ref_lbl in [
            (100, "#21262d", "100"),
            (80, "#238636", "80"),
            (60, "#9e6a03", "60"),
            (0, "#30363d", "0"),
        ]:
            y_pos = top_pad + plot_h * (1.0 - ref_val / 100.0)
            self.chart_canvas.create_line(
                left_pad, y_pos, left_pad + plot_w, y_pos, fill=ref_color, dash=(3, 3)
            )
            self.chart_canvas.create_text(
                left_pad - 8,
                y_pos,
                text=ref_lbl,
                anchor="e",
                fill="#8b949e",
                font=("Segoe UI", 8),
            )

        items = []
        short_names = {
            "Ngon cai": "Cái",
            "Ngon tro": "Trỏ",
            "Ngon giua": "Giữa",
            "Ngon ap ut": "Áp út",
            "Ngon ut": "Út",
        }
        for hand_side, prefix in [("Left", "T."), ("Right", "P.")]:
            for fname in FINGER_NAMES:
                val = float(scores.get(hand_side, {}).get(fname, 0.0))
                items.append((f"{prefix}{short_names.get(fname, fname)}", val, hand_side))

        n_bars = len(items)
        slot_w = plot_w / max(1, n_bars)
        bar_w = min(36, slot_w * 0.62)

        for idx, (label, val, hand_side) in enumerate(items):
            cx = left_pad + idx * slot_w + slot_w / 2.0
            bar_h = (val / 100.0) * plot_h
            x0 = cx - bar_w / 2.0
            x1 = cx + bar_w / 2.0
            y1 = top_pad + plot_h
            y0 = y1 - bar_h

            color = "#238636" if val >= 80 else ("#d29922" if val >= 65 else "#da3633")
            self.chart_canvas.create_rectangle(
                x0, y0, x1, y1, fill=color, outline="#f0f6fc", width=1
            )
            self.chart_canvas.create_text(
                cx,
                max(10, y0 - 9),
                text=f"{val:.0f}",
                fill="#c9d1d9",
                font=("Segoe UI", 8, "bold"),
            )
            lbl_color = "#58a6ff" if hand_side == "Left" else "#d2a8ff"
            self.chart_canvas.create_text(
                cx,
                y1 + 13,
                text=label,
                fill=lbl_color,
                font=("Segoe UI", 8, "bold"),
            )

        # Nhãn phân biệt Tay Trái và Tay Phải
        self.chart_canvas.create_text(
            left_pad + plot_w * 0.25,
            y1 + 27,
            text="◄ TAY TRÁI (Left Hand) ►",
            fill="#58a6ff",
            font=("Segoe UI", 8),
        )
        self.chart_canvas.create_text(
            left_pad + plot_w * 0.75,
            y1 + 27,
            text="◄ TAY PHẢI (Right Hand) ►",
            fill="#d2a8ff",
            font=("Segoe UI", 8),
        )

        # Khung Huy hiệu Điểm Tổng Thể bên phải biểu đồ
        final_sc = float(self.summary_data.get("final_score", 0.0))
        badge_x = left_pad + plot_w + (right_pad // 2)
        badge_y = h // 2
        badge_col = (
            "#238636" if final_sc >= 80 else ("#d29922" if final_sc >= 60 else "#da3633")
        )
        self.chart_canvas.create_oval(
            badge_x - 52,
            badge_y - 52,
            badge_x + 52,
            badge_y + 52,
            fill="#161b22",
            outline=badge_col,
            width=4,
        )
        self.chart_canvas.create_text(
            badge_x,
            badge_y - 12,
            text=f"{final_sc:.1f}",
            fill="white",
            font=("Segoe UI", 18, "bold"),
        )
        self.chart_canvas.create_text(
            badge_x,
            badge_y + 16,
            text="ĐIỂM TỔNG",
            fill="#8b949e",
            font=("Segoe UI", 8, "bold"),
        )

    def draw_waveform_comparison(
        self, exp_wave: List[float], usr_wave: List[float]
    ) -> None:
        """Vẽ biểu đồ sóng đối chiếu biên độ ngón tay Chuyên gia vs. Học viên."""
        self.wave_canvas.delete("all")
        w = self.wave_canvas.winfo_width()
        h = self.wave_canvas.winfo_height()
        if w < 100:
            w = 430
        if h < 40:
            h = 110

        if not exp_wave or not usr_wave:
            self.wave_canvas.create_text(
                w // 2,
                h // 2,
                text="Không có dữ liệu sóng cho sự kiện này",
                fill="#8b949e",
                font=("Segoe UI", 9, "italic"),
            )
            return

        pad_x = 18
        pad_y = 16
        plot_w = max(50, w - 2 * pad_x)
        plot_h = max(30, h - 2 * pad_y)

        all_vals = exp_wave + usr_wave
        min_v = min(all_vals)
        max_v = max(all_vals)
        span = (max_v - min_v) if (max_v - min_v) > 1e-4 else 1.0

        # Lưới ngang
        for frac in [0.0, 0.5, 1.0]:
            gy = pad_y + frac * plot_h
            self.wave_canvas.create_line(
                pad_x, gy, pad_x + plot_w, gy, fill="#21262d", dash=(2, 2)
            )

        def to_points(wave: List[float]) -> List[float]:
            pts: List[float] = []
            n = len(wave)
            for i, val in enumerate(wave):
                px = pad_x + (i / max(1, n - 1)) * plot_w
                py = pad_y + plot_h * (1.0 - (val - min_v) / span)
                pts.extend([px, py])
            return pts

        exp_pts = to_points(exp_wave)
        usr_pts = to_points(usr_wave)

        if len(exp_pts) >= 4:
            self.wave_canvas.create_line(*exp_pts, fill="#3fb950", width=2, smooth=True)
        if len(usr_pts) >= 4:
            self.wave_canvas.create_line(*usr_pts, fill="#f85149", width=2, smooth=True)

        # Đánh dấu tâm sự kiện (Note Onset Peak)
        mid_x = pad_x + plot_w / 2.0
        self.wave_canvas.create_line(
            mid_x, pad_y, mid_x, pad_y + plot_h, fill="#58a6ff", dash=(3, 2)
        )
        self.wave_canvas.create_text(
            mid_x + 4,
            pad_y + 6,
            text="Onset",
            anchor="w",
            fill="#58a6ff",
            font=("Segoe UI", 7),
        )

    def append_output(self, text: str) -> None:
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)

    def reset_data(self) -> None:
        try:
            if os.path.exists(self.data_dir):
                for filename in os.listdir(self.data_dir):
                    if filename.endswith(".json") and filename not in (
                        "expert_data.json",
                        "keyboard_layout.json",
                        "sheet_music_database.json",
                    ):
                        os.remove(os.path.join(self.data_dir, filename))
            self.output_text.delete(1.0, tk.END)
            self.timeline_data = []
            self.summary_data = {}
            self.user_raw_data = []
            self.clear_timeline()
            self.clear_review_panel()
            self.draw_finger_bar_chart()
            self.append_output(
                "Đã xóa dữ liệu học viên cũ (Giữ lại cache chuyên gia & bàn phím)!\n"
                "Bấm '▶ CHẠY CHẤM ĐIỂM' để quét lại video học viên.\n"
            )
            messagebox.showinfo(
                "Thành công",
                "Đã xóa dữ liệu học viên.\nNhấn 'Chạy Chấm Điểm' để tiếp tục!",
            )
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xóa dữ liệu: {e}")

    def clear_timeline(self) -> None:
        for widget in self.tl_inner.winfo_children():
            widget.destroy()
        self.timeline_cells = []

    def clear_review_panel(self) -> None:
        self.stop_video_loop()
        self.review_second_label.config(
            text="Chọn một ô trên Timeline để đối chiếu Video & Biểu đồ Sóng"
        )
        self.review_advice_label.config(text="")
        self.user_img_label.config(image="")
        self.user_img_label.image = None
        self.expert_img_label.config(image="")
        self.expert_img_label.image = None
        self.wave_canvas.delete("all")
        self.review_detail_text.delete(1.0, tk.END)

    def run_script(self) -> None:
        self.append_output("⏳ Đang khởi động hệ thống Computer Vision & DSP...\n\n")
        evaluator_path = os.path.join(self.current_dir, "src", "evaluator.py")

        try:
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            process = subprocess.Popen(
                [
                    sys.executable,
                    evaluator_path,
                    self.selected_user_video,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=flags,
            )
            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    clean_line = strip_ansi(line)
                    self.root.after(0, self.append_output, clean_line)
                process.stdout.close()
            process.wait()
            self.root.after(0, self.on_evaluation_done)
        except Exception as e:
            self.root.after(0, self.append_output, f"\n❌ Lỗi hệ thống: {str(e)}\n")
            self.root.after(0, self.finish_evaluation)

    def on_evaluation_done(self) -> None:
        self.finish_evaluation()
        summary_path = os.path.join(self.data_dir, "evaluation_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                self.summary_data = json.load(f)
            self.draw_finger_bar_chart()

        timeline_path = os.path.join(self.data_dir, "timeline_report.json")
        if os.path.exists(timeline_path):
            with open(timeline_path, "r", encoding="utf-8") as f:
                self.timeline_data = json.load(f)
            self.build_timeline()
            first_err = next(
                (t for t in self.timeline_data if t.get("status") == "ERROR"),
                self.timeline_data[0] if self.timeline_data else None,
            )
            if first_err:
                self.on_cell_click(first_err.get("event_id", 1))
        else:
            self.append_output(
                "\n⚠️ Không tìm thấy file timeline_report.json. Kiểm tra lại quá trình chấm điểm.\n"
            )

    def build_timeline(self) -> None:
        self.clear_timeline()
        self.load_raw_data()
        cell_w = 44
        cell_h = 44

        for idx, item in enumerate(self.timeline_data):
            s = item.get("second", 0)
            event_id = item.get("event_id", idx + 1)
            status = item["status"]
            score = item.get("score", 100)

            if status == "NO_HAND":
                bg_color = "#30363d"
                fg_color = "#6e7681"
                hover_color = "#484f58"
            elif status == "ERROR":
                bg_color = "#da3633"
                fg_color = "white"
                hover_color = "#ff4d4d"
            else:
                bg_color = "#238636"
                fg_color = "white"
                hover_color = "#2ea043"

            note_label_text = item.get("sheet_note") or f"{s}s"
            if not item.get("sheet_note") and "event_id" in item:
                hand_side = item.get("error_hand")
                finger_name = item.get("error_finger")
                user_frame_idx = item.get("user_frame", 0)
                expert_frame_idx = item.get("expert_frame", 0)
                if hand_side and finger_name:
                    # Ưu tiên lấy nốt nhạc trực tiếp từ dữ liệu bàn tay học viên (không phụ thuộc video mẫu)
                    x_pos = self.get_finger_x(
                        self.user_raw_data, user_frame_idx, hand_side, finger_name
                    )
                    if x_pos is None:
                        x_pos = self.get_finger_x(
                            self.expert_raw_data, expert_frame_idx, hand_side, finger_name
                        )
                    if x_pos is not None:
                        full_note = self.map_x_to_note(x_pos)
                        m = re.search(r"\(([A-G][0-9])", full_note)
                        if m:
                            note_label_text = m.group(1)
                        else:
                            note_label_text = full_note.split()[0][:3]

            cell_outer = tk.Frame(self.tl_inner, bg=bg_color, padx=1, pady=1)
            cell_outer.grid(row=0, column=idx, padx=2, pady=4)

            cell = tk.Frame(
                cell_outer, bg=bg_color, width=cell_w, height=cell_h, cursor="hand2"
            )
            cell.grid_propagate(False)
            cell.grid(row=0, column=0)

            lbl_s = tk.Label(
                cell,
                text=note_label_text,
                font=("Segoe UI", 9, "bold"),
                bg=bg_color,
                fg=fg_color,
                width=5,
            )
            lbl_s.grid(row=0, column=0, pady=(4, 0))

            score_text = f"{int(score)}" if status != "NO_HAND" else "—"
            lbl_sc = tk.Label(
                cell,
                text=score_text,
                font=("Segoe UI", 7),
                bg=bg_color,
                fg=fg_color,
                width=5,
            )
            lbl_sc.grid(row=1, column=0)

            # Cho phép click vào cả ô Đỏ (ERROR) và ô Xanh (OK) để đối chiếu bất kỳ nốt nào
            for widget in [cell_outer, cell, lbl_s, lbl_sc]:
                widget.bind(
                    "<Button-1>", lambda e, eid=event_id: self.on_cell_click(eid)
                )
                widget.bind(
                    "<Enter>", lambda e, c=cell_outer, hc=hover_color: c.config(bg=hc)
                )
                widget.bind(
                    "<Leave>", lambda e, c=cell_outer, bc=bg_color: c.config(bg=bc)
                )

            self.timeline_cells.append(cell)

    def on_cell_click(self, event_id: int) -> None:
        self.stop_video_loop()
        self.selected_second = event_id
        item = next(
            (
                t
                for t in self.timeline_data
                if t.get("event_id") == event_id or t.get("second") == event_id
            ),
            None,
        )
        if not item:
            return

        self.load_raw_data()

        hand_side = item.get("error_hand")
        finger_name = item.get("error_finger")
        user_frame_idx = item.get("user_frame", 0)
        expert_frame_idx = item.get("expert_frame", 0)

        self.curr_expert_note = "Không phát hiện"
        self.curr_user_note = "Không phát hiện"

        if hand_side and finger_name:
            x_exp = self.get_finger_x(
                self.expert_raw_data, expert_frame_idx, hand_side, finger_name
            )
            x_usr = self.get_finger_x(
                self.user_raw_data, user_frame_idx, hand_side, finger_name
            )
            if x_exp is not None:
                self.curr_expert_note = self.map_x_to_note(x_exp)
            if x_usr is not None:
                self.curr_user_note = self.map_x_to_note(x_usr)

        hand_viet = "Tay Trái" if hand_side == "Left" else "Tay Phải"
        finger_viet = FINGER_DISPLAY_NAMES.get(finger_name, finger_name or "N/A")
        score = item.get("score", 0)
        advice = item.get("advice", "")
        status = item.get("status", "OK")

        s = item.get("second", 0)
        icon = "🔴" if status == "ERROR" else "🟢"
        self.review_second_label.config(
            text=f"{icon}  Sự kiện #{item.get('event_id', 1)} ({s}s)  —  Điểm: {score:.1f}/100  —  {hand_viet} / {finger_viet}"
        )
        self.review_advice_label.config(text=f"💡  {advice}")

        # Vẽ biểu đồ sóng quỹ đạo đối chiếu
        self.draw_waveform_comparison(
            item.get("expert_wave", []), item.get("user_wave", [])
        )

        self.review_detail_text.delete(1.0, tk.END)
        self.review_detail_text.insert(tk.END, self._generate_review_text(item))

        self.start_video_loop(expert_frame_idx, user_frame_idx)

    def _generate_review_text(self, item: Dict[str, Any]) -> str:
        lines = []
        second = item.get("second", 0)
        status = item["status"]
        score = item.get("score", 0)
        hand_viet = "Tay Trái" if item.get("error_hand") == "Left" else "Tay Phải"
        raw_finger = item.get("error_finger", "Không xác định")
        finger_viet = FINGER_DISPLAY_NAMES.get(raw_finger, raw_finger)
        advice = item.get("advice", "")

        expert_note = getattr(self, "curr_expert_note", "Không phát hiện")
        user_note = getattr(self, "curr_user_note", "Không phát hiện")

        joint_angle = item.get("joint_angle")
        item_mode = item.get("mode", self.eval_mode.get())
        sheet_note = item.get("sheet_note")
        played_note = item.get("played_note")
        sheet_finger_num = item.get("sheet_finger_num")

        lines.append(
            f"📍 Thời điểm gõ phím: {second}s (Frame: {item.get('user_frame', 0)})"
        )
        lines.append(f"📊 Điểm kỹ thuật & bản nhạc: {score:.1f} / 100")
        if joint_angle is not None:
            lines.append(
                f"📐 Góc cong khớp ngón tay 3D: {joint_angle:.1f}° (Chuẩn Piano: 120° – 160°)"
            )
        if sheet_note:
            lines.append(
                f"🎼 Nốt yêu cầu trên Sheet nhạc : {item.get('sheet_full_note', sheet_note)} (Ngón số {sheet_finger_num})"
            )
            lines.append(
                f"🎹 Nốt học viên thực tế đã đánh: {item.get('played_full_note', played_note or user_note)}"
            )
        else:
            lines.append(f"🎹 Nốt nhạc nhận diện trên phím : {user_note}")
            if item_mode == "reference":
                lines.append(f"🎹 Nốt chuẩn (Video mẫu)      : {expert_note}")
        lines.append("")

        if status == "ERROR":
            lines.append("❌ PHÂN TÍCH NGUYÊN NHÂN LỖI:")
            if item_mode == "standalone" and joint_angle is not None:
                if joint_angle > 165.0:
                    lines.append(
                        f"   ⚠️ LỖI BẸT NGÓN / SẬP ĐỐT: Góc khớp đạt {joint_angle:.1f}° (> 165°), ngón tay duỗi quá thẳng khi bấm phím."
                    )
                elif joint_angle < 110.0:
                    lines.append(
                        f"   ⚠️ LỖI QUẶP NGÓN: Góc khớp đạt {joint_angle:.1f}° (< 110°), đốt ngón bị co quặp gây cứng cổ tay."
                    )
                else:
                    lines.append(
                        "   ⚠️ LỖI DAO ĐỘNG CỔ TAY: Cổ tay nhấp nhổm hoặc tốc độ nhấn phím chưa dứt khoát."
                    )
            elif expert_note != "Không phát hiện" and user_note != "Không phát hiện":
                if expert_note != user_note:
                    lines.append(
                        "   ⚠️ Học viên đánh SAI VỊ TRÍ PHÍM (Lệch tay sang phím bên cạnh)."
                    )
                else:
                    lines.append(
                        "   ⚠️ Học viên đặt ĐÚNG VỊ TRÍ PHÍM nhưng SAI KỸ THUẬT / LỆCH NHỊP."
                    )
            else:
                lines.append("   ⚠️ Biên độ chuyển động ngón tay không khớp hoặc lệch nhịp.")

            lines.append(f"   ► Ngón tay cần sửa: {hand_viet} — {finger_viet}")
            lines.append("")
            lines.append("🎯 BÀI TẬP KHẮC PHỤC:")
            if "cai" in raw_finger.lower():
                lines.append("   • Ngón cái cần đặt thẳng, tránh gập sâu vào lòng bàn tay.")
                lines.append("   • Tập giữ ngón cái song song với mặt phím, cổ tay thả lỏng.")
            elif "tro" in raw_finger.lower():
                lines.append("   • Ngón trỏ cần bấm phím bằng đầu ngón, giữ đốt ngón cong tròn.")
                lines.append("   • Giữ khuỷu tay ổn định, thả lỏng vai khi gõ phím.")
            elif "giua" in raw_finger.lower():
                lines.append("   • Ngón giữa là trục bàn tay — cần giữ độ cong tự nhiên.")
                lines.append("   • Tránh duỗi thẳng ngón giữa khi đánh các nốt móc đơn liên tiếp.")
            elif "ap" in raw_finger.lower():
                lines.append("   • Ngón áp út thường yếu nhất — luyện thêm bài tập Hanon số 3, 4.")
                lines.append("   • Tập nhấc độc lập ngón áp út 5 phút mỗi ngày để tăng lực.")
            elif "ut" in raw_finger.lower():
                lines.append("   • Ngón út cần đứng vững trên đầu ngón, tránh sập đốt thứ nhất.")
                lines.append("   • Giữ cổ tay ngang bằng để truyền lực đều xuống ngón út.")
        else:
            lines.append("✅ ĐẠT YÊU CẦU KỸ THUẬT:")
            lines.append(f"   • {advice}")
            lines.append(
                "   • Đường sóng biên độ của Học viên (Đỏ) bám sát đường chuẩn Chuyên gia (Xanh lá)."
            )

        return "\n".join(lines)

    def load_raw_data(self) -> None:
        if not self.expert_raw_data:
            expert_path = os.path.join(self.data_dir, "expert_data.json")
            if os.path.exists(expert_path):
                try:
                    with open(expert_path, "r", encoding="utf-8") as f:
                        self.expert_raw_data = json.load(f)
                except Exception:
                    self.expert_raw_data = []

        if not self.user_raw_data:
            user_path = os.path.join(self.data_dir, "user_data.json")
            if os.path.exists(user_path):
                try:
                    with open(user_path, "r", encoding="utf-8") as f:
                        self.user_raw_data = json.load(f)
                except Exception:
                    self.user_raw_data = []

    def get_finger_x(
        self,
        data: List[Dict[str, Any]],
        frame_idx: int,
        hand_side: str,
        finger_name: str,
    ) -> Optional[float]:
        if not data or frame_idx >= len(data):
            return None
        frame_data = data[frame_idx]
        hands = frame_data.get("hands", [])
        if not hands:
            return None

        if len(hands) > 2:
            def hand_size(h: List[Dict[str, float]]) -> float:
                xs = [lm["x"] for lm in h]
                ys = [lm["y"] for lm in h]
                return (max(xs) - min(xs)) * (max(ys) - min(ys))

            hands = sorted(hands, key=hand_size, reverse=True)[:2]

        sorted_hands = sorted(hands, key=lambda h: h[0]["x"])
        target_hand = None
        if len(sorted_hands) == 1:
            if sorted_hands[0][0]["x"] > 0.5 and hand_side == "Right":
                target_hand = sorted_hands[0]
            elif sorted_hands[0][0]["x"] <= 0.5 and hand_side == "Left":
                target_hand = sorted_hands[0]
        elif len(sorted_hands) >= 2:
            target_hand = sorted_hands[0] if hand_side == "Left" else sorted_hands[1]

        if not target_hand:
            return None

        lm_id = FINGER_TIPS_MAP.get(finger_name, 8)
        if lm_id < len(target_hand):
            return float(target_hand[lm_id]["x"])
        return None

    def map_x_to_note(self, x: float) -> str:
        layout_path = os.path.join(self.data_dir, "keyboard_layout.json")
        if os.path.exists(layout_path):
            try:
                with open(layout_path, "r", encoding="utf-8") as f:
                    layout_data = json.load(f)
                boundaries = layout_data.get("key_boundaries", [])
                notes_list = layout_data.get("notes_list", DEFAULT_NOTES_LIST)
                if boundaries and notes_list:
                    idx = 0
                    while idx < len(boundaries) and x > boundaries[idx]:
                        idx += 1
                    idx = min(idx, len(notes_list) - 1)
                    return str(notes_list[idx])
            except Exception:
                pass

        x_min, x_max = 0.10, 0.90
        x_clamped = max(x_min, min(x_max, x))
        num_notes = len(DEFAULT_NOTES_LIST)
        idx = int((x_clamped - x_min) / (x_max - x_min) * (num_notes - 1))
        idx = max(0, min(num_notes - 1, idx))
        return DEFAULT_NOTES_LIST[idx]

    def on_close(self) -> None:
        self.stop_video_loop()
        self.root.destroy()

    def stop_video_loop(self) -> None:
        self.is_playing_loop = False
        if self.loop_timer_id:
            try:
                self.root.after_cancel(self.loop_timer_id)
            except Exception:
                pass
            self.loop_timer_id = None

        if self.cap_expert:
            self.cap_expert.release()
            self.cap_expert = None

        if self.cap_user:
            self.cap_user.release()
            self.cap_user = None

    def _render_ideal_posture_diagram(
        self,
        joint_angle: float = 142.0,
        sheet_note: Optional[str] = None,
        played_note: Optional[str] = None,
        finger_num: Optional[int] = None,
    ) -> np.ndarray:
        """
        Vẽ Khuông Nhạc 5 Dòng Kẻ Tương Tác (khi ở chế độ Chấm Theo Sheet Nhạc)
        hoặc Mô hình Vòm Tay Chuẩn Công Thái Học AI (khi ở chế độ Độc lập 3D).
        """
        img = np.full((115, 200, 3), (18, 24, 32), dtype=np.uint8)
        if sheet_note:
            # Vẽ Khuông nhạc 5 dòng kẻ trực quan và vị trí nốt nhạc trên Sheet
            top_y = 32
            sp = 10
            for i in range(5):
                ly = top_y + i * sp
                cv2.line(img, (14, ly), (186, ly), (130, 145, 160), 1, cv2.LINE_AA)

            scale_order = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5", "F5", "G5"]
            c4_y = top_y + 5 * sp
            idx_s = scale_order.index(sheet_note) if sheet_note in scale_order else 2
            ny = int(c4_y - idx_s * (sp / 2.0))
            if idx_s == 0:
                cv2.line(img, (88, ny), (112, ny), (180, 190, 200), 1, cv2.LINE_AA)

            is_match = (played_note is None) or (played_note == sheet_note)
            note_col = (80, 220, 100) if is_match else (80, 90, 250)
            cv2.ellipse(img, (100, ny), (7, 5), -20, 0, 360, note_col, -1, cv2.LINE_AA)
            cv2.line(img, (106, ny), (106, max(8, ny - 22)), note_col, 2, cv2.LINE_AA)

            cv2.putText(
                img,
                f"Sheet: {sheet_note} (Ngon {finger_num or 1})",
                (12, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (88, 186, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                f"Danh: {played_note or sheet_note} | Goc: {joint_angle:.0f} deg",
                (12, 104),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (100, 230, 120) if is_match else (255, 160, 90),
                1,
                cv2.LINE_AA,
            )
            return img

        # Vẽ mặt phím đàn Piano bên dưới
        cv2.rectangle(img, (15, 92), (185, 108), (235, 235, 235), -1)
        for kx in range(15, 186, 28):
            cv2.line(img, (kx, 92), (kx, 108), (40, 40, 40), 1)

        # Vẽ vòm ngón tay chuẩn Công thái học (Màu Xanh Lá: Cổ tay -> MCP -> PIP -> TIP)
        pts_ideal = np.array([[28, 72], [75, 42], [128, 48], [152, 88]], np.int32)
        for i in range(len(pts_ideal) - 1):
            cv2.line(
                img,
                tuple(pts_ideal[i]),
                tuple(pts_ideal[i + 1]),
                (54, 185, 63),
                3,
                cv2.LINE_AA,
            )
        for pt in pts_ideal:
            cv2.circle(img, tuple(pt), 4, (88, 166, 255), -1, cv2.LINE_AA)

        cv2.putText(
            img,
            "Chuan AI: 120-160 deg",
            (16, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (80, 220, 100),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"Goc HV: {joint_angle:.1f} deg",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (240, 180, 80),
            1,
            cv2.LINE_AA,
        )
        return img

    def start_video_loop(self, expert_frame: int, user_frame: int) -> None:
        self.stop_video_loop()

        user_video = os.path.join(self.videos_dir, "user_sample.mp4")
        expert_video = os.path.join(self.videos_dir, "expert_sample.mp4")

        if not os.path.exists(user_video):
            return

        self.cap_user = cv2.VideoCapture(user_video)
        if (
            self.eval_mode.get() == "reference"
            and os.path.exists(expert_video)
        ):
            self.cap_expert = cv2.VideoCapture(expert_video)
        else:
            self.cap_expert = None

        self.expert_start_f = max(0, expert_frame - 15)
        self.expert_end_f = expert_frame + 15
        self.expert_curr_f = self.expert_start_f

        self.user_start_f = max(0, user_frame - 15)
        self.user_end_f = user_frame + 15
        self.user_curr_f = self.user_start_f

        if self.cap_expert:
            self.cap_expert.set(cv2.CAP_PROP_POS_FRAMES, self.expert_start_f)
        self.cap_user.set(cv2.CAP_PROP_POS_FRAMES, self.user_start_f)

        self.is_playing_loop = True
        self.is_video_paused = False
        self.update_video_loop()

    def draw_landmarks_on_frame(
        self, frame: np.ndarray, frame_idx: int, data: List[Dict[str, Any]]
    ) -> None:
        if not data or frame_idx >= len(data):
            return
        frame_data = data[frame_idx]
        hands_landmarks = frame_data.get("hands", [])
        if not hands_landmarks:
            return

        h, w = frame.shape[:2]
        try:
            for hand in hands_landmarks:
                for start_id, end_id in MEDIAPIPE_CONNECTIONS:
                    if start_id < len(hand) and end_id < len(hand):
                        pt1 = (
                            int(hand[start_id]["x"] * w),
                            int(hand[start_id]["y"] * h),
                        )
                        pt2 = (int(hand[end_id]["x"] * w), int(hand[end_id]["y"] * h))
                        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

                for lm in hand:
                    pt = (int(lm["x"] * w), int(lm["y"] * h))
                    cv2.circle(frame, pt, 4, (0, 0, 255), -1)
        except Exception:
            pass

    def update_video_loop(self) -> None:
        if not self.is_playing_loop or self.is_video_paused:
            return

        reached_end = False
        if self.cap_expert and self.expert_curr_f > self.expert_end_f:
            reached_end = True
        elif self.cap_user and self.user_curr_f > self.user_end_f:
            reached_end = True

        if reached_end:
            self.is_video_paused = True

            def unpause() -> None:
                self.is_video_paused = False
                if self.cap_expert:
                    self.expert_curr_f = self.expert_start_f
                    self.cap_expert.set(cv2.CAP_PROP_POS_FRAMES, self.expert_start_f)
                if self.cap_user:
                    self.user_curr_f = self.user_start_f
                    self.cap_user.set(cv2.CAP_PROP_POS_FRAMES, self.user_start_f)
                self.update_video_loop()

            self.loop_timer_id = self.root.after(800, unpause)
            return

        if self.cap_expert and self.cap_expert.isOpened():
            ret_exp, frame_exp = self.cap_expert.read()
            if ret_exp and frame_exp is not None:
                self.draw_landmarks_on_frame(
                    frame_exp, self.expert_curr_f, self.expert_raw_data
                )
                self.expert_curr_f += 1
                self._display_frame_on_label(
                    frame_exp, self.expert_img_label, border_color=(35, 134, 54)
                )
        else:
            # Ở Chế độ Độc lập (không dùng video mẫu), hiển thị Mô hình Vòm Tay Chuẩn Công Thái Học AI
            cur_item = next(
                (
                    t
                    for t in self.timeline_data
                    if t.get("event_id") == self.selected_second
                ),
                {},
            )
            ang = float(cur_item.get("joint_angle", 142.0))
            ideal_diagram = self._render_ideal_posture_diagram(
                ang,
                cur_item.get("sheet_note"),
                cur_item.get("played_note"),
                cur_item.get("sheet_finger_num"),
            )
            self._display_frame_on_label(
                ideal_diagram, self.expert_img_label, border_color=(35, 134, 54)
            )

        if self.cap_user and self.cap_user.isOpened():
            ret_usr, frame_usr = self.cap_user.read()
            if ret_usr and frame_usr is not None:
                self.draw_landmarks_on_frame(
                    frame_usr, self.user_curr_f, self.user_raw_data
                )
                self.user_curr_f += 1
                self._display_frame_on_label(
                    frame_usr, self.user_img_label, border_color=(218, 54, 51)
                )

        self.loop_timer_id = self.root.after(33, self.update_video_loop)

    def _display_frame_on_label(
        self,
        frame: np.ndarray,
        label_widget: tk.Label,
        border_color: tuple = (100, 100, 100),
    ) -> None:
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            target_w = 200
            target_h = int(h * target_w / w)
            frame_resized = cv2.resize(frame_rgb, (target_w, target_h))

            border_px = 3
            bordered = np.full(
                (target_h + border_px * 2, target_w + border_px * 2, 3),
                border_color,
                dtype=np.uint8,
            )
            bordered[
                border_px : border_px + target_h, border_px : border_px + target_w
            ] = frame_resized

            pil_img = Image.fromarray(bordered)
            tk_img = ImageTk.PhotoImage(pil_img)
            label_widget.config(image=tk_img, text="")
            label_widget.image = tk_img
        except Exception:
            pass

    def start_evaluation(self) -> None:
        self.run_btn.config(state=tk.DISABLED, bg="#2d333b")
        self.reset_btn.config(state=tk.DISABLED, bg="#2d333b")
        self.choose_btn.config(state=tk.DISABLED, bg="#2d333b")
        self.status_label.config(
            text="Đang phân tích video & vẽ biểu đồ... vui lòng đợi.", fg="#58a6ff"
        )
        self.output_text.delete(1.0, tk.END)
        self.timeline_data = []
        self.clear_timeline()
        self.clear_review_panel()
        thread = threading.Thread(target=self.run_script, daemon=True)
        thread.start()

    def finish_evaluation(self) -> None:
        self.run_btn.config(state=tk.NORMAL, bg="#238636")
        self.reset_btn.config(state=tk.NORMAL, bg="#b08800")
        self.choose_btn.config(state=tk.NORMAL, bg="#1f6feb")
        self.status_label.config(
            text="✅  Hoàn thành chấm điểm! Click ô bất kỳ trên Timeline để xem Video & Biểu đồ Sóng.",
            fg="#3fb950",
        )


if __name__ == "__main__":
    root = tk.Tk()
    w, h = 1340, 860
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
    try:
        root.state("zoomed")
    except Exception:
        pass
    app = PianoAIGUI(root)
    root.mainloop()
