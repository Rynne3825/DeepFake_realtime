# =============================================================================
# File: ui/app_window.py
# Mục đích: Giao diện người dùng chính của hệ thống.
# =============================================================================

import os
import ctypes
import tkinter as tk
from tkinter import filedialog
from typing import Optional, Any

import cv2
import numpy as np
from PIL import Image, ImageTk

import config.globals as globals
from config.backend_status import describe_backend
from config.metadata import name as APP_NAME, version as APP_VERSION
from config.performance_presets import KEY_TO_LABEL, LABEL_TO_KEY, PRESETS, apply_preset, get_preset
from utils.fps_counter import FPSCounter
from stream.web_cam import WebcamCapture
from core.face_analyzer import get_one_face
from core.face_swapper import process_frame
from core.face_enhancer import enhance_frame


# =============================================================================
# BẢNG MÀU & FONT
# =============================================================================
C = {
    "bg":       "#0d0d14",
    "sidebar":  "#111119",
    "section":  "#18182a",
    "input":    "#22223a",
    "accent":   "#7c6cf7",
    "green":    "#2ecc71",
    "red":      "#e74c3c",
    "red_dim":  "#992d22",
    "yellow":   "#f39c12",
    "text":     "#eeeef4",
    "dim":      "#7a7a98",
    "border":   "#2a2a3e",
}

F = {
    "title":  ("Segoe UI", 15, "bold"),
    "sec":    ("Segoe UI Semibold", 9),
    "btn":    ("Segoe UI Semibold", 9),
    "label":  ("Segoe UI", 9),
    "fps":    ("Consolas", 13, "bold"),
    "status": ("Segoe UI", 9),
}

# Chiều rộng sidebar (px)
SIDEBAR_W = 240


class DeepFakeApp:
    """Giao diện chính của hệ thống hoán đổi khuôn mặt thời gian thực."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)
        self.root.configure(bg=C["bg"], highlightthickness=0, bd=0)
        self.root.option_add("*tearOff", False)
        self.root.option_add("*highlightThickness", 0)

        # Ép title bar + viền cửa sổ Windows sang màu tối/đen
        self._apply_dark_titlebar()

        # Trạng thái nội bộ
        self._webcam = WebcamCapture()
        self._fps_counter = FPSCounter()
        self._source_face: Optional[Any] = None
        self._source_image: Optional[np.ndarray] = None
        self._is_processing: bool = False
        self._update_job = None

        self._build_ui()
        self._sync_ui_with_globals()
        self._bind_hotkeys()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # =========================================================================
    # PHÍM TẮT
    # =========================================================================
    def _bind_hotkeys(self):
        """Gán phím tắt cho các chức năng quan trọng."""
        # Escape = Dừng khẩn cấp
        self.root.bind("<Escape>", lambda e: self._emergency_stop())
        # Ctrl+Q = Thoát app
        self.root.bind("<Control-q>", lambda e: self._kill_app())
        # Ctrl+W = Bật/tắt webcam
        self.root.bind("<Control-w>", lambda e: self._toggle_webcam())

    # =========================================================================
    # XÂY DỰNG GIAO DIỆN
    # =========================================================================
    def _build_ui(self):
        """Xây dựng toàn bộ layout."""

        # === HEADER (sát đỉnh, không padding) ===
        header = tk.Frame(self.root, bg=C["bg"], height=44)
        header.pack(fill="x", pady=0, padx=0, anchor="n")
        header.pack_propagate(False)

        hi = tk.Frame(header, bg=C["bg"])
        hi.pack(fill="both", expand=True, padx=14, pady=0)

        tk.Label(hi, text="🎭", font=("Segoe UI", 18),
                 bg=C["bg"], fg=C["accent"]).pack(side="left")
        tk.Label(hi, text=f" {APP_NAME}", font=F["title"],
                 bg=C["bg"], fg=C["text"]).pack(side="left")
        tk.Label(hi, text=f" v{APP_VERSION}", font=("Segoe UI", 8),
                 bg=C["bg"], fg=C["dim"]).pack(side="left", pady=(3, 0))

        # Nút Thoát App (Ctrl+Q)
        tk.Button(
            hi, text="✕ Thoát (Ctrl+Q)", command=self._kill_app,
            bg=C["red"], fg="#fff", font=F["btn"],
            relief="flat", cursor="hand2", bd=0, padx=10, pady=3,
            activebackground=C["red_dim"], activeforeground="#fff"
        ).pack(side="right")

        # Trạng thái
        self._status_label = tk.Label(
            hi, text="⏳ Sẵn sàng", font=F["status"],
            bg=C["bg"], fg=C["dim"]
        )
        self._status_label.pack(side="right", padx=12)

        # Đường kẻ accent (sát header)
        tk.Frame(self.root, bg=C["accent"], height=1).pack(fill="x", pady=0)

        # === NỘI DUNG CHÍNH (sát header) ===
        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill="both", expand=True, pady=0)

        # --- Sidebar (scrollable) ---
        sc = tk.Frame(body, bg=C["sidebar"], width=SIDEBAR_W)
        sc.pack(side="left", fill="y")
        sc.pack_propagate(False)

        self._canvas = tk.Canvas(sc, bg=C["sidebar"], highlightthickness=0,
                                 width=SIDEBAR_W - 8)
        sb = tk.Scrollbar(sc, orient="vertical", command=self._canvas.yview)
        self._inner = tk.Frame(self._canvas, bg=C["sidebar"])

        self._inner.bind("<Configure>",
                         lambda e: self._canvas.configure(
                             scrollregion=self._canvas.bbox("all")))
        self._canvas.create_window((0, 0), window=self._inner, anchor="nw",
                                   width=SIDEBAR_W - 8)
        self._canvas.configure(yscrollcommand=sb.set)

        sb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        # Cuộn chuột
        self._canvas.bind_all(
            "<MouseWheel>",
            lambda e: self._canvas.yview_scroll(int(-e.delta / 120), "units")
        )

        self._build_sidebar(self._inner)

        # Viền ngăn cách
        tk.Frame(body, bg=C["border"], width=1).pack(side="left", fill="y")

        # --- Video ---
        vp = tk.Frame(body, bg=C["bg"])
        vp.pack(side="right", fill="both", expand=True)
        self._build_video(vp)

    # =========================================================================
    # SIDEBAR
    # =========================================================================
    def _build_sidebar(self, p):
        """Xây dựng sidebar."""
        pad = 10

        # ── ẢNH NGUỒN ──
        self._sec_title(p, "📷 ẢNH NGUỒN")
        s1 = self._sec(p)

        # Preview - dùng pixel cố định, KHÔNG dùng width/height ký tự
        preview_frame = tk.Frame(s1, bg=C["input"], height=130)
        preview_frame.pack(fill="x", padx=pad, pady=(pad, 4))
        preview_frame.pack_propagate(False)

        self._source_preview = tk.Label(
            preview_frame, text="Chưa chọn ảnh",
            bg=C["input"], fg=C["dim"], font=F["label"]
        )
        self._source_preview.pack(expand=True)
        self._btn(s1, "📂 Chọn ảnh nguồn", self._select_source, C["accent"])

        # ── WEBCAM ──
        self._sec_title(p, "🎥 WEBCAM")
        s2 = self._sec(p)

        tk.Label(s2, text="Preset hiệu năng:", bg=C["section"],
                 fg=C["text"], font=F["label"]).pack(anchor="w", padx=pad, pady=(pad, 0))

        self._var_preset = tk.StringVar(value=KEY_TO_LABEL[globals.quality_preset])
        preset_labels = [preset.label for preset in PRESETS.values()]
        pf = tk.Frame(s2, bg=C["section"])
        pf.pack(fill="x", padx=pad, pady=(2, 4))
        self._preset_menu = tk.OptionMenu(
            pf, self._var_preset, *preset_labels, command=self._on_preset_change
        )
        self._preset_menu.config(
            bg=C["input"], fg=C["text"], activebackground=C["accent"],
            activeforeground="#fff", highlightthickness=0, bd=0,
            font=F["label"], relief="flat"
        )
        self._preset_menu["menu"].config(
            bg=C["input"], fg=C["text"], activebackground=C["accent"],
            activeforeground="#fff", font=F["label"], bd=0
        )
        self._preset_menu.pack(fill="x")
        self._preset_hint = tk.Label(
            s2, text="", justify="left", wraplength=190,
            bg=C["section"], fg=C["dim"], font=("Segoe UI", 8)
        )
        self._preset_hint.pack(fill="x", padx=pad, pady=(0, 6))

        self._backend_label = tk.Label(
            s2, text="", justify="left",
            bg=C["section"], fg=C["dim"], font=F["label"], anchor="w"
        )
        self._backend_label.pack(fill="x", padx=pad, pady=(0, 6))

        self._btn_webcam = self._btn(s2, "▶ Webcam (Ctrl+W)",
                                     self._toggle_webcam, C["green"])
        self._btn_emergency = self._btn(s2, "⚠ DỪNG KHẨN CẤP (Esc)",
                                        self._emergency_stop, C["red"])

        # FPS
        fr = tk.Frame(s2, bg=C["section"])
        fr.pack(fill="x", padx=pad, pady=(2, pad))
        tk.Label(fr, text="⚡ FPS:", font=F["label"],
                 bg=C["section"], fg=C["dim"]).pack(side="left")
        self._fps_label = tk.Label(fr, text="0.0", font=F["fps"],
                                   bg=C["section"], fg=C["green"])
        self._fps_label.pack(side="right")

        # ── TÙY CHỌN AI ──
        self._sec_title(p, "🤖 TÙY CHỌN AI")
        s3 = self._sec(p)

        self._var_enhancer = tk.BooleanVar(value=False)
        self._chk(s3, "✨ Làm nét khuôn mặt", self._var_enhancer,
                  self._on_enhancer_toggle)

        self._var_many = tk.BooleanVar(value=False)
        self._chk(s3, "👥 Swap nhiều mặt", self._var_many,
                  self._on_many_toggle)

        self._var_masking = tk.BooleanVar(value=False)
        self._chk(s3, "🛡 Chống lẹm bằng face mask", self._var_masking,
                  self._on_masking_toggle)

        self._var_mouth_mask = tk.BooleanVar(value=False)
        self._chk(s3, "👄 Giữ miệng gốc", self._var_mouth_mask,
                  self._on_mouth_mask_toggle)

        tk.Label(s3, text="Kích thước vùng miệng gốc:", bg=C["section"],
                 fg=C["text"], font=F["label"]).pack(anchor="w", padx=pad, pady=(6, 0))
        mf = tk.Frame(s3, bg=C["section"])
        mf.pack(fill="x", padx=pad, pady=(0, 6))
        self._mouth_mask_val = tk.Label(
            mf, text="0", font=F["btn"], bg=C["section"], fg=C["accent"], width=4
        )
        self._mouth_mask_val.pack(side="right")
        self._mouth_mask_slider = tk.Scale(
            mf, from_=0, to=100, resolution=5, orient="horizontal",
            bg=C["section"], fg=C["text"], troughcolor=C["input"],
            highlightthickness=0, showvalue=False,
            command=self._on_mouth_mask_size
        )
        self._mouth_mask_slider.set(0)
        self._mouth_mask_slider.pack(side="left", fill="x", expand=True)

        self._var_mirror = tk.BooleanVar(value=True)
        globals.live_mirror = True
        self._chk(s3, "🪞 Lật gương (Mirror)", self._var_mirror,
                  self._on_mirror_toggle)

        # ── TINH CHỈNH LÀM NÉT ──
        self._sec_title(p, "🔧 TINH CHỈNH LÀM NÉT")
        s4 = self._sec(p)

        # Slider cường độ
        tk.Label(s4, text="Mức độ nét:", bg=C["section"],
                 fg=C["text"], font=F["label"]).pack(anchor="w", padx=pad, pady=(pad, 0))

        sf = tk.Frame(s4, bg=C["section"])
        sf.pack(fill="x", padx=pad, pady=(0, 4))
        self._str_val = tk.Label(sf, text="0.60", font=F["btn"],
                                 bg=C["section"], fg=C["accent"], width=4)
        self._str_val.pack(side="right")
        self._slider = tk.Scale(
            sf, from_=0.1, to=1.0, resolution=0.05, orient="horizontal",
            bg=C["section"], fg=C["text"], troughcolor=C["input"],
            highlightthickness=0, showvalue=False,
            command=self._on_strength
        )
        self._slider.set(0.6)
        self._slider.pack(side="left", fill="x", expand=True)

        # Dropdown model
        tk.Label(s4, text="Mô hình:", bg=C["section"],
                 fg=C["text"], font=F["label"]).pack(anchor="w", padx=pad, pady=(4, 0))

        self._var_model = tk.StringVar(value="gfpgan-1024.onnx")
        opts = ["gfpgan-1024.onnx", "GPEN-BFR-512.onnx", "GPEN-BFR-256.onnx"]

        mf = tk.Frame(s4, bg=C["section"])
        mf.pack(fill="x", padx=pad, pady=(2, pad))
        dd = tk.OptionMenu(mf, self._var_model, *opts, command=self._on_model)
        dd.config(bg=C["input"], fg=C["text"], activebackground=C["accent"],
                  activeforeground="#fff", highlightthickness=0, bd=0,
                  font=F["label"], relief="flat")
        dd["menu"].config(bg=C["input"], fg=C["text"],
                          activebackground=C["accent"],
                          activeforeground="#fff", font=F["label"], bd=0)
        dd.pack(fill="x")

        # Spacer cuối
        tk.Frame(p, bg=C["sidebar"], height=15).pack()

    # =========================================================================
    # VIDEO
    # =========================================================================
    def _build_video(self, parent):
        """Vùng hiển thị video."""
        vf = tk.Frame(parent, bg=C["border"], padx=1, pady=1)
        vf.pack(fill="both", expand=True, padx=10, pady=10)

        vi = tk.Frame(vf, bg=C["bg"])
        vi.pack(fill="both", expand=True)

        self._video_label = tk.Label(
            vi,
            text="🎥\n\nChọn ảnh nguồn  →  Bật Webcam  →  Bắt đầu!",
            bg=C["bg"], fg=C["dim"],
            font=("Segoe UI", 12), justify="center"
        )
        self._video_label.pack(fill="both", expand=True)

    # =========================================================================
    # UI HELPERS
    # =========================================================================
    def _sec_title(self, parent, text):
        tk.Label(parent, text=text, font=F["sec"],
                 bg=C["sidebar"], fg=C["dim"], anchor="w"
                 ).pack(fill="x", padx=10, pady=(12, 3))

    def _sec(self, parent):
        f = tk.Frame(parent, bg=C["section"],
                     highlightbackground=C["border"], highlightthickness=1)
        f.pack(fill="x", padx=8, pady=(0, 2))
        return f

    def _btn(self, parent, text, cmd, color):
        b = tk.Button(
            parent, text=text, command=cmd,
            bg=color, fg="#fff", font=F["btn"],
            relief="flat", cursor="hand2", bd=0, padx=8, pady=5,
            activebackground=color, activeforeground="#fff"
        )
        b.pack(fill="x", padx=10, pady=4)
        return b

    def _chk(self, parent, text, var, cmd):
        cb = tk.Checkbutton(
            parent, text=text, variable=var, command=cmd,
            bg=C["section"], fg=C["text"],
            selectcolor=C["input"],
            activebackground=C["section"],
            activeforeground=C["text"],
            font=F["label"], anchor="w", highlightthickness=0
        )
        cb.pack(fill="x", padx=10, pady=2)
        return cb

    # =========================================================================
    # SỰ KIỆN
    # =========================================================================
    def _select_source(self):
        """Chọn ảnh nguồn."""
        path = filedialog.askopenfilename(
            title="Chọn ảnh khuôn mặt nguồn",
            filetypes=[("Ảnh", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not path:
            return

        globals.source_path = path
        img = cv2.imread(path)
        if img is None:
            self._update_status("❌ Không đọc được ảnh!", C["red"])
            return

        self._source_face = get_one_face(img)
        if self._source_face is None:
            self._update_status("⚠️ Không tìm thấy khuôn mặt!", C["yellow"])
            return

        self._source_image = img
        self._update_status("✅ Đã tải ảnh nguồn!", C["green"])

        # Preview giữ đúng tỉ lệ gốc
        h, w = img.shape[:2]
        # Lấy kích thước thực tế của khung preview
        self._source_preview.update_idletasks()
        box_w = self._source_preview.master.winfo_width() - 4
        box_h = self._source_preview.master.winfo_height() - 4
        if box_w < 50:
            box_w = 200
        if box_h < 50:
            box_h = 120
        scale = min(box_w / w, box_h / h)
        nw, nh = int(w * scale), int(h * scale)
        preview = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(preview))
        self._source_preview.configure(image=photo, text="")
        self._source_preview._photo = photo

    def _toggle_webcam(self):
        if globals.webcam_active:
            self._stop_webcam()
        else:
            self._start_webcam()

    def _start_webcam(self):
        if self._source_face is None:
            self._update_status("⚠️ Hãy chọn ảnh nguồn trước!", C["yellow"])
            return
        success = self._webcam.start(
            width=globals.webcam_width,
            height=globals.webcam_height,
            fps=globals.webcam_fps,
        )
        if not success:
            self._update_status("❌ Không thể mở Webcam!", C["red"])
            return
        globals.webcam_active = True
        self._is_processing = True
        self._btn_webcam.configure(text="⏹ Dừng Webcam", bg=C["red"])
        self._update_status("🟢 Webcam đang hoạt động", C["green"])
        self._process_loop()

    def _stop_webcam(self):
        self._is_processing = False
        globals.webcam_active = False
        self._webcam.release()
        self._btn_webcam.configure(text="▶ Webcam (Ctrl+W)", bg=C["green"])
        self._update_status("⏸ Webcam đã tắt", C["dim"])
        if self._update_job:
            self.root.after_cancel(self._update_job)
            self._update_job = None

    def _emergency_stop(self):
        """Dừng khẩn cấp webcam (phím tắt: Escape)."""
        self._is_processing = False
        globals.webcam_active = False
        try:
            self._webcam.release()
        except Exception:
            pass
        if self._update_job:
            self.root.after_cancel(self._update_job)
            self._update_job = None
        self._btn_webcam.configure(text="▶ Webcam (Ctrl+W)", bg=C["green"])
        self._video_label.configure(
            image="",
            text="⚠️  Webcam đã dừng khẩn cấp\n\nBấm lại Webcam để tiếp tục."
        )
        self._video_label._photo = None
        self._fps_label.configure(text="0.0", fg=C["red"])
        self._update_status("🔴 ĐÃ DỪNG KHẨN CẤP", C["red"])

    def _kill_app(self):
        """Thoát app ngay lập tức (phím tắt: Ctrl+Q)."""
        try:
            self._is_processing = False
            globals.webcam_active = False
            self._webcam.release()
        except Exception:
            pass
        if self._update_job:
            try:
                self.root.after_cancel(self._update_job)
            except Exception:
                pass
        try:
            self.root.destroy()
        except Exception:
            pass
        os._exit(0)

    def _process_loop(self):
        """
        Vòng lặp xử lý chính: Đọc frame -> AI xử lý -> Hiển thị.

        GIẢI THÍCH CHO BÁO CÁO:
            Hàm này được gọi lặp lại liên tục bằng root.after() của Tkinter.
            Mỗi lần gọi, nó thực hiện:
            1. Đọc 1 khung hình từ Webcam
            2. (Tùy chọn) Lật gương nếu bật
            3. Hoán đổi khuôn mặt bằng AI (Face Swap)
            4. (Tùy chọn) Làm nét bằng GFPGAN
            5. Hiển thị kết quả lên giao diện
            6. Cập nhật FPS
        """
        if not self._is_processing:
            return

        ret, frame = self._webcam.read()
        if ret and frame is not None:
            if globals.live_mirror:
                frame = cv2.flip(frame, 1)
            if globals.enable_swapper and self._source_face is not None:
                frame = process_frame(self._source_face, frame)
            if globals.enable_enhancer:
                frame = enhance_frame(frame)
            self._display_frame(frame)
            self._fps_counter.tick()
            fps = self._fps_counter.get_fps()
            color = C["green"] if fps >= 20 else (C["yellow"] if fps >= 10 else C["red"])
            self._fps_label.configure(text=f"{fps:.1f}", fg=color)

        self._update_job = self.root.after(1, self._process_loop)

    def _display_frame(self, frame: np.ndarray):
        """Hiển thị frame lên giao diện."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lw = self._video_label.winfo_width()
        lh = self._video_label.winfo_height()
        if lw > 1 and lh > 1:
            h, w = rgb.shape[:2]
            scale = min(lw / w, lh / h)
            nw, nh = int(w * scale), int(h * scale)
            if nw > 0 and nh > 0:
                rgb = cv2.resize(rgb, (nw, nh))
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._video_label.configure(image=photo, text="")
        self._video_label._photo = photo

    # =========================================================================
    # CALLBACKS
    # =========================================================================
    def _on_enhancer_toggle(self):
        globals.enable_enhancer = self._var_enhancer.get()
        st = "BẬT ✨" if globals.enable_enhancer else "TẮT"
        self._update_status(f"Làm nét: {st}", C["accent"])

    def _on_strength(self, val):
        v = float(val)
        globals.enhancement_strength = v
        self._str_val.configure(text=f"{v:.2f}")

    def _on_model(self, val):
        globals.enhancer_model = val
        self._update_status(f"Model → {val}", C["accent"])

    def _on_preset_change(self, label):
        key = LABEL_TO_KEY[label]
        was_running = globals.webcam_active
        preset = apply_preset(globals, key)
        self._sync_ui_with_globals()

        if was_running:
            self._stop_webcam()
            self._start_webcam()

        self._update_status(
            f"Preset {preset.label}: {preset.webcam_width}x{preset.webcam_height} @ {preset.webcam_fps} FPS",
            C["accent"],
        )

    def _on_many_toggle(self):
        globals.many_faces = self._var_many.get()

    def _on_masking_toggle(self):
        globals.enable_masking = self._var_masking.get()
        st = "BẬT 🛡" if globals.enable_masking else "TẮT"
        self._update_status(f"Face mask: {st}", C["accent"])

    def _on_mouth_mask_toggle(self):
        globals.mouth_mask = self._var_mouth_mask.get()
        st = "BẬT 👄" if globals.mouth_mask else "TẮT"
        self._update_status(f"Giữ miệng gốc: {st}", C["accent"])

    def _on_mouth_mask_size(self, val):
        size = float(val)
        globals.mouth_mask_size = size
        self._mouth_mask_val.configure(text=f"{int(size)}")

    def _on_mirror_toggle(self):
        globals.live_mirror = self._var_mirror.get()

    # =========================================================================
    # TIỆN ÍCH
    # =========================================================================
    def _apply_dark_titlebar(self):
        """Ép title bar + viền cửa sổ Windows sang màu tối/đen.
        Sử dụng Windows DWM API (chỉ hoạt động trên Win 10 build 18985+ / Win 11).
        """
        try:
            self.root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            # DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            value = ctypes.c_int(1)  # 1 = dark mode
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
                ctypes.byref(value), ctypes.sizeof(value)
            )
            # DWMWA_BORDER_COLOR = 34 — đặt viền thành màu đen
            DWMWA_BORDER_COLOR = 34
            # COLORREF: 0x00BBGGRR — đen = 0x000D0D14 (match nền app)
            black_border = ctypes.c_int(0x00140D0D)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_BORDER_COLOR,
                ctypes.byref(black_border), ctypes.sizeof(black_border)
            )
            # DWMWA_CAPTION_COLOR = 35 — đặt màu nền title bar
            DWMWA_CAPTION_COLOR = 35
            caption_color = ctypes.c_int(0x00140D0D)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_CAPTION_COLOR,
                ctypes.byref(caption_color), ctypes.sizeof(caption_color)
            )
        except Exception:
            pass  # Bỏ qua nếu không hỗ trợ (Linux, macOS, Win cũ)

    def _update_status(self, msg, color=None):
        self._status_label.configure(text=msg, fg=color or C["dim"])

    def _sync_ui_with_globals(self):
        preset = get_preset(globals.quality_preset)
        backend = describe_backend(globals.execution_providers)
        self._var_preset.set(KEY_TO_LABEL[preset.key])
        self._preset_hint.configure(
            text=(
                f"{preset.description}\n"
                f"{preset.webcam_width}x{preset.webcam_height} @ {preset.webcam_fps} FPS"
            )
        )
        self._backend_label.configure(text=backend.label, fg=C[backend.color])
        self._var_enhancer.set(globals.enable_enhancer)
        self._var_many.set(globals.many_faces)
        self._var_masking.set(globals.enable_masking)
        self._var_mouth_mask.set(globals.mouth_mask)
        self._mouth_mask_slider.set(globals.mouth_mask_size)
        self._mouth_mask_val.configure(text=f"{int(globals.mouth_mask_size)}")
        self._var_mirror.set(globals.live_mirror)
        self._var_model.set(globals.enhancer_model)
        self._slider.set(globals.enhancement_strength)
        self._str_val.configure(text=f"{globals.enhancement_strength:.2f}")

    def _on_close(self):
        self._stop_webcam()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
