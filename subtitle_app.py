import sys
import os
import json
import subprocess
import imageio_ffmpeg

# ===== pythonw.exe 无控制台修复 =====
# whisper 内部会写 sys.stdout/stderr，pythonw.exe 下它们为 None 导致崩溃
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

# ===== 便携版 Qt6 DLL 路径 ====
# PyInstaller / 便携版需要手动指定 Qt6 DLL 目录
_qt6_bin = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "PyQt6", "Qt6", "bin")
if os.path.isdir(_qt6_bin):
    os.add_dll_directory(_qt6_bin)
    os.environ["PATH"] = _qt6_bin + os.pathsep + os.environ.get("PATH", "")

# ===== 国内镜像设置 =====
# HuggingFace 镜像（新版 whisper 内部优先用 HF hub 下载）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 如果上面不生效，可尝试备用镜像:
# os.environ["HF_ENDPOINT"] = "https://hf.xeduapi.com"

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QProgressBar,
    QComboBox, QMessageBox, QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

# ---- 按钮样式 ----
BTN_STYLE = """
    QPushButton {
        border: 1px solid #999;
        border-radius: 5px;
        padding: 6px 16px;
    }
    QPushButton:hover {
        background-color: #e8e8e8;
    }
    QPushButton:pressed {
        background-color: #d0d0d0;
    }
"""
BTN_STYLE_DISABLED = BTN_STYLE + """
    QPushButton:disabled {
        border-color: #aaa;
        color: #888;
    }
"""


def _model_dir():
    """模型存放目录（可写）：EXE 模式下放 exe 旁边，否则放项目目录"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _find_model(model_name):
    """查找 .pt 模型文件，返回路径或 None
    优先查可写目录，其次查 EXE 内置目录 (sys._MEIPASS)
    """
    filename = f"{model_name}.pt"

    # 1) 可写目录（用户下载的、手动放的）
    writable = os.path.join(_model_dir(), filename)
    if os.path.exists(writable):
        return writable

    # 2) EXE 内置目录（打包时 bundled 的只读模型）
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        bundled = os.path.join(meipass, filename)
        if os.path.exists(bundled):
            return bundled

    return None


class TranscribeThread(QThread):
    """转录线程"""
    progress = pyqtSignal(str, int)  # (消息, 百分比)
    finished = pyqtSignal(bool, str)

    def __init__(self, input_file, output_dir, model_size, language):
        super().__init__()
        self.input_file = input_file
        self.output_dir = output_dir
        self.model_size = model_size
        self.language = language

    def _load_whisper_model(self, whisper):
        """加载 Whisper 模型：本地 .pt > 国内镜像下载 > whisper 默认"""

        # 1) 本地 .pt 文件 — 直接加载
        local_pt = _find_model(self.model_size)
        if local_pt:
            self.progress.emit(f"从本地加载模型: {local_pt}", 35)
            return whisper.load_model(local_pt)

        # 2) 从国内镜像下载 .pt 到本地
        self.progress.emit(f"正在下载模型 {self.model_size}（国内镜像）...", 35)
        model_path = self._download_from_mirror()
        if model_path and os.path.exists(model_path):
            return whisper.load_model(model_path)

        # 3) 兜底：whisper 默认下载（HF_ENDPOINT 已指向国内镜像）
        self.progress.emit(f"正在下载模型 {self.model_size}（HF 镜像）...", 35)
        return whisper.load_model(self.model_size)

    def _download_from_mirror(self):
        """从国内镜像下载 Whisper .pt 模型文件"""
        import urllib.request

        dest = os.path.join(_model_dir(), f"{self.model_size}.pt")

        # GitHub Release 文件名（v20231117 是最后一个 release）
        pt_files = {
            "tiny":   "tiny.pt",
            "base":   "base.pt",
            "small":  "small.pt",
            "medium": "medium.pt",
            "large":  "large-v3.pt",
        }
        filename = pt_files.get(self.model_size)
        if not filename:
            return None

        # GitHub Release CDN 代理（国内可访问）
        mirrors = [
            f"https://mirror.ghproxy.com/https://github.com/openai/whisper/releases/download/v20231117/{filename}",
            f"https://ghproxy.net/https://github.com/openai/whisper/releases/download/v20231117/{filename}",
            f"https://gh-proxy.com/https://github.com/openai/whisper/releases/download/v20231117/{filename}",
        ]

        for url in mirrors:
            try:
                self.progress.emit(f"下载: {url[:60]}...", 35)
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    total = int(resp.headers.get("Content-Length", 0))
                    data = bytearray()
                    downloaded = 0
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        data.extend(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = 35 + int(downloaded / total * 10)
                            self.progress.emit(
                                f"下载 {filename}: {downloaded//1024//1024}MB / {total//1024//1024}MB", pct
                            )
                with open(dest, "wb") as f:
                    f.write(data)
                self.progress.emit(f"模型下载完成: {dest}", 45)
                return dest
            except Exception as e:
                self.progress.emit(f"镜像失败: {e}", 35)
                continue

        return None

    def run(self):
        try:
            self.progress.emit("正在提取音频...", 10)

            # 提取音频
            audio_path = os.path.join(self.output_dir, "temp_audio.wav")
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe, "-i", self.input_file,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                "-y", audio_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            self.progress.emit("正在加载模型...", 30)
            import whisper
            import numpy as np
            from scipy.io import wavfile

            # --- 加载模型（含国内镜像逻辑）---
            model = self._load_whisper_model(whisper)

            self.progress.emit("正在转录...", 50)
            sample_rate, audio_data = wavfile.read(audio_path)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            audio = audio_data.astype(np.float32) / 32768.0

            # language为None时自动检测语言
            if self.language:
                result = model.transcribe(audio, language=self.language, verbose=False)
            else:
                result = model.transcribe(audio, verbose=False)

            self.progress.emit("正在生成字幕...", 90)
            # 生成 SRT
            srt_lines = []
            for i, seg in enumerate(result["segments"], 1):
                start = seg["start"]
                end = seg["end"]
                text = seg["text"].strip()

                sh = int(start // 3600)
                sm = int((start % 3600) // 60)
                ss = start % 60
                eh = int(end // 3600)
                em = int((end % 3600) // 60)
                es = end % 60

                srt_lines.append(str(i))
                srt_lines.append(
                    "{:02d}:{:02d}:{:06.3f}".format(sh, sm, ss).replace(".", ",")
                    + " --> "
                    + "{:02d}:{:02d}:{:06.3f}".format(eh, em, es).replace(".", ",")
                )
                srt_lines.append(text)
                srt_lines.append("")

            # 保存文件
            base_name = os.path.splitext(os.path.basename(self.input_file))[0]
            srt_path = os.path.join(self.output_dir, f"{base_name}.srt")
            json_path = os.path.join(self.output_dir, f"{base_name}.json")

            with open(srt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(srt_lines))

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result["segments"], f, ensure_ascii=False, indent=2)

            # 清理临时文件
            if os.path.exists(audio_path):
                os.remove(audio_path)

            self.progress.emit(f"转录完成！共 {len(result['segments'])} 个片段", 100)
            self.finished.emit(True, srt_path)

        except Exception as e:
            self.finished.emit(False, str(e))

        finally:
            # 释放显存
            try:
                import torch
                if 'model' in locals():
                    del model
                if 'result' in locals():
                    del result
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


class SubtitleApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频字幕提取工具")
        self.setMinimumSize(700, 500)
        self.init_ui()

        # 进度条定时器
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.current_progress = 0

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 输入文件组
        input_group = QGroupBox("输入设置")
        input_layout = QVBoxLayout()

        # 文件选择
        file_layout = QHBoxLayout()
        self.file_label = QLabel("未选择文件")
        self.file_label.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 3px;")
        self.select_btn = QPushButton("选择文件")
        self.select_btn.clicked.connect(self.select_file)
        self.select_btn.setStyleSheet(BTN_STYLE)
        file_layout.addWidget(self.file_label, 1)
        file_layout.addWidget(self.select_btn)
        input_layout.addLayout(file_layout)

        # 模型选择（自动检测本地/云端）
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("模型大小:"))
        self.model_combo = QComboBox()

        # 扫描本地 .pt 文件（可写目录 + EXE 内置）
        local_models = set()
        for name in ["tiny", "base", "small", "medium", "large"]:
            if _find_model(name):
                local_models.add(name)

        # 构建下拉项：本地模型标注"本地"，其余标注"云端"
        self.model_items = []  # 保存 (显示文本, model_size)
        for name in ["tiny", "base", "small", "medium", "large"]:
            if name in local_models:
                label = f"{name}（本地）"
            else:
                label = f"{name}（云端）"
            self.model_items.append((label, name))
            self.model_combo.addItem(label)

        # 默认选中本地已有的最大模型，否则选 medium
        for m in ["large", "medium", "base", "small", "tiny"]:
            if m in local_models:
                self.model_combo.setCurrentText(f"{m}（本地）")
                break
        else:
            self.model_combo.setCurrentText("medium（云端）")
        model_layout.addWidget(self.model_combo)

        # GPU / CPU 状态
        try:
            import torch
            device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
            device_color = "#0070CB" if torch.cuda.is_available() else "#888"
        except Exception:
            device = "CPU"
            device_color = "#888"
        self.device_label = QLabel(device)
        self.device_label.setStyleSheet(
            f"color: {device_color}; font-weight: bold; padding: 2px 6px;"
        )
        model_layout.addWidget(self.device_label)

        model_layout.addWidget(QLabel("字幕语言:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems([
            "自动检测",
            "中文 (zh)", "英文 (en)", "日文 (ja)", "韩文 (ko)",
            "法文 (fr)", "德文 (de)", "西班牙文 (es)", "俄文 (ru)",
            "阿拉伯文 (ar)", "葡萄牙文 (pt)", "意大利文 (it)",
            "荷兰文 (nl)", "波兰文 (pl)", "土耳其文 (tr)",
            "瑞典文 (sv)", "丹麦文 (da)", "芬兰文 (fi)",
            "挪威文 (no)", "匈牙利文 (hu)", "捷克文 (cs)",
            "希腊文 (el)", "希伯来文 (he)", "泰文 (th)",
            "越南文 (vi)", "印尼文 (id)", "马来文 (ms)"
        ])
        self.lang_combo.setCurrentText("自动检测")
        model_layout.addWidget(self.lang_combo)
        model_layout.addStretch()
        input_layout.addLayout(model_layout)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # 输出区域
        output_group = QGroupBox("转录结果")
        output_layout = QVBoxLayout()
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Consolas", 10))
        output_layout.addWidget(self.output_text)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # 状态行（状态标签 + 进度条）
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(2, 0, 1, 0)
        self.status_label = QLabel("就绪")
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #5F5F5F;
                border-radius: 3px;
                text-align: center;
                height: 18px;
            }
            QProgressBar::chunk {
                background-color: #0070CB;
                border-radius: 2px;
            }
        """)
        self.progress_bar.hide()

        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar, 1)
        layout.addLayout(status_layout)

        # 按钮组
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始转录")
        self.start_btn.clicked.connect(self.start_transcribe)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet(BTN_STYLE_DISABLED)

        self.reselect_btn = QPushButton("重新选择")
        self.reselect_btn.clicked.connect(self.reselect)
        self.reselect_btn.setStyleSheet(BTN_STYLE)

        self.open_folder_btn = QPushButton("打开文件夹")
        self.open_folder_btn.clicked.connect(self.open_output_folder)
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.setStyleSheet(BTN_STYLE_DISABLED)

        self.exit_btn = QPushButton("退出")
        self.exit_btn.clicked.connect(self.close)
        self.exit_btn.setStyleSheet(BTN_STYLE)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.reselect_btn)
        btn_layout.addWidget(self.open_folder_btn)
        btn_layout.addWidget(self.exit_btn)

        btn_group = QGroupBox("操作")
        btn_group.setLayout(btn_layout)
        layout.addWidget(btn_group)

        self.input_file = None
        self.output_dir = None
        self.thread = None

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频或音频文件", "",
            "媒体文件 (*.mp4 *.mkv *.avi *.mov *.wav *.mp3 *.flac *.m4a);;所有文件 (*)"
        )
        if file_path:
            self.input_file = file_path
            self.file_label.setText(file_path)
            self.start_btn.setEnabled(True)
            self.status_label.setText("文件已选择，可以开始转录")

    def open_output_folder(self):
        if self.output_dir and os.path.isdir(self.output_dir):
            os.startfile(self.output_dir)

    def closeEvent(self, event):
        if hasattr(self, "thread") and self.thread is not None and self.thread.isRunning():
            self.thread.terminate()
            self.thread.wait(3000)
        event.accept()

    def reselect(self):
        self.input_file = None
        self.output_dir = None
        self.file_label.setText("未选择文件")
        self.start_btn.setEnabled(False)
        self.open_folder_btn.setEnabled(False)
        self.output_text.clear()
        self.status_label.setText("就绪")
        self.progress_bar.hide()

    def start_transcribe(self):
        if not self.input_file:
            return

        # 获取输出目录（与输入文件同目录）
        self.output_dir = os.path.dirname(self.input_file)
        # 从下拉框提取模型名（去除 "（本地）" / "（云端下载）" 后缀）
        model_size = self.model_combo.currentText().split("（")[0]
        # 从下拉框提取语言代码，如 "中文 (zh)" -> "zh"，"自动检测" -> None
        lang_text = self.lang_combo.currentText()
        if lang_text == "自动检测":
            language = None
        else:
            language = lang_text.split("(")[1].rstrip(")")

        # 禁用按钮
        self.start_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.reselect_btn.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.output_text.clear()

        # 启动转录线程
        self.thread = TranscribeThread(self.input_file, self.output_dir, model_size, language)
        self.thread.progress.connect(self.on_progress)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

    def on_progress(self, message, percent):
        self.status_label.setText(message)
        self.output_text.append(message)
        self.progress_bar.setValue(percent)
        self.current_progress = percent

        # 转录阶段(50%)启动定时器，缓慢推进到90%
        if percent == 50:
            self.progress_timer.start(500)  # 每500ms更新一次
        elif percent >= 90:
            self.progress_timer.stop()

    def update_progress(self):
        # 从50%缓慢推进到89%
        if self.current_progress < 89:
            self.current_progress += 1
            self.progress_bar.setValue(self.current_progress)
        else:
            self.progress_timer.stop()

    def on_finished(self, success, message):
        self.progress_timer.stop()
        self.progress_bar.setValue(100)
        self.progress_bar.hide()
        self.start_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.reselect_btn.setEnabled(True)

        if success:
            self.open_folder_btn.setEnabled(True)
            self.status_label.setText(f"完成！字幕已保存到: {message}")
            self.output_text.append(f"\n字幕文件已保存到: {message}")

            # 读取并显示 SRT 内容
            try:
                with open(message, "r", encoding="utf-8") as f:
                    content = f.read()
                self.output_text.append("\n=== 字幕内容 ===\n")
                self.output_text.append(content)
            except:
                pass

            QMessageBox.information(self, "完成", f"字幕已生成！\n{message}")
        else:
            self.status_label.setText(f"错误: {message}")
            self.output_text.append(f"\n错误: {message}")
            QMessageBox.critical(self, "错误", f"转录失败:\n{message}")


def main():
    # 高DPI适配
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    window = SubtitleApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
