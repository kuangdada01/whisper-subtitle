import sys
import os
import imageio_ffmpeg

from subtitle_utils import save_subtitle_files, funasr_result_to_segments

# ===== pythonw.exe 无控制台修复 =====
# 无控制台模式下 sys.stdout/stderr 为 None，库内部写它们会导致崩溃
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

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QProgressBar,
    QComboBox, QMessageBox, QGroupBox, QSizePolicy
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
    QPushButton:focus {
        border: 1px solid black;
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


class TranscriptionCancelled(Exception):
    """Raised when the user asks the worker thread to stop."""


def _run_translate_worker(input_path, target_lang, progress_cb, is_cancelled,
                          message_cb=None):
    """在独立进程中运行翻译工作器（隔离 bnb 与 funasr 交替加载的崩溃），返回译文列表

    独立进程每次重新加载翻译模型，与转录进程的 CUDA 状态完全隔离。
    进度通过工作器 stdout 的 __PROGRESS__ i n 行上报；取消时抛出 TranscriptionCancelled。
    bnb 在 RTX 50 系显卡上偶发加载崩溃（访问冲突），此处自动重启工作器重试。
    """
    import json
    import queue
    import subprocess
    import tempfile
    import threading

    script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "translate_worker.py")
    max_attempts = 3
    last_error = None

    for attempt in range(1, max_attempts + 1):
        fd, out_path = tempfile.mkstemp(suffix=".json", prefix="tr_out_")
        os.close(fd)
        proc = subprocess.Popen(
            [sys.executable, script, input_path, target_lang, out_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        q = queue.Queue()
        lines_tail = []

        def reader():
            try:
                for line in proc.stdout:
                    q.put(line)
            except Exception:
                pass

        threading.Thread(target=reader, daemon=True).start()

        def handle(line):
            nonlocal lines_tail
            line = line.strip()
            if line:
                lines_tail.append(line)
                lines_tail = lines_tail[-20:]
            if line.startswith("__PROGRESS__ "):
                try:
                    _, i, n = line.split()
                    progress_cb(int(i), int(n))
                except ValueError:
                    pass

        try:
            while proc.poll() is None:
                if is_cancelled():
                    proc.kill()
                    raise TranscriptionCancelled()
                try:
                    handle(q.get(timeout=0.5))
                except queue.Empty:
                    continue
            while True:
                try:
                    handle(q.get(timeout=0.5))
                except queue.Empty:
                    break
            if proc.returncode != 0:
                detail = "\n".join(lines_tail[-10:])
                last_error = RuntimeError(
                    f"翻译进程异常退出（代码 {proc.returncode}）\n{detail}")
            elif not os.path.exists(out_path):
                last_error = RuntimeError("翻译进程未生成结果文件")
            else:
                with open(out_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        finally:
            if proc.poll() is None:
                proc.kill()
            try:
                os.remove(out_path)
            except OSError:
                pass

        if attempt < max_attempts:
            if message_cb:
                message_cb(
                    f"翻译进程异常，正在自动重试（{attempt}/{max_attempts - 1}）…",
                    93)
        else:
            raise last_error


def _cuda_available():
    """后台检查 CUDA 是否可用（避免在 UI 线程 import torch）"""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _ensure_ffmpeg_on_path():
    """确保 ffmpeg 能通过 PATH 找到。"""
    import subprocess
    import shutil

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)

    # imageio-ffmpeg 的可能不是 ffmpeg.exe，需要确保标准名字可用
    ffmpeg_name = "ffmpeg.exe"
    ffmpeg_standard = os.path.join(ffmpeg_dir, ffmpeg_name)
    if not os.path.isfile(ffmpeg_standard):
        try:
            shutil.copy2(ffmpeg_exe, ffmpeg_standard)
        except Exception:
            pass

    if ffmpeg_dir and os.path.isdir(ffmpeg_dir):
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if ffmpeg_dir not in path_parts:
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

    return ffmpeg_exe


class TranscribeThread(QThread):
    """转录线程"""
    progress = pyqtSignal(str, int)  # (消息, 百分比)
    finished = pyqtSignal(bool, str)

    def __init__(self, input_file, output_dir, language, funasr_cached=None, target_lang=None):
        super().__init__()
        self.input_file = input_file
        self.output_dir = output_dir
        self.language = language
        self.target_lang = target_lang  # 翻译目标语言；None 表示不翻译
        # FunASR 缓存的 {lang: {"asr":.., "vad":.., "punc":..}} 模型；None 表示本线程内加载
        self.funasr_cached = funasr_cached
        self.funasr_models = None

    def _check_cancelled(self):
        if self.isInterruptionRequested():
            raise TranscriptionCancelled()

    def _run_funasr(self, cached_models):
        """FunASR 引擎：ffmpeg 提取 16k WAV -> 按语言选模型 -> 切行/分段 -> 补标点
        - 中文:  paraformer-zh(带 VAD+逐字时间戳) + ct-punc
        - 英/日/韩/粤: fsmn-vad 分段 + SenseVoiceSmall(多语种、自带标点)
        """
        import subprocess

        self.progress.emit("正在提取音频...", 10)
        audio_path = os.path.join(self.output_dir, "temp_audio.wav")
        ffmpeg_exe = _ensure_ffmpeg_on_path()
        cmd = [
            ffmpeg_exe, "-i", self.input_file,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-y", audio_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        self._check_cancelled()

        from funasr import AutoModel
        from subtitle_utils import funasr_result_to_segments, clean_funasr_text

        device = "cuda:0" if _cuda_available() else "cpu"
        lang = self.language or "zh"  # 'zh' | 'en' | 'ja' | 'ko' | 'yue'

        # cached_models: {lang: {"asr":..., "vad":..., "punc":...}}
        if cached_models is None:
            cached_models = {}
        models = cached_models.get(lang)
        if models is None:
            self.progress.emit("正在加载 FunASR 模型...", 30)
            if lang == "zh":
                asr = AutoModel(model="paraformer-zh", vad_model="fsmn-vad",
                                disable_update=True, device=device)
                punc = AutoModel(model="ct-punc", disable_update=True, device=device)
                vad = None
            else:
                vad = AutoModel(model="fsmn-vad", disable_update=True, device=device)
                asr = AutoModel(model="iic/SenseVoiceSmall",
                                disable_update=True, device=device)
                punc = None
            models = {"asr": asr, "vad": vad, "punc": punc}
            self._check_cancelled()
        cached_models[lang] = models
        self.funasr_models = cached_models

        asr, vad, punc = models["asr"], models["vad"], models["punc"]

        self.progress.emit("正在转录...", 50)
        if vad is None:
            # 中文：逐字时间戳，按停顿/字数切行
            res = asr.generate(input=audio_path, batch_size_s=300)
            self._check_cancelled()
            self.progress.emit("正在生成字幕...", 90)
            segments = funasr_result_to_segments(res, punc)
        else:
            # 英/日/韩/粤：VAD 分段得到每段起止，逐段转录
            import soundfile as sf

            vad_res = vad.generate(input=audio_path, max_single_segment_time=60000)
            self._check_cancelled()
            vad_segs = vad_res[0]["value"] if vad_res else []
            if not vad_segs:
                raise RuntimeError("未检测到语音内容")

            audio, sr = sf.read(audio_path, dtype="float32")
            clips = [audio[int(s * sr / 1000):int(e * sr / 1000)] for s, e in vad_segs]

            res = asr.generate(input=clips, language=lang, use_itn=True,
                               batch_size_s=300)
            self._check_cancelled()

            self.progress.emit("正在生成字幕...", 90)
            segments = []
            for (s_ms, e_ms), item in zip(vad_segs, res):
                text = clean_funasr_text(item.get("text", ""))
                if text:
                    segments.append({
                        "start": round(s_ms / 1000.0, 3),
                        "end": round(e_ms / 1000.0, 3),
                        "text": text,
                    })

        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass

        if not segments:
            raise RuntimeError("未识别到语音内容")

        base_name = os.path.splitext(os.path.basename(self.input_file))[0]
        base_path = os.path.join(self.output_dir, base_name)
        srt_path = save_subtitle_files(base_path, segments)
        self.progress.emit(f"转录完成！共 {len(segments)} 个片段", 90)

        if not self.target_lang:
            self.progress.emit(f"转录完成！共 {len(segments)} 个片段", 100)
            self.finished.emit(True, srt_path)
            return

        # 翻译：先释放 FunASR 模型显存，再在独立进程中加载翻译模型
        self.progress.emit("正在准备翻译...", 92)
        cached_models.clear()
        self.funasr_models = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        from subtitle_utils import save_translated_srt

        self.progress.emit("正在加载翻译模型（首次约 1 分钟）...", 93)
        import json
        import tempfile
        fd, in_path = tempfile.mkstemp(suffix=".json", prefix="tr_in_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump([{"text": s["text"]} for s in segments], f, ensure_ascii=False)
            translated = _run_translate_worker(
                in_path, self.target_lang,
                lambda i, n: self.progress.emit(
                    f"正在翻译 {i}/{n}...", 93 + int(i / n * 6)),
                self.isInterruptionRequested,
                lambda m, p: self.progress.emit(m, p),
            )
        finally:
            try:
                os.remove(in_path)
            except OSError:
                pass
        self._check_cancelled()
        for seg, t in zip(segments, translated):
            seg["translated_text"] = t
        srt_path = save_translated_srt(base_path, segments, self.target_lang)
        self.progress.emit(f"转录+翻译完成！共 {len(segments)} 段", 100)
        self.finished.emit(True, srt_path)

    def run(self):
        try:
            self._run_funasr(self.funasr_cached)
        except TranscriptionCancelled:
            self.finished.emit(False, "已取消")
        except Exception as e:
            self.finished.emit(False, str(e))


class TranslateThread(QThread):
    """翻译现有字幕文件线程"""
    progress = pyqtSignal(str, int)  # (消息, 百分比)
    finished = pyqtSignal(bool, str)

    def __init__(self, file_path, target_lang):
        super().__init__()
        self.file_path = file_path
        self.target_lang = target_lang

    def run(self):
        try:
            from subtitle_utils import load_segments, save_translated_srt

            self.progress.emit("正在读取字幕...", 5)
            segments = load_segments(self.file_path)
            if not segments:
                raise RuntimeError("字幕文件为空或格式无法解析")

            self.progress.emit("正在加载翻译模型（首次约 1 分钟）...", 10)
            translated = _run_translate_worker(
                self.file_path, self.target_lang,
                lambda i, n: self.progress.emit(
                    f"正在翻译 {i}/{n}...", 10 + int(i / n * 85)),
                self.isInterruptionRequested,
                lambda m, p: self.progress.emit(m, p),
            )
            if self.isInterruptionRequested():
                raise TranscriptionCancelled()
            for seg, t in zip(segments, translated):
                seg["translated_text"] = t
            base = os.path.splitext(self.file_path)[0]
            srt_path = save_translated_srt(base, segments, self.target_lang)
            self.progress.emit("翻译完成！", 100)
            self.finished.emit(True, srt_path)
        except TranscriptionCancelled:
            self.finished.emit(False, "已取消")
        except Exception as e:
            self.finished.emit(False, str(e))


class DeviceCheckThread(QThread):
    """后台检测 GPU / CPU（torch import 较慢，不阻塞窗口显示）"""
    result = pyqtSignal(str)  # "GPU (CUDA)" 或 "CPU"

    def run(self):
        try:
            import torch
            device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        except Exception:
            device = "CPU"
        self.result.emit(device)


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
        self.close_after_cancel = False

        # 模型缓存：按语言缓存 FunASR 模型 {lang: {"asr":.., "vad":.., "punc":..}}
        self._funasr_models = None

        # 异步检测 GPU / CPU
        self.device_thread = None
        self._start_device_check()

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

        # 字幕语言 + 翻译目标语言（一行）
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("字幕语言:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems([
            "自动检测",
            "中文 (zh)", "英文 (en)", "日文 (ja)", "韩文 (ko)", "粤语 (yue)",
        ])
        self.lang_combo.setCurrentText("自动检测")
        lang_layout.addWidget(self.lang_combo)

        lang_layout.addWidget(QLabel("翻译目标语言:"))
        self.translate_combo = QComboBox()
        self.translate_combo.addItems(["不翻译", "中文 (zh)", "英文 (en)", "日文 (ja)", "韩文 (ko)"])
        lang_layout.addWidget(self.translate_combo)

        # GPU / CPU 状态（异步检测，避免启动时卡在 torch import）
        self.device_label = QLabel("检测中...")
        self.device_label.setStyleSheet(
            "color: #888; font-weight: bold; padding: 2px 6px;"
        )
        lang_layout.addWidget(self.device_label)
        lang_layout.addStretch()
        input_layout.addLayout(lang_layout)

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
        # 允许标签收缩，长消息不撑开窗口
        self.status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setFixedWidth(200)
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

        status_layout.addWidget(self.status_label, 1)
        status_layout.addWidget(self.progress_bar)
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

        self.translate_btn = QPushButton("翻译现有字幕")
        self.translate_btn.clicked.connect(self.translate_existing)
        self.translate_btn.setStyleSheet(BTN_STYLE)

        self.exit_btn = QPushButton("退出")
        self.exit_btn.clicked.connect(self.close)
        self.exit_btn.setStyleSheet(BTN_STYLE)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.reselect_btn)
        btn_layout.addWidget(self.open_folder_btn)
        btn_layout.addWidget(self.translate_btn)
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

    def _start_device_check(self):
        def _on_done():
            self.device_thread = None

        self.device_thread = DeviceCheckThread()
        self.device_thread.result.connect(self._apply_device)
        self.device_thread.finished.connect(_on_done)
        self.device_thread.start()

    def _apply_device(self, device):
        color = "#0070CB" if device == "GPU (CUDA)" else "#888"
        self.device_label.setText(device)
        self.device_label.setStyleSheet(
            f"color: {color}; font-weight: bold; padding: 2px 6px;"
        )

    def open_output_folder(self):
        if self.output_dir and os.path.isdir(self.output_dir):
            os.startfile(self.output_dir)

    def closeEvent(self, event):
        if hasattr(self, "thread") and self.thread is not None and self.thread.isRunning():
            if self.close_after_cancel:
                event.ignore()
                return
            reply = QMessageBox.question(
                self,
                "正在转录",
                "转录仍在进行。要取消任务并在当前阶段结束后退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.close_after_cancel = True
                self.thread.requestInterruption()
                self.status_label.setText("正在取消，等待当前阶段结束...")
                self.output_text.append("\n正在取消，等待当前阶段结束...")
                self.start_btn.setEnabled(False)
                self.select_btn.setEnabled(False)
                self.reselect_btn.setEnabled(False)
                self.translate_btn.setEnabled(False)
            event.ignore()
            return
        # 等待后台 GPU 检测线程结束（最长 5 秒），避免销毁运行中的线程
        if self.device_thread is not None and self.device_thread.isRunning():
            self.device_thread.wait(5000)
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

    def _target_lang(self):
        """从翻译下拉框取语言代码；"不翻译" 返回 None"""
        text = self.translate_combo.currentText()
        if text == "不翻译":
            return None
        return text.split("(")[1].rstrip(")")

    def translate_existing(self):
        """翻译已有的 .srt / .json 字幕文件"""
        target_lang = self._target_lang()
        if not target_lang:
            QMessageBox.warning(self, "提示", '请先在"翻译目标语言"中选择目标语言')
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择字幕文件", "", "字幕文件 (*.srt *.json);;所有文件 (*)"
        )
        if not file_path:
            return

        self.start_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.reselect_btn.setEnabled(False)
        self.translate_btn.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.output_text.clear()

        self.thread = TranslateThread(file_path, target_lang)
        self.thread.progress.connect(self.on_progress)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

    def start_transcribe(self):
        if not self.input_file:
            return

        # 获取输出目录（与输入文件同目录）
        self.output_dir = os.path.dirname(self.input_file)
        # 从下拉框提取语言代码，如 "中文 (zh)" -> "zh"，"自动检测" -> None
        lang_text = self.lang_combo.currentText()
        if lang_text == "自动检测":
            language = None
        else:
            language = lang_text.split("(")[1].rstrip(")")

        # 自动检测/中文走 paraformer-zh（逐字时间戳），其余走 SenseVoice
        funasr_lang_map = {None: "zh", "zh": "zh", "en": "en", "ja": "ja", "ko": "ko", "yue": "yue"}
        language = funasr_lang_map.get(language, "zh")

        target_lang = self._target_lang()

        # 禁用按钮
        self.start_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.reselect_btn.setEnabled(False)
        self.translate_btn.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.output_text.clear()

        # 需要翻译时丢弃 FunASR 缓存，转录后释放显存给翻译模型
        funasr_cached = self._funasr_models
        if target_lang and funasr_cached:
            self._funasr_models = None
            funasr_cached = None

        # 启动转录线程
        self.thread = TranscribeThread(
            self.input_file, self.output_dir, language, funasr_cached, target_lang,
        )
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
        self.translate_btn.setEnabled(True)

        # 回存模型供下次转录复用（加载失败则为 None，下次重新加载）
        self._funasr_models = getattr(self.thread, "funasr_models", None)

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
            except Exception:
                pass

            QMessageBox.information(self, "完成", f"字幕已生成！\n{message}")
        elif message == "已取消":
            self.status_label.setText("已取消")
            self.output_text.append("\n任务已取消")
        else:
            self.status_label.setText(f"错误: {message}")
            self.output_text.append(f"\n错误: {message}")
            QMessageBox.critical(self, "错误", f"转录失败:\n{message}")

        if self.close_after_cancel:
            self.thread = None
            self.close_after_cancel = False
            QTimer.singleShot(0, self.close)


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
