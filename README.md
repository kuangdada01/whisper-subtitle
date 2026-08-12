# whisper-subtitle

基于 FunASR（Paraformer / SenseVoice）的视频/音频语音转字幕工具，支持图形界面和命令行两种使用方式，可选 LLM 翻译。

## 功能特性

- 视频（mp4/mkv/avi/mov）或音频（wav/mp3/flac/m4a）自动转录为 **SRT 字幕** 文件
- 中文使用 **Paraformer-large**（VAD + 逐字时间戳 + 标点恢复），英/日/韩/粤使用 **SenseVoiceSmall**（多语种、自带标点）
- 可选 **LLM 翻译**（Qwen3-4B，4bit 量化，批量并行）将字幕翻译为中/英/日/韩，也可直接翻译已有的 SRT/JSON 字幕
- 支持 **GPU (CUDA)** 加速，无 GPU 时自动回退 CPU
- 本地缓存模型文件，一次下载永久复用
- 输出 **SRT 字幕** 文件

## 安装

```bash
# 克隆仓库
git clone https://github.com/kuangdada01/whisper-subtitle.git
cd whisper-subtitle

# 安装依赖
pip install -r requirements.txt
```

> GPU 用户请先安装 [PyTorch CUDA 版](https://pytorch.org/get-started/locally/)，否则会自动使用 CPU。

## 使用方式

### 图形界面

```bash
python subtitle_app.py
```

操作步骤：

1. 点击 **选择文件**，选取视频或音频
2. 选择 **字幕语言**：自动检测 / 中文 / 英文 / 日文 / 韩文 / 粤语
3. 选择 **翻译目标语言**：不翻译 / 中文 / 英文 / 日文 / 韩文
4. 点击 **开始转录**，等待完成
5. 字幕文件 `.srt` 生成在源文件同目录下；选择翻译后另存为 `{文件名}.{目标语言}.srt`

也可点击 **翻译已有字幕文件**，直接翻译已存在的 SRT/JSON 字幕。

### 命令行

```bash
# 先提取音频
ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 -y temp_audio.wav

# 运行转录（可选 --lang zh/en/ja/ko/yue，--translate 指定翻译目标语言）
python transcribe.py --lang zh --translate en
```

## 模型说明

| 模型 | 用途 |
|---|---|
| iic/Speech-Paraformer-large (paraformer-zh) | 中文语音识别（VAD + 逐字时间戳） |
| iic/ct-punc | 中文标点恢复 |
| iic/SenseVoiceSmall | 英/日/韩/粤多语种识别（自带标点） |
| iic/fsmn-vad | 语音活动检测（音频分段） |
| Qwen/Qwen3-4B | LLM 字幕翻译（4bit 量化，约 2.5GB 显存，非思考模式） |

首次使用会自动下载模型并缓存到本地，之后无需重复下载。

模型缓存位置：
- FunASR 模型：`models_cache\models`（modelscope 下载缓存目录；缓存路径须为纯 ASCII，否则 SenseVoice 的 sentencepiece 分词器无法加载）
- Transformers 模型：`models_cache\hf`（`subtitle_utils.py` 将 `HF_HOME` 指向项目内 `models_cache\hf`）
- 翻译模型 Qwen3-4B 原版：`models_cache\qwen3-4b-hf`（bf16，仅用于生成量化缓存）
- 翻译模型 Qwen3-4B 预量化缓存：`models_cache\qwen3-4b-bnb4bit`（NF4 预量化，加载约 5 秒；不存在时自动回退到 CPU 现量化，约 47 秒）。重新生成缓存：`python make_4bit_cache.py models_cache\qwen3-4b-hf models_cache\qwen3-4b-bnb4bit`
- 旧 Qwen3-8B 缓存已删除

翻译在独立子进程（`translate_worker.py`）中执行：bnb 量化模型与 FunASR 在同一进程交替加载会触发崩溃（访问冲突），子进程每次全新加载，与转录进程隔离；即使子进程异常，主程序也只会提示错误而不会闪退。

> 稳定性说明（RTX 50 系 / Blackwell）：bnb 在显卡上直接做 4bit 量化（`device_map` 指向 GPU）的量化内核不稳定，偶发访问冲突导致进程崩溃；因此加载时先在 CPU 完成量化再整体移入显存（`device_map={"": "cpu"}` + `.to("cuda")`），**推理仍 100% 在 GPU 执行**（约 2.5GB 显存）。字幕按 16 行一批并行生成（左 padding，规避右 padding 导致的输出噪声），每行附带前 2 句原文作为上下文，提升指代/省略主语的翻译准确性（实测约 0.25 秒/行，1800 行约 8 分钟）。预量化缓存（`qwen3-4b-bnb4bit`）直接读取量化权重，无量化开销且规避了加载期 bf16 临时占用导致的内存耗尽崩溃。若子进程仍异常退出，GUI 会自动重启工作器重试（最多 3 次）并提示。

## 技术栈

| 技术 | 用途 |
|---|---|
| Python 3.10+ | 开发语言 |
| PyQt6 | 图形界面 |
| FunASR | 语音识别引擎 |
| Transformers (Qwen3-4B, 4bit) | LLM 字幕翻译（独立子进程） |
| PyTorch + CUDA | GPU 加速推理 |
| imageio-ffmpeg | 视频音频提取 |
| SciPy / NumPy | 音频数据处理 |

## 许可证

MIT
