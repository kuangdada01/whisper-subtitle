# whisper-subtitle

基于 OpenAI Whisper 的视频/音频语音转字幕工具，支持图形界面和命令行两种使用方式。

## 功能特性

- 视频（mp4/mkv/avi/mov）或音频（wav/mp3/flac/m4a）自动转录为 **SRT 字幕** 文件
- 支持中、英、日、韩等 20+ 种语言字幕，也可自动检测源语言
- 支持 **GPU (CUDA)** 加速，无 GPU 时自动回退 CPU
- 首次使用自动从国内 CDN 镜像下载模型，无需手动配置
- 本地缓存模型文件，一次下载永久复用
- 同时输出 SRT 字幕 和 JSON 时间轴数据

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
2. 选择 **模型大小**：tiny / base / small / medium / large
3. 选择 **字幕语言**，默认自动检测
4. 点击 **开始转录**，等待完成
5. 字幕文件 `.srt` 和 `.json` 生成在源文件同目录下

### 命令行

```bash
# 先提取音频
ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 -y temp_audio.wav

# 运行转录
python transcribe.py
```

## 模型说明

| 模型 | 大小 | 速度 | 适用场景 |
|---|---|---|---|
| tiny | ~75 MB | 最快 | 实时/快速预览 |
| base | ~145 MB | 快 | 短音频 |
| small | ~488 MB | 中等 | 日常使用 |
| **medium** | **~1.5 GB** | **较慢** | **推荐，准确度与速度平衡** |
| large | ~3.1 GB | 慢 | 最高准确度 |

首次使用会自动下载所选模型并缓存到本地，之后无需重复下载。

## 技术栈

| 技术 | 用途 |
|---|---|
| Python 3.10+ | 开发语言 |
| PyQt6 | 图形界面 |
| OpenAI Whisper | 语音识别引擎 |
| PyTorch + CUDA | GPU 加速推理 |
| imageio-ffmpeg | 视频音频提取 |
| SciPy / NumPy | 音频数据处理 |

## 许可证

MIT
