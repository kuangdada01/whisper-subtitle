# -*- mode: python ; coding: utf-8 -*-
# onedir 模式：启动免解压（onefile 每次启动需解压 2.86GB 到临时目录，启动要数分钟）
# 产物为 dist/视频字幕提取工具/ 文件夹（exe + _internal/ 依赖），整个文件夹分发


a = Analysis(
    ['subtitle_app.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='视频字幕提取工具',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='视频字幕提取工具',
)
