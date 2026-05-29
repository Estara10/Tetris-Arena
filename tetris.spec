# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Tetris AI (lightweight version, no torch)."""

import os
from pathlib import Path

_base = Path(SPECPATH)

# ---- data files to bundle ------------------------------------------------
datas = []

# Background image
bg = _base / "background.png"
if bg.is_file():
    datas.append((str(bg), "."))

# Model checkpoints (for DQN inference if torch is present)
models_dir = _base / "models"
if models_dir.is_dir():
    datas.append((str(models_dir), "models"))

# ---- hidden imports PyInstaller can't detect ----------------------------
hiddenimports = [
    "pygame._view",
]

# ---- excluded modules (training / dev only, shrink exe size) ------------
excludes = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.utils.tensorboard",
    "tensorboard",
    "numpy",
    "matplotlib",
    "PIL",
    "docx",
    "xml.etree",
]

a = Analysis(
    [str(_base / "main.py")],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Tetris_AI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
