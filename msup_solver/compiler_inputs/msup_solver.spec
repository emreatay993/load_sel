# -*- mode: python ; coding: utf-8 -*-

# IMPORTANT!!:  Change the name of the source script below before running this spec with pyinstaller

# Use the following command for building the main script for WE Plotter as an exe file:

#pyinstaller msup_solver.spec --clean --noconfirm

a = Analysis(
    ['MSUP_Smart_Solver_w_Modal_Def.py'],
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
    exclude_binaries=False,
    name='MSUP Transient Expansion Tool ',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    onefile=False,
    icon="icon.ico",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MSUP_Transient_Expansion_Tool',
)
