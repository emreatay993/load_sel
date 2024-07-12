# -*- mode: python ; coding: utf-8 -*-

# IMPORTANT!!:  Change the name of the source script below before running this spec with pyinstaller

# Use the following command for building the main script for WE Plotter as an exe file:

#pyinstaller we_plotter.spec --clean


from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import copy_metadata

datas = []
datas += collect_data_files('ansys.api')
datas += collect_data_files('ansys.mechanical')
datas += collect_data_files('ansys.platform')
datas += collect_data_files('pythonnet')
datas += collect_data_files('clr_loader')
datas += copy_metadata('ansys-tools-path')
datas += copy_metadata('ansys-mechanical-core')
datas += copy_metadata('ansys-api-mechanical')
datas += copy_metadata('ansys-platform-instancemanagement')


a = Analysis(
    ['anan.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['ansys.api.mechanical'],
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
    name='anan',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    upx=True,
    upx_exclude=[],
    name='anan',
)
