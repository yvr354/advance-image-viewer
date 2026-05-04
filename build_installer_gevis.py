"""
Gevis Image Viewer — Installer EXE Builder
Packages install_gevis.py + the built app into a single Gevis_ImageViewer_Setup.exe

Steps:
  1.  venv\Scripts\python build_gevis.py              ← builds the app
  2.  venv\Scripts\python build_installer_gevis.py    ← builds the installer EXE

Give dist\Gevis_ImageViewer_Setup.exe to anyone — no Python needed.
"""

import subprocess
import sys
import os
from pathlib import Path

ROOT     = Path(__file__).parent
APP_DIR  = ROOT / "dist" / "Gevis_ImageViewer"
ICON     = ROOT / "resources" / "icons" / "logo_gevis.ico"
LOGO_PNG = ROOT / "resources" / "icons" / "logo_gevis.png"

# Tcl/Tk DLLs — required for tkinter installer UI
_TCL_BIN = Path(r"C:\ProgramData\anaconda3\Library\bin")
TK_DLL   = _TCL_BIN / "tk86t.dll"
TCL_DLL  = _TCL_BIN / "tcl86t.dll"

print("=" * 60)
print("  Gevis Image Viewer — Building Installer EXE")
print("=" * 60)

if not APP_DIR.exists():
    print(f"\nERROR: App not built yet.")
    print(f"  Missing: {APP_DIR}")
    print(f"\n  Run first:  venv\\Scripts\\python build_gevis.py")
    input("\nPress Enter to exit...")
    sys.exit(1)

print(f"\n  App folder  : {APP_DIR}")
print(f"  Output      : dist\\Gevis_ImageViewer_Setup.exe\n")

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--noconfirm",
    "--onefile",                          # single EXE — everything packed in
    "--windowed",                         # no console window
    "--name", "Gevis_ImageViewer_Setup",
    "--icon", str(ICON),
    # Bundle the entire built app inside the installer
    "--add-data", f"{APP_DIR};app_files",
    # Bundle logo for splash screen
    "--add-data", f"{LOGO_PNG};resources/icons",
    "--hidden-import", "PIL",
    "--hidden-import", "PIL.Image",
    "--hidden-import", "PIL.ImageTk",
]

# Bundle Tcl/Tk DLLs so tkinter works on any machine
if TK_DLL.exists():
    cmd += ["--add-binary", f"{TK_DLL};."]
if TCL_DLL.exists():
    cmd += ["--add-binary", f"{TCL_DLL};."]

cmd += ["install_gevis.py"]

print("Running PyInstaller (this takes 1-3 minutes — app is large)...\n")
result = subprocess.run(cmd, cwd=ROOT)

if result.returncode == 0:
    out = ROOT / "dist" / "Gevis_ImageViewer_Setup.exe"
    size_mb = out.stat().st_size / 1024 / 1024 if out.exists() else 0
    print("\n" + "=" * 60)
    print("  INSTALLER BUILT SUCCESSFULLY!")
    print(f"  File : {out}")
    print(f"  Size : {size_mb:.0f} MB")
    print("=" * 60)
    print("\n  Give this single EXE to anyone.")
    print("  Double-click > splash screen > Install > Done.")
else:
    print("\n" + "=" * 60)
    print("  BUILD FAILED — check errors above")
    print("=" * 60)

input("\nPress Enter to exit...")
