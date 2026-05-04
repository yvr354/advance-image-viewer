"""
Gevis Image Viewer — EXE Builder
Run with: venv\Scripts\python build_gevis.py
"""

import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

print("=" * 60)
print("  Gevis Image Viewer — Building EXE")
print("=" * 60)

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--noconfirm",
    "--windowed",
    "--name", "Gevis_ImageViewer",
    "--icon", os.path.join("resources", "icons", "logo_gevis.ico"),
    "--add-data", "resources;resources",
    "--add-data", "src;src",
    "--add-data", "gevis-logo;gevis-logo",
    "--hidden-import", "PyQt6.QtOpenGL",
    "--hidden-import", "PyQt6.QtOpenGLWidgets",
    "--hidden-import", "cv2",
    "--hidden-import", "tifffile",
    "--hidden-import", "imageio",
    "--hidden-import", "pywt",
    "--hidden-import", "skimage",
    "--hidden-import", "scipy",
    "--hidden-import", "numpy",
    "--hidden-import", "imageio.plugins",
    "--hidden-import", "imageio.plugins.pillow",
    "--hidden-import", "imageio.plugins.tifffile",
    "--copy-metadata", "imageio",
    "--copy-metadata", "tifffile",
    "--collect-all", "pyqtgraph",
    "--collect-all", "OpenGL",
    "main.py",
]

print("\nRunning PyInstaller...\n")
result = subprocess.run(cmd, cwd=ROOT)

if result.returncode == 0:
    exe_path = os.path.join(ROOT, "dist", "Gevis_ImageViewer", "Gevis_ImageViewer.exe")
    print("\n" + "=" * 60)
    print("  BUILD SUCCESSFUL!")
    print(f"  EXE: {exe_path}")
    print("=" * 60)
else:
    print("\n" + "=" * 60)
    print("  BUILD FAILED — check errors above")
    print("=" * 60)
