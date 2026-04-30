"""
VyuhaAI Image Viewer — EXE Builder
Run this file to package the app into a Windows installer.
Usage: venv\Scripts\python build_exe.py
"""

import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

print("=" * 60)
print("  VyuhaAI Image Viewer — Building EXE")
print("=" * 60)

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--noconfirm",
    "--windowed",
    "--name", "VyuhaAI_ImageViewer",
    "--icon", os.path.join("resources", "icons", "logo.ico"),
    "--add-data", "resources;resources",
    "--add-data", "src;src",
    "--hidden-import", "PyQt6.QtOpenGL",
    "--hidden-import", "PyQt6.QtOpenGLWidgets",
    "--hidden-import", "pyqtgraph",
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
    "--copy-metadata", "pyqtgraph",
    "--copy-metadata", "numpy",
    "--copy-metadata", "opencv-python",
    "main.py",
]

print("\nRunning PyInstaller...\n")
result = subprocess.run(cmd, cwd=ROOT)

if result.returncode == 0:
    exe_path = os.path.join(ROOT, "dist", "VyuhaAI_ImageViewer", "VyuhaAI_ImageViewer.exe")
    print("\n" + "=" * 60)
    print("  BUILD SUCCESSFUL!")
    print(f"  EXE location:")
    print(f"  {exe_path}")
    print("=" * 60)
    print("\nDouble-click the EXE to test it.")
    print("Then use Inno Setup with installer.iss to create the installer.")
else:
    print("\n" + "=" * 60)
    print("  BUILD FAILED — check errors above")
    print("=" * 60)

input("\nPress Enter to exit...")
