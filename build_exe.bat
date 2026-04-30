@echo off
echo ================================================
echo  VyuhaAI Image Viewer - Build Installer
echo ================================================
echo.

call venv\Scripts\activate.bat

echo Step 1: Converting logo to .ico ...
python -c "from PIL import Image; img = Image.open('resources/icons/logo.png'); img.save('resources/icons/logo.ico', format='ICO', sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])"

echo Step 2: Building with PyInstaller ...
pyinstaller ^
    --noconfirm ^
    --windowed ^
    --name "VyuhaAI_ImageViewer" ^
    --icon "resources/icons/logo.ico" ^
    --add-data "resources;resources" ^
    --hidden-import "cv2" ^
    --hidden-import "numpy" ^
    --hidden-import "tifffile" ^
    --hidden-import "pyqtgraph" ^
    --hidden-import "OpenGL" ^
    --hidden-import "OpenGL.GL" ^
    --hidden-import "imageio" ^
    --collect-all "pyqtgraph" ^
    --collect-all "OpenGL" ^
    main.py

echo.
echo Step 3: Creating installer with Inno Setup ...
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
    echo.
    echo ================================================
    echo  Done! Installer: dist\VyuhaAI_ImageViewer_Setup_v1.0.exe
    echo ================================================
) else (
    echo Inno Setup not found — skipping installer step.
    echo App folder is ready at: dist\VyuhaAI_ImageViewer\
    echo Download Inno Setup from: https://jrsoftware.org/isinfo.php
)

pause
