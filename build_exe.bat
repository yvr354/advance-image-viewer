@echo off
echo Building YVR Advanced Image Viewer .exe ...
call venv\Scripts\activate.bat
pyinstaller ^
    --onefile ^
    --windowed ^
    --name "YVR_ImageViewer" ^
    --add-data "resources;resources" ^
    --hidden-import "cv2" ^
    --hidden-import "numpy" ^
    --hidden-import "tifffile" ^
    --hidden-import "pyqtgraph" ^
    main.py
echo.
echo Build complete. EXE is in dist\YVR_ImageViewer.exe
pause
