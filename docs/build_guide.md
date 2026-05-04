# Build Guide — Creating the EXE

## Which branch builds which EXE

| Branch | App EXE | Installer EXE |
|---|---|---|
| `gevis-imageviewer` | `Gevis_ImageViewer.exe` | `Gevis_ImageViewer_Setup.exe` |
| `main` | `VyuhaAI_ImageViewer.exe` | `VyuhaAI_ImageViewer_Setup.exe` |

---

## Gevis Build — branch: gevis-imageviewer

### Step 1 — Switch to the correct branch
```
git checkout gevis-imageviewer
```

### Step 2 — Build the app
```
venv\Scripts\python build_gevis.py
```
Output: `dist\Gevis_ImageViewer\Gevis_ImageViewer.exe`
Wait for: `BUILD SUCCESSFUL!`

### Step 3 — Build the installer
```
venv\Scripts\python build_installer_gevis.py
```
Output: `dist\Gevis_ImageViewer_Setup.exe` (~168 MB)
Wait for: `INSTALLER BUILT SUCCESSFULLY!`

### Step 4 — Give to user
Hand over `dist\Gevis_ImageViewer_Setup.exe` — single file, no Python needed.
Double-click → splash screen → Install → app is on Desktop + Start Menu.

---

## VyuhaAI Build — branch: main

### Step 1 — Switch to the correct branch
```
git checkout main
```

### Step 2 — Build the app
```
venv\Scripts\python build_exe.py
```
Output: `dist\VyuhaAI_ImageViewer\VyuhaAI_ImageViewer.exe`
Wait for: `BUILD SUCCESSFUL!`

### Step 3 — Build the installer
```
venv\Scripts\python build_installer.py
```
Output: `dist\VyuhaAI_ImageViewer_Setup.exe` (~169 MB)
Wait for: `INSTALLER BUILT SUCCESSFULLY!`

### Step 4 — Give to user
Hand over `dist\VyuhaAI_ImageViewer_Setup.exe` — single file, no Python needed.

---

## Notes

- Always run Step 2 before Step 3 — installer needs the built app folder
- `dist\` folder is NOT committed to git — rebuild any time
- If tkinter DLL error on installer: the `build_installer_gevis.py` already bundles `tk86t.dll` and `tcl86t.dll` from Anaconda automatically
- Python to use: always `venv\Scripts\python` — not system Python, not Anaconda directly
