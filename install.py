"""
VyuhaAI Image Viewer — Windows Setup
Double-click VyuhaAI_ImageViewer_Setup.exe to install.
Uninstall via Windows Settings → Apps → VyuhaAI Image Viewer → Uninstall.
"""

import os
import sys
import shutil
import winreg
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import threading


# ── Paths ──────────────────────────────────────────────────────────────────

if getattr(sys, "frozen", False):
    SRC = Path(sys._MEIPASS) / "app_files"
else:
    SRC = Path(__file__).parent / "dist" / "VyuhaAI_ImageViewer"

DEST    = Path(os.environ["LOCALAPPDATA"]) / "VyuhaAI Image Viewer"
EXE     = DEST / "VyuhaAI_ImageViewer.exe"
SETUP   = DEST / "VyuhaAI_ImageViewer_Setup.exe"
START   = Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs/VyuhaAI Image Viewer"
DESKTOP = Path.home() / "Desktop"
EXTS    = [".tiff", ".tif", ".bmp", ".png", ".jpg", ".jpeg", ".pgm", ".ppm", ".exr"]
UNREG   = rf'"{SETUP}" /uninstall'


# ── Install ────────────────────────────────────────────────────────────────

def do_install(log):
    try:
        log("Copying application files…")
        if DEST.exists():
            shutil.rmtree(DEST)
        shutil.copytree(SRC, DEST)
        log(f"  Installed to: {DEST}")

        log("Creating Start Menu shortcut…")
        START.mkdir(parents=True, exist_ok=True)
        _shortcut(EXE, START / "VyuhaAI Image Viewer.lnk")

        log("Creating Desktop shortcut…")
        _shortcut(EXE, DESKTOP / "VyuhaAI Image Viewer.lnk")

        log("Registering file associations…")
        _reg(r"Software\Classes\VyuhaAI.ImageFile", "", "VyuhaAI Image Viewer")
        _reg(r"Software\Classes\VyuhaAI.ImageFile\DefaultIcon", "", f"{EXE},0")
        _reg(r"Software\Classes\VyuhaAI.ImageFile\shell\open\command", "", f'"{EXE}" "%1"')
        for ext in EXTS:
            _reg(f"Software\\Classes\\{ext}", "", "VyuhaAI.ImageFile")
            log(f"  {ext} → VyuhaAI Image Viewer")

        # Copy this Setup EXE into install folder so Control Panel uninstall works
        if getattr(sys, "frozen", False):
            shutil.copy2(sys.executable, SETUP)

        log("Registering with Windows (Add/Remove Programs)…")
        key = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\VyuhaAI_ImageViewer"
        _reg(key, "DisplayName",     "VyuhaAI Image Viewer")
        _reg(key, "DisplayVersion",  "1.0.0")
        _reg(key, "Publisher",       "VyuhaAI")
        _reg(key, "DisplayIcon",     str(EXE))
        _reg(key, "InstallLocation", str(DEST))
        _reg(key, "UninstallString", UNREG)   # Windows calls this on uninstall
        _reg(key, "NoModify",        "1")
        _reg(key, "NoRepair",        "1")

        import ctypes
        ctypes.windll.shell32.SHChangeNotify(0x08000000, 0, None, None)

        log("")
        log("✓  Installation complete!")
        log(f"   Location : {DEST}")
        log("   Shortcuts: Desktop + Start Menu")
        log(f"   Opens    : {', '.join(EXTS)}")
        log("   Uninstall: Windows Settings → Apps → VyuhaAI Image Viewer")
        return True
    except Exception as e:
        log(f"ERROR: {e}")
        return False


# ── Uninstall ──────────────────────────────────────────────────────────────

def do_uninstall(log):
    try:
        log("Removing application files…")
        if DEST.exists():
            shutil.rmtree(DEST)

        log("Removing shortcuts…")
        for p in [DESKTOP / "VyuhaAI Image Viewer.lnk",
                  START   / "VyuhaAI Image Viewer.lnk"]:
            if p.exists():
                p.unlink()
        try:
            START.rmdir()
        except Exception:
            pass

        log("Cleaning registry…")
        for ext in EXTS:
            _del_reg(f"Software\\Classes\\{ext}")
        for sub in [
            r"Software\Classes\VyuhaAI.ImageFile\shell\open\command",
            r"Software\Classes\VyuhaAI.ImageFile\shell\open",
            r"Software\Classes\VyuhaAI.ImageFile\shell",
            r"Software\Classes\VyuhaAI.ImageFile\DefaultIcon",
            r"Software\Classes\VyuhaAI.ImageFile",
            r"Software\Microsoft\Windows\CurrentVersion\Uninstall\VyuhaAI_ImageViewer",
        ]:
            _del_reg(sub)

        log("✓  Uninstall complete.")
        return True
    except Exception as e:
        log(f"ERROR: {e}")
        return False


# ── Registry helpers ───────────────────────────────────────────────────────

def _reg(subkey, name, value):
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, subkey) as k:
        winreg.SetValueEx(k, name, 0, winreg.REG_SZ, str(value))

def _del_reg(subkey):
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, subkey)
    except (FileNotFoundError, OSError):
        pass

def _shortcut(target: Path, link: Path):
    ps = (
        f'$s=(New-Object -COM WScript.Shell).CreateShortcut("{link}");'
        f'$s.TargetPath="{target}";'
        f'$s.IconLocation="{target},0";'
        f'$s.Save()'
    )
    subprocess.run(["powershell", "-Command", ps], capture_output=True)


# ── Logo helper ────────────────────────────────────────────────────────────

def _logo_path():
    if getattr(sys, "frozen", False):
        p = Path(sys._MEIPASS) / "resources" / "icons" / "logo.png"
    else:
        p = Path(__file__).parent / "resources" / "icons" / "logo.png"
    return p if p.exists() else None


# ── Uninstall dialog (called by Windows Settings → Apps → Uninstall) ───────

class UninstallUI(tk.Tk):
    """Minimal confirm + progress shown when Windows calls /uninstall."""
    def __init__(self):
        super().__init__()
        self.title("VyuhaAI Image Viewer — Uninstall")
        self.resizable(False, False)
        self.configure(bg="#0A0A0F")
        self.geometry("420x280")
        self.update_idletasks()
        x = (self.winfo_screenwidth()  - 420) // 2
        y = (self.winfo_screenheight() - 280) // 2
        self.geometry(f"+{x}+{y}")
        self._build()

    def _build(self):
        tk.Frame(self, bg="#FF5252", height=4).pack(fill="x")
        tk.Label(self, text="VyuhaAI Image Viewer",
                 font=("Segoe UI", 14, "bold"),
                 fg="#FF5252", bg="#0A0A0F").pack(pady=(16, 2))
        tk.Label(self, text="Remove this application from your PC?",
                 font=("Segoe UI", 10), fg="#888899", bg="#0A0A0F").pack()

        log_frame = tk.Frame(self, bg="#0A0A0F")
        log_frame.pack(fill="both", expand=True, padx=16, pady=10)
        self._log = tk.Text(log_frame, height=5,
                            bg="#111118", fg="#CCCCDD",
                            font=("Consolas", 9),
                            relief="flat", state="disabled", wrap="word")
        self._log.pack(fill="both", expand=True)

        self._progress = ttk.Progressbar(self, mode="indeterminate", length=390)
        self._progress.pack(pady=(0, 8))

        btn_frame = tk.Frame(self, bg="#0A0A0F")
        btn_frame.pack(pady=(0, 14))
        tk.Button(btn_frame, text="  Yes, Uninstall  ",
                  font=("Segoe UI", 10, "bold"),
                  bg="#FF5252", fg="white", relief="flat",
                  padx=16, pady=6, cursor="hand2",
                  command=self._start).pack(side="left", padx=6)
        tk.Button(btn_frame, text="  Cancel  ",
                  font=("Segoe UI", 10),
                  bg="#1A1A2A", fg="#888899", relief="flat",
                  padx=16, pady=6, cursor="hand2",
                  command=self.destroy).pack(side="left", padx=6)

    def _log_line(self, text):
        self._log.configure(state="normal")
        self._log.insert("end", text + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")
        self.update()

    def _start(self):
        self._progress.start(12)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        ok = do_uninstall(self._log_line)
        self._progress.stop()
        if ok:
            self.after(0, lambda: (
                messagebox.showinfo("Done", "VyuhaAI Image Viewer has been removed."),
                self.destroy()
            ))


# ── Install UI (normal double-click) ──────────────────────────────────────

class SplashScreen(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.overrideredirect(True)
        self.configure(bg="#0A0A0F")
        w, h = 400, 260
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        logo_p = _logo_path()
        if logo_p:
            try:
                from PIL import Image, ImageTk
                img = Image.open(logo_p).resize((96, 96), Image.LANCZOS)
                self._photo = ImageTk.PhotoImage(img)
                tk.Label(self, image=self._photo, bg="#0A0A0F").pack(pady=(32, 8))
            except Exception:
                pass

        tk.Label(self, text="VyuhaAI Image Viewer",
                 font=("Segoe UI", 16, "bold"), fg="#00B4D8", bg="#0A0A0F").pack()
        tk.Label(self, text="Industrial Image Analysis",
                 font=("Segoe UI", 10), fg="#555566", bg="#0A0A0F").pack(pady=(4, 0))
        tk.Label(self, text="Preparing setup…",
                 font=("Segoe UI", 9), fg="#333344", bg="#0A0A0F").pack(pady=(16, 0))
        tk.Frame(self, bg="#00B4D8", height=3).pack(fill="x", side="bottom")
        self.lift()
        self.update()


class InstallerUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.withdraw()
        splash = SplashScreen(self)
        self.after(1800, lambda: self._show_main(splash))

    def _show_main(self, splash):
        splash.destroy()
        self.title("VyuhaAI Image Viewer — Setup")
        self.resizable(False, False)
        self.configure(bg="#0A0A0F")
        self.geometry("520x460")
        self.update_idletasks()
        x = (self.winfo_screenwidth()  - 520) // 2
        y = (self.winfo_screenheight() - 460) // 2
        self.geometry(f"+{x}+{y}")
        self._build_logo()
        self._build()
        self.deiconify()

    def _build_logo(self):
        logo_p = _logo_path()
        if not logo_p:
            return
        try:
            from PIL import Image, ImageTk
            img = Image.open(logo_p).resize((52, 52), Image.LANCZOS)
            self._logo_photo = ImageTk.PhotoImage(img)
            tk.Label(self, image=self._logo_photo, bg="#0A0A0F").pack(pady=(14, 0))
        except Exception:
            pass

    def _build(self):
        tk.Frame(self, bg="#00B4D8", height=5).pack(fill="x")

        tk.Label(self, text="VyuhaAI Image Viewer",
                 font=("Segoe UI", 17, "bold"), fg="#00B4D8", bg="#0A0A0F").pack(pady=(14, 2))
        tk.Label(self, text="Version 1.0.0  —  Industrial Image Analysis",
                 font=("Segoe UI", 10), fg="#888899", bg="#0A0A0F").pack()
        tk.Label(self, text=f"Install to:  {DEST}",
                 font=("Segoe UI", 9), fg="#555566", bg="#0A0A0F").pack(pady=(6, 0))
        tk.Label(self, text="Opens: .tiff  .tif  .bmp  .png  .jpg  .jpeg  .pgm  .ppm  .exr",
                 font=("Segoe UI", 9), fg="#555566", bg="#0A0A0F").pack(pady=(2, 0))

        already = DEST.exists() and EXE.exists()
        self._status_lbl = tk.Label(
            self,
            text="● Already installed — click Install to update" if already else "● Ready to install",
            font=("Segoe UI", 9, "bold"),
            fg="#00B4D8" if already else "#555566",
            bg="#0A0A0F"
        )
        self._status_lbl.pack(pady=(6, 0))

        log_frame = tk.Frame(self, bg="#0A0A0F")
        log_frame.pack(fill="both", expand=True, padx=18, pady=10)
        self._log = tk.Text(log_frame, height=7,
                            bg="#111118", fg="#CCCCDD",
                            font=("Consolas", 9),
                            relief="flat", state="disabled", wrap="word")
        self._log.pack(fill="both", expand=True)

        self._progress = ttk.Progressbar(self, mode="indeterminate", length=484)
        self._progress.pack(pady=(0, 10))

        # Only two buttons: Install + Close
        btn_frame = tk.Frame(self, bg="#0A0A0F")
        btn_frame.pack(pady=(0, 16))

        self._btn_install = tk.Button(
            btn_frame, text="  Install  ",
            font=("Segoe UI", 11, "bold"),
            bg="#00B4D8", fg="white", relief="flat",
            padx=24, pady=8, cursor="hand2",
            command=self._start_install
        )
        self._btn_install.pack(side="left", padx=8)

        tk.Button(
            btn_frame, text="  Close  ",
            font=("Segoe UI", 11),
            bg="#1A1A2A", fg="#888899", relief="flat",
            padx=20, pady=8, cursor="hand2",
            command=self.destroy
        ).pack(side="left", padx=8)

    def _log_line(self, text):
        self._log.configure(state="normal")
        self._log.insert("end", text + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")
        self.update()

    def _start_install(self):
        if not SRC.exists():
            messagebox.showerror("Error",
                f"App files not found:\n{SRC}\n\nRun build_exe.py first.")
            return
        self._btn_install.configure(state="disabled")
        self._progress.start(12)
        threading.Thread(target=self._run_install, daemon=True).start()

    def _run_install(self):
        ok = do_install(self._log_line)
        self._progress.stop()
        self._btn_install.configure(state="normal")
        if ok:
            self._status_lbl.configure(text="● Installed successfully", fg="#00FF88")
            self.after(0, self._ask_launch)

    def _ask_launch(self):
        ans = messagebox.askyesno(
            "Setup Complete",
            "VyuhaAI Image Viewer installed!\n\n"
            "• Desktop shortcut created\n"
            "• Appears in Start Menu — search 'VyuhaAI'\n"
            "• Double-click any .tiff .bmp .png to open it\n"
            "• Uninstall: Windows Settings → Apps\n\n"
            "Launch the app now?"
        )
        if ans and EXE.exists():
            subprocess.Popen([str(EXE)], creationflags=0x00000008)
            self.destroy()


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Called by Windows Settings → Apps → Uninstall
    if "/uninstall" in sys.argv or "--uninstall" in sys.argv:
        app = UninstallUI()
        app.mainloop()
    else:
        # Normal double-click — show installer
        app = InstallerUI()
        app.mainloop()
