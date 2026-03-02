"""gui.py — Tkinter desktop front-end for the Container Packing Optimizer.

Usage (development):
    python app/gui.py

Usage (frozen PyInstaller exe/app):
    Double-click ContainerOptimizer.exe / ContainerOptimizer.app

UX flow:
    1. User clicks "Select Excel File" → file-picker dialog
    2. User clicks "Run Optimizer" → background thread calls main()
    3. Real-time log streams into the scrollable text area
    4. On completion: "Open Report" and "Open Output Folder" buttons activate
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ── Resolve paths relative to the executable (frozen) or script (dev) ─────────
if getattr(sys, "frozen", False):
    _APP_DIR = Path(sys.executable).resolve().parent
    # Add the bundled app package to sys.path if needed
    _BUNDLE = getattr(sys, "_MEIPASS", None)
    if _BUNDLE:
        sys.path.insert(0, str(_BUNDLE))
else:
    _APP_DIR = Path(__file__).resolve().parent.parent  # project root

# # ── Colour palette ─────────────────────────────────────────────────────────────
# _BG        = "#F0F4F8"   # light background
# _PANEL     = "#FFFFFF"   # white card / panel
# _ACCENT    = "#2563EB"   # friendly blue
# _ACCENT_H  = "#1D4ED8"   # hover (darker blue)
# _TEXT      = "#1E293B"   # near-black body text
# _MUTED     = "#64748B"   # secondary text
# _GREEN     = "#15803D"
# _RED       = "#DC2626"
# _YELLOW    = "#B45309"
# _MONO_FONT = ("Courier New", 10)
# _UI_FONT   = ("Segoe UI", 13) if sys.platform == "win32" else ("Helvetica", 13)
# _TITLE_FONT= ("Segoe UI", 16, "bold") if sys.platform == "win32" else ("Helvetica", 16, "bold")

# --- Clean Professional Palette ---

# --- Clean Professional Palette ---
# --- Dark cohesive palette (flat + consistent) ---
_BG        = "#0B1220"   # window background
_PANEL     = "#0F172A"   # darker than before
_PANEL_H   = "#16213A"
_FIELD     = "#0B1020"   # text/log fields
_BORDER    = "#22304A"   # subtle borders

_BTN     = "#111C33"
_BTN_H   = "#1A2A4A"

_TEXT      = "#E5E7EB"
_MUTED     = "#9CA3AF"

_ACCENT    = "#3B82F6"
_ACCENT_H  = "#2563EB"

_GREEN     = "#22C55E"
_RED       = "#EF4444"
_YELLOW    = "#F59E0B"

_MONO_FONT  = ("SF Mono", 11) if sys.platform == "darwin" else ("Courier New", 10)
_UI_FONT    = ("SF Pro Text", 13) if sys.platform == "darwin" else ("Segoe UI", 13)
_TITLE_FONT = ("SF Pro Display", 18, "bold") if sys.platform == "darwin" else ("Segoe UI", 18, "bold")


class _RedirectText(io.StringIO):
    """Route stdout/stderr to the Tkinter Text widget."""

    def __init__(self, widget: tk.Text):
        super().__init__()
        self._w = widget

    def write(self, s: str) -> int:
        self._w.configure(state="normal")
        self._w.insert(tk.END, s)
        self._w.see(tk.END)
        self._w.configure(state="disabled")
        return len(s)

    def flush(self):
        pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self._style_ttk()
        self.title("Container Packing Optimizer")
        self.configure(bg=_BG)
        self.resizable(True, True)
        self.minsize(760, 560)

        # State
        self._excel_path: str = ""
        self._out_dir: Path | None = None
        self._report_path: Path | None = None
        self._running = False

        self._build_ui()
        self._center_window(900, 680)

    # ── Replace _set_btn_enabled in the App class ─────────────────────────────────

    def _set_btn_enabled(self, btn: _Button, enabled: bool):
        btn.set_enabled(enabled)
    # ── Layout ────────────────────────────────────────────────────────────────
    def _style_ttk(self):
        style = ttk.Style(self)

        # "clam" is the least ugly cross-platform ttk theme
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "TProgressbar",
            troughcolor=_PANEL,
            background=_ACCENT,
            bordercolor=_BORDER,
            lightcolor=_ACCENT,
            darkcolor=_ACCENT,
        )

        style.configure(
            "Vertical.TScrollbar",
            gripcount=0,
            background=_PANEL,
            troughcolor=_BG,
            bordercolor=_BORDER,
            arrowcolor=_MUTED,
        )
        style.configure(
            "Horizontal.TScrollbar",
            gripcount=0,
            background=_PANEL,
            troughcolor=_BG,
            bordercolor=_BORDER,
            arrowcolor=_MUTED,
        )

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=_ACCENT, height=4)
        hdr.pack(fill="x")

        title_fr = tk.Frame(self, bg=_BG, pady=16)
        title_fr.pack(fill="x", padx=24)
        tk.Label(title_fr, text="Container Packing Optimizer",
                 font=_TITLE_FONT, bg=_BG, fg=_TEXT).pack(side="left")
        self._cfg_label = tk.Label(
            title_fr, text="", font=("Courier New", 10),
            bg=_BG, fg=_MUTED, anchor="e", justify="right"
        )
        self._cfg_label.pack(side="right")
        self._update_config_label()

        # File selection card
        card = tk.Frame(self, bg=_PANEL, padx=16, pady=12)
        card.pack(fill="x", padx=24, pady=(0, 8))

        tk.Label(card, text="Excel Input File", font=(*_UI_FONT[:1], _UI_FONT[1], "bold"),
                 bg=_PANEL, fg=_TEXT).grid(row=0, column=0, sticky="w")

        self._file_var = tk.StringVar(value="No file selected")
        tk.Label(card, textvariable=self._file_var,
                 font=_MONO_FONT, bg=_PANEL, fg=_MUTED,
                 wraplength=560, anchor="w", justify="left"
                 ).grid(row=1, column=0, sticky="ew", pady=(4, 0))
        card.columnconfigure(0, weight=1)

        btn_fr = tk.Frame(card, bg=_PANEL)
        btn_fr.grid(row=0, column=1, rowspan=2, sticky="e", padx=(12, 0))
        self._browse_btn = _Button(btn_fr, "Browse…", self._on_browse)
        self._browse_btn.pack()

        # Run controls
        ctrl = tk.Frame(self, bg=_BG)
        ctrl.pack(fill="x", padx=24, pady=4)

        self._run_btn = _Button(ctrl, "Run Optimizer", self._on_run,
                                accent=True, width=18)
        self._run_btn.pack(side="left")

        self._progress = ttk.Progressbar(ctrl, mode="indeterminate", length=200)
        self._progress.pack(side="left", padx=(12, 0))

        self._status_var = tk.StringVar(value="Ready")
        tk.Label(ctrl, textvariable=self._status_var,
                 bg=_BG, fg=_MUTED, font=_UI_FONT).pack(side="left", padx=8)

        # Log area
        log_fr = tk.Frame(self, bg=_PANEL, padx=12, pady=10)
        log_fr.pack(fill="both", expand=True, padx=24, pady=(8, 0))

        tk.Label(log_fr, text="Run Log", font=(*_UI_FONT[:1], _UI_FONT[1], "bold"),
                 bg=_PANEL, fg=_TEXT, anchor="w").pack(fill="x", padx=8, pady=(6, 2))

        txt_fr = tk.Frame(log_fr, bg=_PANEL)
        txt_fr.pack(fill="both", expand=True)

        self._log_box = tk.Text(
            txt_fr, state="disabled", wrap="none",
            bg=_FIELD, fg=_TEXT, insertbackground=_TEXT,
            font=_MONO_FONT,
            relief="flat",
            highlightthickness=1,
            highlightbackground=_BORDER,
            highlightcolor=_BORDER,
            selectbackground=_ACCENT_H,
            selectforeground="#FFFFFF",
        )
        v_scroll = ttk.Scrollbar(txt_fr, orient="vertical",
                                  command=self._log_box.yview)
        h_scroll = ttk.Scrollbar(txt_fr, orient="horizontal",
                                  command=self._log_box.xview)
        self._log_box.configure(yscrollcommand=v_scroll.set,
                                 xscrollcommand=h_scroll.set)
        h_scroll.pack(side="bottom", fill="x")
        v_scroll.pack(side="right", fill="y")
        self._log_box.pack(side="left", fill="both", expand=True)

        # Bottom action bar
        bot = tk.Frame(self, bg=_BG, pady=12)
        bot.pack(fill="x", padx=24)

        self._report_btn = _Button(bot, "Open Report (Excel)", self._open_report,
                                   state="disabled")
        self._report_btn.pack(side="left", padx=(0, 8))

        self._folder_btn = _Button(bot, "Open Output Folder", self._open_folder,
                                   state="disabled")
        self._folder_btn.pack(side="left")

        tk.Label(bot, text="Tip: place optimizer_config.json next to this app to customise container dims",
                 bg=_BG, fg=_MUTED, font=(*_UI_FONT[:1], 10)).pack(side="right")

    # ── Config label ──────────────────────────────────────────────────────────

    def _update_config_label(self):
        try:
            from config import _CONFIG_SOURCE, _USING_DEFAULTS
            from config import CONTAINER_LENGTH_CM, CONTAINER_WIDTH_CM, CONTAINER_HEIGHT_CM
            if _USING_DEFAULTS:
                src = "Built-in defaults (no optimizer_config.json found)"
                color = _YELLOW
            else:
                src = _CONFIG_SOURCE or "custom"
                color = _GREEN
            self._cfg_label.config(
                text=f"Config: {src}\n"
                     f"Container: {CONTAINER_LENGTH_CM}×{CONTAINER_WIDTH_CM}×{CONTAINER_HEIGHT_CM} cm",
                fg=_MUTED,
            )
        except Exception:
            pass

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_browse(self):
        path = filedialog.askopenfilename(
            title="Select Order Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")],
            initialdir=str(_APP_DIR),
        )
        if path:
            self._excel_path = path
            self._file_var.set(path)

    def _on_run(self):
        if self._running:
            return
        if not self._excel_path:
            messagebox.showwarning("No file", "Please select an Excel file first.")
            return

        self._running = True
       
        self._set_btn_enabled(self._run_btn, False)
        self._set_btn_enabled(self._report_btn, False)
        self._set_btn_enabled(self._folder_btn, False)

        self._status_var.set("Running…")
        self._progress.start(12)

        # Clear log
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", tk.END)
        self._log_box.configure(state="disabled")

        thread = threading.Thread(target=self._run_worker, daemon=True)
        thread.start()

    def _run_worker(self):
        """Background thread — calls main(), streams stdout to log widget."""
        old_stdout, old_stderr = sys.stdout, sys.stderr
        redirector = _RedirectText(self._log_box)
        sys.stdout = redirector
        sys.stderr = redirector

        report_path: Path | None = None
        out_dir: Path | None = None

        try:
            # Import here (inside thread) so any import-time prints are captured
            from main import main as run_optimizer, _base_dir, _setup_outputs

            base = _base_dir(None)
            out_dir = _setup_outputs(base)

            containers = run_optimizer(
                excel_path=self._excel_path,
                no_plot=True,          # GUI never shows plots inline
                base_dir=str(base),
            )

            # Find report
            candidate = out_dir / "report.xlsx"
            if candidate.exists():
                report_path = candidate

            self.after(0, self._on_success, out_dir, report_path)

        except Exception as exc:
            tb = traceback.format_exc()
            sys.stdout.write(f"\n\n{'='*60}\nERROR: {exc}\n{tb}\n")
            self.after(0, self._on_error, str(exc))

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _on_success(self, out_dir: Path, report_path: Path | None):
        self._running = False
        self._progress.stop()
        self._status_var.set("Completed successfully")
        self._out_dir = out_dir
        self._report_path = report_path

        self._set_btn_enabled(self._run_btn, True)
        if report_path:
            self._set_btn_enabled(self._report_btn, True)
        self._set_btn_enabled(self._folder_btn, True)

        self._log_append(f"\n{'='*60}\nDone! Outputs saved to: {out_dir}\n")

        if report_path:
            _open_file(report_path)

    def _on_error(self, msg: str):
        self._running = False
        self._progress.stop()
        self._status_var.set("Error — see log")
        self._set_btn_enabled(self._run_btn, True)
        messagebox.showerror("Optimizer Error", f"Run failed:\n\n{msg}\n\nSee the log for details.")

    def _open_report(self):
        if self._report_path and self._report_path.exists():
            _open_file(self._report_path)
        else:
            messagebox.showinfo("No report", "Report file not found.")

    def _open_folder(self):
        if self._out_dir and self._out_dir.exists():
            _open_file(self._out_dir)
        else:
            messagebox.showinfo("No folder", "Output folder not found.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log_append(self, text: str):
        self._log_box.configure(state="normal")
        self._log_box.insert(tk.END, text)
        self._log_box.see(tk.END)
        self._log_box.configure(state="disabled")

    def _center_window(self, w: int, h: int):
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")


# ── Styled button helper ───────────────────────────────────────────────────────
class _Button(tk.Button):
    def __init__(self, parent, text, command, accent=False,
                 state="normal", width=None):
        self._accent = accent
        self._enabled = (state == "normal")
        self._command = command

        bg, fg = (_ACCENT, "#FFFFFF") if accent else (_BTN, _TEXT)

        kwargs = dict(
            text=text,
            command=self._on_click,          # always wired, we gate internally
            bg=bg, fg=fg,
            activebackground=_ACCENT_H if accent else _BTN_H,
            activeforeground="#FFFFFF" if accent else _TEXT,
            relief="flat",
            cursor="hand2",
            font=_UI_FONT,
            padx=14, pady=8,
            bd=0,
            highlightthickness=1,
            highlightbackground=_BORDER,
            highlightcolor=_BORDER,
        )
        if width:
            kwargs["width"] = width

        super().__init__(parent, **kwargs)

        self.bind("<Enter>", lambda _: self._on_enter())
        self.bind("<Leave>", lambda _: self._on_leave())

        if not self._enabled:
            self._apply_disabled_style()

    def _on_click(self):
        if self._enabled:
            self._command()

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        if enabled:
            self.configure(
                cursor="hand2",
                bg=_ACCENT if self._accent else _BTN,
                fg="#FFFFFF" if self._accent else _TEXT,
                bd=0,
                relief="flat",
                highlightthickness=1,
                highlightbackground=_BORDER,
            )
        else:
            self._apply_disabled_style()

    def _apply_disabled_style(self):
        self.configure(
            cursor="arrow",
            bg="#1E2D42",      # clearly distinct from _BG on all platforms
            fg="#4A5A6B",
            bd=1,
            relief="solid",
            highlightthickness=0,
        )

    def _on_enter(self):
        if not self._enabled:
            return
        self.configure(bg=_ACCENT_H if self._accent else _BTN_H)

    def _on_leave(self):
        if not self._enabled:
            return
        self.configure(bg=_ACCENT if self._accent else _BTN)
# ── Cross-platform file/folder open ───────────────────────────────────────────

def _open_file(path: Path):
    p = str(path)
    if sys.platform == "darwin":
        subprocess.Popen(["open", p])
    elif sys.platform == "win32":
        os.startfile(p)
    else:
        subprocess.Popen(["xdg-open", p])


# ── OR-Tools startup check ────────────────────────────────────────────────────

def _check_ortools():
    """
    Verify OR-Tools native extension loads before the main window opens.
    On Windows, the most common failure is a missing Visual C++ runtime.
    Show a clear, actionable messagebox rather than a cryptic log error.
    """
    try:
        import importlib
        importlib.import_module("ortools.sat.python.cp_model")
    except (ImportError, OSError) as exc:
        root = tk.Tk()
        root.withdraw()
        tk.messagebox.showerror(
            "Startup error — OR-Tools failed to load",
            f"OR-Tools could not be loaded:\n\n{exc}\n\n"
            "On Windows this is almost always caused by a missing\n"
            "Microsoft Visual C++ runtime.\n\n"
            "One-time fix (free, ~25 MB):\n"
            "  https://aka.ms/vs/17/release/vc_redist.x64.exe\n\n"
            "Download, install, then restart this application.",
        )
        raise SystemExit(1)


# ── Entry point ───────────────────────────────────────────────────────────────

def run_gui():
    _check_ortools()
    app = App()
    app.mainloop()


if __name__ == "__main__":
    run_gui()
