"""gui.py — Tkinter desktop front-end for the Container Packing Optimizer.

Usage (development):
    python app/gui.py

Usage (frozen PyInstaller exe/app):
    Double-click ContainerOptimizer.exe / ContainerOptimizer.app

UX flow:
    1. User clicks "Browse…" → file-picker dialog; column picker appears
    2. User confirms order-quantity column (auto-detected from Excel headers)
    3. User clicks "Run Optimizer" → background thread calls main()
    4. Status message shows result; log is collapsible (hidden by default)
    5. On completion: "Open Report" and "Open Output Folder" buttons activate
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
    _BUNDLE = getattr(sys, "_MEIPASS", None)
    if _BUNDLE:
        sys.path.insert(0, str(_BUNDLE))
else:
    _APP_DIR = Path(__file__).resolve().parent.parent  # project root

# ── Colour palette ─────────────────────────────────────────────────────────────
_BG        = "#0B1220"
_PANEL     = "#0F172A"
_PANEL_H   = "#16213A"
_FIELD     = "#0B1020"
_BORDER    = "#22304A"

_BTN       = "#111C33"
_BTN_H     = "#1A2A4A"

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

# Candidates used to auto-select the order-quantity column.
# Must match the candidate list in parse_pallet_excel_v3 exactly so the
# column picker always defaults to the same column the parser will use.
_COUNT_COL_CANDIDATES = [
    "External Packaging Quantity", "external packaging",
    "Total order full pallets", "Total number of pallets",
    "full pallets", "order full pallets", "number of pallets",
]


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
        self.minsize(760, 480)

        # State
        self._excel_path: str = ""
        self._count_col: str | None = None   # user-selected order-qty column
        self._out_dir: Path | None = None
        self._report_path: Path | None = None
        self._running = False
        self._log_visible = False             # log collapsed by default

        self._build_ui()
        self._center_window(900, 640)

    def _set_btn_enabled(self, btn: _Button, enabled: bool):
        btn.set_enabled(enabled)

    # ── TTK styling ───────────────────────────────────────────────────────────
    def _style_ttk(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TProgressbar",
                        troughcolor=_PANEL, background=_ACCENT,
                        bordercolor=_BORDER, lightcolor=_ACCENT, darkcolor=_ACCENT)
        for orient in ("Vertical", "Horizontal"):
            style.configure(f"{orient}.TScrollbar",
                            gripcount=0, background=_PANEL, troughcolor=_BG,
                            bordercolor=_BORDER, arrowcolor=_MUTED)
        style.configure("TCombobox",
                        fieldbackground=_FIELD, background=_PANEL,
                        foreground=_TEXT, selectbackground=_ACCENT,
                        selectforeground="#FFFFFF", arrowcolor=_MUTED)
        style.map("TCombobox",
                  fieldbackground=[("readonly", _FIELD)],
                  foreground=[("readonly", _TEXT)])

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header stripe
        tk.Frame(self, bg=_ACCENT, height=4).pack(fill="x")

        # Title row
        title_fr = tk.Frame(self, bg=_BG, pady=16)
        title_fr.pack(fill="x", padx=24)
        tk.Label(title_fr, text="Container Packing Optimizer",
                 font=_TITLE_FONT, bg=_BG, fg=_TEXT).pack(side="left")
        self._cfg_label = tk.Label(title_fr, text="", font=("Courier New", 10),
                                   bg=_BG, fg=_MUTED, anchor="e", justify="right")
        self._cfg_label.pack(side="right")
        self._update_config_label()

        # ── File selection card ────────────────────────────────────────────
        card = tk.Frame(self, bg=_PANEL, padx=16, pady=12)
        card.pack(fill="x", padx=24, pady=(0, 6))
        card.columnconfigure(0, weight=1)

        tk.Label(card, text="Excel Input File",
                 font=(*_UI_FONT[:1], _UI_FONT[1], "bold"),
                 bg=_PANEL, fg=_TEXT).grid(row=0, column=0, sticky="w")

        self._file_var = tk.StringVar(value="No file selected")
        tk.Label(card, textvariable=self._file_var,
                 font=_MONO_FONT, bg=_PANEL, fg=_MUTED,
                 wraplength=560, anchor="w", justify="left"
                 ).grid(row=1, column=0, sticky="ew", pady=(4, 0))

        btn_fr = tk.Frame(card, bg=_PANEL)
        btn_fr.grid(row=0, column=1, rowspan=2, sticky="e", padx=(12, 0))
        self._browse_btn = _Button(btn_fr, "Browse…", self._on_browse)
        self._browse_btn.pack()

        # ── Column picker (hidden until file selected) ─────────────────────
        self._col_frame = tk.Frame(self, bg=_PANEL, padx=16, pady=8)
        # packed dynamically in _on_browse; not shown at start

        col_inner = tk.Frame(self._col_frame, bg=_PANEL)
        col_inner.pack(fill="x")
        tk.Label(col_inner, text="Order quantity column:",
                 font=_UI_FONT, bg=_PANEL, fg=_TEXT).pack(side="left")

        self._col_var = tk.StringVar()
        self._col_combo = ttk.Combobox(col_inner, textvariable=self._col_var,
                                       state="readonly", width=40,
                                       font=_MONO_FONT)
        self._col_combo.pack(side="left", padx=(8, 0))
        self._col_combo.bind("<<ComboboxSelected>>", self._on_col_selected)

        self._col_hint = tk.Label(self._col_frame, text="", font=(*_UI_FONT[:1], 10),
                                  bg=_PANEL, fg=_MUTED, anchor="w")
        self._col_hint.pack(fill="x", pady=(2, 0))

        # ── Run controls ──────────────────────────────────────────────────
        self._ctrl_frame = tk.Frame(self, bg=_BG)
        ctrl = self._ctrl_frame
        ctrl.pack(fill="x", padx=24, pady=6)

        self._run_btn = _Button(ctrl, "Run Optimizer", self._on_run,
                                accent=True, width=18)
        self._run_btn.pack(side="left")

        self._progress = ttk.Progressbar(ctrl, mode="indeterminate", length=200)
        self._progress.pack(side="left", padx=(12, 0))

        self._status_var = tk.StringVar(value="Ready")
        tk.Label(ctrl, textvariable=self._status_var,
                 bg=_BG, fg=_MUTED, font=_UI_FONT).pack(side="left", padx=8)

        # ── User message label (always visible, color-coded) ──────────────
        self._msg_frame = tk.Frame(self, bg=_BG)
        self._msg_frame.pack(fill="x", padx=24, pady=(0, 4))
        self._msg_label = tk.Label(
            self._msg_frame, text="", font=_UI_FONT,
            bg=_BG, fg=_MUTED,
            wraplength=820, anchor="w", justify="left"
        )
        self._msg_label.pack(fill="x")

        # ── Log toggle bar ────────────────────────────────────────────────
        log_bar = tk.Frame(self, bg=_PANEL, padx=12, pady=6)
        log_bar.pack(fill="x", padx=24, pady=(0, 0))

        self._log_toggle_btn = _Button(log_bar, "▶  Show Log", self._toggle_log)
        self._log_toggle_btn.pack(side="left")

        self._copy_btn = _Button(log_bar, "Copy Log", self._copy_log)
        self._copy_btn.pack(side="left", padx=(8, 0))

        # ── Log panel (hidden by default) ─────────────────────────────────
        self._log_outer = tk.Frame(self, bg=_PANEL, padx=12, pady=6)
        # NOT packed yet — shown on toggle

        txt_fr = tk.Frame(self._log_outer, bg=_PANEL)
        txt_fr.pack(fill="both", expand=True)

        self._log_box = tk.Text(
            txt_fr, state="disabled", wrap="none",
            bg=_FIELD, fg=_TEXT, insertbackground=_TEXT,
            font=_MONO_FONT, relief="flat",
            highlightthickness=1,
            highlightbackground=_BORDER, highlightcolor=_BORDER,
            selectbackground=_ACCENT_H, selectforeground="#FFFFFF",
        )
        v_scroll = ttk.Scrollbar(txt_fr, orient="vertical", command=self._log_box.yview)
        h_scroll = ttk.Scrollbar(txt_fr, orient="horizontal", command=self._log_box.xview)
        self._log_box.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        h_scroll.pack(side="bottom", fill="x")
        v_scroll.pack(side="right", fill="y")
        self._log_box.pack(side="left", fill="both", expand=True)

        # ── Bottom action bar ─────────────────────────────────────────────
        bot = tk.Frame(self, bg=_BG, pady=12)
        bot.pack(fill="x", padx=24, side="bottom")

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
            src = "Built-in defaults" if _USING_DEFAULTS else (_CONFIG_SOURCE or "custom")
            self._cfg_label.config(
                text=f"Config: {src}\nContainer: {CONTAINER_LENGTH_CM}×{CONTAINER_WIDTH_CM}×{CONTAINER_HEIGHT_CM} cm",
                fg=_MUTED,
            )
        except Exception:
            pass

    # ── Log toggle / copy ─────────────────────────────────────────────────────
    def _toggle_log(self):
        self._log_visible = not self._log_visible
        if self._log_visible:
            self._log_outer.pack(fill="both", expand=True, padx=24, pady=(0, 4))
            self._log_toggle_btn.configure(text="▼  Hide Log")
            self._center_window(900, 760)
        else:
            self._log_outer.pack_forget()
            self._log_toggle_btn.configure(text="▶  Show Log")
            self._center_window(900, 640)

    def _copy_log(self):
        content = self._log_box.get("1.0", tk.END)
        if content.strip():
            self.clipboard_clear()
            self.clipboard_append(content)

    # ── Column picker callbacks ───────────────────────────────────────────────
    def _on_col_selected(self, _event=None):
        self._count_col = self._col_var.get() or None

    def _load_excel_columns(self, path: str):
        """Read just the header row from the Excel file to populate the column picker."""
        try:
            import pandas as pd
            # _detect_header_row from parse_xlsx to find the real header
            sys.path.insert(0, str(_APP_DIR / "app") if not getattr(sys, "frozen", False)
                            else getattr(sys, "_MEIPASS", str(_APP_DIR)))
            try:
                from utils.parse_xlsx import _detect_header_row
                hrow = _detect_header_row(path)
            except Exception:
                hrow = 0
            df = pd.read_excel(path, sheet_name=0, header=hrow, nrows=0)
            return list(df.columns)
        except Exception:
            return []

    def _show_column_picker(self, columns: list[str]):
        """Populate and show the column picker card."""
        self._col_combo["values"] = columns

        # Auto-select using the same prefix-matching logic as parse_pallet_excel_v3
        # so the picker always defaults to the column the parser will actually use.
        best = None
        try:
            import pandas as pd
            from utils.parse_xlsx import _find_col_optional
            dummy = pd.DataFrame(columns=columns)
            best = _find_col_optional(dummy, _COUNT_COL_CANDIDATES)
        except Exception:
            pass

        if best:
            self._col_var.set(best)
            self._count_col = best
            self._col_hint.config(text=f"Auto-detected '{best}' — change if incorrect.",
                                  fg=_MUTED)
            self._col_frame.configure(highlightbackground=_BORDER, highlightthickness=0)
        else:
            self._col_var.set("")
            self._count_col = None
            self._col_hint.config(
                text="Could not auto-detect the order quantity column. Please select it above.",
                fg=_YELLOW,
            )

        # Show the card between the file card and the run controls
        self._col_frame.pack(fill="x", padx=24, pady=(0, 6),
                             before=self._ctrl_frame)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _on_browse(self):
        path = filedialog.askopenfilename(
            title="Select Order Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")],
            initialdir=str(_APP_DIR),
        )
        if not path:
            return

        self._excel_path = path
        self._file_var.set(path)
        self._set_user_message("", _MUTED)

        # Load columns and show picker
        columns = self._load_excel_columns(path)
        if columns:
            self._show_column_picker(columns)
        else:
            self._col_frame.pack_forget()
            self._count_col = None

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
        self._set_user_message("", _MUTED)

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
        n_containers: int = 0

        try:
            from main import main as run_optimizer, _base_dir, _setup_outputs

            base = _base_dir(None)
            out_dir = _setup_outputs(base)

            containers = run_optimizer(
                excel_path=self._excel_path,
                no_plot=True,
                base_dir=str(base),
                count_col_override=self._count_col,
            )

            n_containers = len(containers) if containers else 0

            candidate = out_dir / "report.xlsx"
            if candidate.exists():
                report_path = candidate

            self.after(0, self._on_success, out_dir, report_path, n_containers)

        except Exception as exc:
            tb = traceback.format_exc()
            sys.stdout.write(f"\n\n{'='*60}\nERROR: {exc}\n{tb}\n")
            self.after(0, self._on_error, str(exc))

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _on_success(self, out_dir: Path, report_path: Path | None, n_containers: int):
        self._running = False
        self._progress.stop()
        self._status_var.set("Completed successfully")
        self._out_dir = out_dir
        self._report_path = report_path

        self._set_btn_enabled(self._run_btn, True)
        if report_path:
            self._set_btn_enabled(self._report_btn, True)
        self._set_btn_enabled(self._folder_btn, True)

        msg = f"Done — {n_containers} container{'s' if n_containers != 1 else ''} packed."
        if report_path:
            msg += " Report saved and opened."
        self._set_user_message(msg, _GREEN)

        self._log_append(f"\n{'='*60}\nDone! Outputs saved to: {out_dir}\n")

        if report_path:
            _open_file(report_path)

    def _on_error(self, msg: str):
        self._running = False
        self._progress.stop()
        self._status_var.set("Run failed")
        self._set_btn_enabled(self._run_btn, True)

        # Show first 3 lines of the error as the user-facing message
        lines = [l for l in msg.splitlines() if l.strip()]
        short = "\n".join(lines[:3])
        self._set_user_message(f"Error: {short}\n(See log for full details — click ▶ Show Log)", _RED)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _set_user_message(self, text: str, color: str):
        self._msg_label.config(text=text, fg=color)

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
    def __init__(self, parent, text, command, accent=False, state="normal", width=None):
        self._accent = accent
        self._enabled = (state == "normal")
        self._command = command

        bg, fg = (_ACCENT, "#FFFFFF") if accent else (_BTN, _TEXT)
        kwargs = dict(
            text=text, command=self._on_click,
            bg=bg, fg=fg,
            activebackground=_ACCENT_H if accent else _BTN_H,
            activeforeground="#FFFFFF" if accent else _TEXT,
            relief="flat", cursor="hand2", font=_UI_FONT,
            padx=14, pady=8, bd=0,
            highlightthickness=1,
            highlightbackground=_BORDER, highlightcolor=_BORDER,
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
            self.configure(cursor="hand2", bg=_ACCENT if self._accent else _BTN,
                           fg="#FFFFFF" if self._accent else _TEXT,
                           bd=0, relief="flat",
                           highlightthickness=1, highlightbackground=_BORDER)
        else:
            self._apply_disabled_style()

    def _apply_disabled_style(self):
        self.configure(cursor="arrow", bg="#1E2D42", fg="#4A5A6B",
                       bd=1, relief="solid", highlightthickness=0)

    def _on_enter(self):
        if self._enabled:
            self.configure(bg=_ACCENT_H if self._accent else _BTN_H)

    def _on_leave(self):
        if self._enabled:
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
