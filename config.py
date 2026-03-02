"""config.py — Single source of truth for all tunable constants.

Priority order for configuration:
  1. optimizer_config.json placed NEXT TO the executable (user-editable, no rebuild needed)
  2. Built-in default values below  →  a WARNING is printed at startup

To customise: copy optimizer_config.json next to the .exe / .app and edit the values.
"""

import json
import sys
from pathlib import Path

# ── Locate user config file ───────────────────────────────────────────────────
if getattr(sys, "frozen", False):
    # Running as a PyInstaller bundle
    _CFG_SEARCH_DIR = Path(sys.executable).resolve().parent
else:
    # Running as a Python script — look one level above app/ (project root)
    _CFG_SEARCH_DIR = Path(__file__).resolve().parent.parent

_CONFIG_FILE = _CFG_SEARCH_DIR / "optimizer_config.json"

# ── Built-in defaults (standard 40ft HC container, EU pallet standards) ───────
_DEFAULTS: dict = {
    # Container internal dimensions (cm)
    "CONTAINER_LENGTH_CM":           1203,
    "CONTAINER_WIDTH_CM":            235,
    "CONTAINER_HEIGHT_CM":           270,
    "CONTAINER_DOOR_HEIGHT_CM":      250,
    # Weight limit (kg)
    "CONTAINER_MAX_WEIGHT_KG":       18000,
    # Gap between consecutive pallet row-blocks along the container length (cm)
    "ROW_GAP_CM":                    5,
    # Solver wall-clock time limit per container (seconds)
    "SOLVER_TIME_LIMIT_SEC":         10,
    # Recommendation objective (see recommend.py for valid values)
    "RECOMMEND_OBJECTIVE":           "min_leftover",
    "RECOMMEND_SECONDARY_OBJECTIVE": "min_pallets",
}

# ── Load or fall back ─────────────────────────────────────────────────────────
def _load() -> tuple[dict, str | None, bool]:
    """Returns (cfg_dict, source_path_or_None, using_defaults)."""
    if _CONFIG_FILE.exists():
        try:
            with _CONFIG_FILE.open(encoding="utf-8") as f:
                user = json.load(f)
            # Merge: user values override defaults; unknown keys are ignored
            merged = {k: user.get(k, v) for k, v in _DEFAULTS.items()}
            return merged, str(_CONFIG_FILE), False
        except Exception as exc:
            print(
                f"\n[config] WARNING: optimizer_config.json found but could not be parsed: {exc}\n"
                f"[config] WARNING: Falling back to built-in defaults.\n"
            )
    else:
        print(
            f"\n[config] WARNING: No optimizer_config.json found at:\n"
            f"[config]   {_CONFIG_FILE}\n"
            f"[config] WARNING: Using built-in defaults (standard 40ft HC container).\n"
            f"[config] TIP:  Copy optimizer_config.json next to the executable to customise.\n"
        )
    return dict(_DEFAULTS), None, True


_cfg, _CONFIG_SOURCE, _USING_DEFAULTS = _load()

# ── Export constants ──────────────────────────────────────────────────────────
CONTAINER_LENGTH_CM           = int(_cfg["CONTAINER_LENGTH_CM"])
CONTAINER_WIDTH_CM            = int(_cfg["CONTAINER_WIDTH_CM"])
CONTAINER_HEIGHT_CM           = int(_cfg["CONTAINER_HEIGHT_CM"])
CONTAINER_DOOR_HEIGHT_CM      = int(_cfg["CONTAINER_DOOR_HEIGHT_CM"])
CONTAINER_MAX_WEIGHT_KG       = int(_cfg["CONTAINER_MAX_WEIGHT_KG"])
ROW_GAP_CM                    = int(_cfg["ROW_GAP_CM"])
SOLVER_TIME_LIMIT_SEC         = int(_cfg["SOLVER_TIME_LIMIT_SEC"])
RECOMMEND_OBJECTIVE           = str(_cfg["RECOMMEND_OBJECTIVE"])
RECOMMEND_SECONDARY_OBJECTIVE = str(_cfg["RECOMMEND_SECONDARY_OBJECTIVE"])
