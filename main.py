"""MAIN PIPELINE — 1D ROW-BLOCK CONTAINER PACKING

Demo-friendly runner:
- When frozen (PyInstaller), uses the executable folder as base directory.
- When not frozen, uses this file's folder as base directory.
- Default input: `input.xlsx` next to the executable/script.
- Outputs written to `outputs/` next to the executable/script:
    - summary.txt
    - containers.json
    - run.log

IMPORTANT:
CPMpy tries to import optional solvers (e.g., Gurobi) if installed.
For a stable executable, build in an environment WITHOUT `gurobipy`
(or build with PyInstaller excluding `gurobipy`).
"""

from typing import List, Dict, Any
from pathlib import Path
import argparse
import datetime
import json
import sys

from utils.parse_xlsx import parse_pallet_excel_v3
from utils.oneDbuildblocks import build_row_blocks_from_pallets
from models.A_1D_multi_container_placement_chatGPT import RowBlock1DOrderModel

from utils.visualize_row_blocks import plot_all_row_block_containers_pallets


def _base_dir(user_base_dir: str | None = None) -> Path:
    """Resolve the base directory for inputs/outputs."""
    if user_base_dir:
        return Path(user_base_dir).expanduser().resolve()
    # PyInstaller frozen app/exe
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # Normal python execution
    return Path(__file__).resolve().parent


def _setup_outputs(base: Path) -> Path:
    out = base / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _log(out_dir: Path, msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with (out_dir / "run.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def select_one_variant_per_block(blocks):
    """Keep exactly one variant per physical block_id (currently the shortest length)."""
    best = {}
    for b in blocks:
        bid = b.block_id
        if bid not in best:
            best[bid] = b
        else:
            # choose shorter length variant to increase chance of fitting
            if b.length_cm < best[bid].length_cm:
                best[bid] = b
    # preserve stable ordering by block_id
    return [best[k] for k in sorted(best.keys())]


def main(
    excel_path: str = "input_final.xlsx",
    sheet_name=0,
    L_cm: int = 1203,
    gap_cm: int = 5,
    Wmax_kg: int = 18000,
    Hdoor_cm: int = 250,
    solver: str = "ortools",
    time_limit: int = 10,
    base_dir: str | None = None,
    no_plot: bool = False,
):
    base = _base_dir(base_dir)
    out_dir = _setup_outputs(base)

    excel_p = Path(excel_path)
    if not excel_p.is_absolute():
        excel_p = (base / excel_p).resolve()

    _log(out_dir, f"Base directory: {base}")
    _log(out_dir, f"Excel input: {excel_p}")

    # ------------------------------------------------------------
    # 1) Parse Excel
    # ------------------------------------------------------------
    _log(out_dir, "=== STEP 1: Parsing Excel ===")

    lengths, widths, heights, pallets_data, meta_per_pallet = parse_pallet_excel_v3(
        str(excel_p),
        sheet_name=sheet_name,
        return_per_pallet_meta=True,
    )

    print(f"Parsed {len(meta_per_pallet)} physical pallets")
    print(f"Distinct pallet rows: {len(pallets_data)}")

    # ------------------------------------------------------------
    # 2) Build row-block instances (and validate multiples)
    # ------------------------------------------------------------
    _log(out_dir, "=== STEP 2: Building Row-Blocks ===")

    blocks, recommendations, warnings = build_row_blocks_from_pallets(
        meta_per_pallet,
        Hdoor_cm=Hdoor_cm,
        require_multiples=True,   # HARD requirement
    )



    print("DEBUG pallet heights unique:", sorted(set(pm["height"] for pm in meta_per_pallet))[:20])
    print("DEBUG block heights unique:", sorted(set(b.height_cm for b in blocks))[:20])



    if warnings:
        print("\nWARNINGS during block construction:")
        for w in warnings:
            print(" -", w)

    if recommendations:
        print("\nORDER NOT VALID FOR FULL ROW-BLOCK MODEL")
        print("You need to add pallets to reach valid multiples:\n")
        for k, v in recommendations.items():
            print(f"  {k}: add {v} pallets")
        summary_path = out_dir / "summary.txt"
        with summary_path.open("w", encoding="utf-8") as f:
            f.write("ORDER NOT VALID FOR FULL ROW-BLOCK MODEL\n")
            f.write("Add pallets to reach valid multiples:\n\n")
            for k, v in recommendations.items():
                f.write(f"- {k}: add {v} pallets\n")
        _log(out_dir, f"Wrote summary: {summary_path}")
        print("\nStopping before optimization.")
        return

    print(f"Constructed {len(blocks)} row-block VARIANTS")
    physical_blocks = len(set(b.block_id for b in blocks))
    print(f"Corresponding to {physical_blocks} physical row-blocks")

    # IMPORTANT: current model cannot enforce mutual exclusion across rotation variants.
    # So we keep only one variant per physical block_id.
    blocks = select_one_variant_per_block(blocks)
    print(f"After choosing ONE variant per block_id: {len(blocks)} blocks")

    # ------------------------------------------------------------
    # 3) Multi-container loop
    # ------------------------------------------------------------
    _log(out_dir, "=== STEP 3: Solving Containers ===")

    remaining_blocks = blocks[:]  # copy
    containers: List[Dict[str, Any]] = []
    container_idx = 1

    while remaining_blocks:
        print(f"\n--- Solving container {container_idx} ---")

        # ---- Flatten remaining blocks into model arrays ----
        lens = [b.length_cm for b in remaining_blocks]
        hs   = [b.height_cm for b in remaining_blocks]
        ws   = [b.weight_kg for b in remaining_blocks]
        vals = [b.value for b in remaining_blocks]

        # ---- Build model ----
        model = RowBlock1DOrderModel(
            lengths_cm=lens,
            heights_cm=hs,
            weights_kg=ws,
            values=vals,
            L_cm=L_cm,
            gap_cm=gap_cm,
            Wmax_kg=Wmax_kg,
            Hdoor_cm=Hdoor_cm,
        )

        solved = model.solve(
            solver=solver,
            time_limit=time_limit,
        )

        if not solved:
            raise RuntimeError(f"No feasible solution for container {container_idx}")

        # ----------------------------------------------------
        # 4) Extract solution
        # ----------------------------------------------------
        chosen_variant_indices = model.loaded_indices_in_order()
        chosen_blocks = [remaining_blocks[i - 1] for i in chosen_variant_indices]

        # Physical block IDs used
        used_block_ids = {b.block_id for b in chosen_blocks}

        if len(chosen_blocks) == 0:
            print("\n!! EMPTY CONTAINER SOLUTION RETURNED !!")
            print(f"Remaining blocks: {len(remaining_blocks)}")

            door_ok = [b for b in remaining_blocks if b.height_cm <= Hdoor_cm]
            print(f"Door-OK blocks (height <= {Hdoor_cm}): {len(door_ok)}")

            heights_unique = sorted({b.height_cm for b in remaining_blocks})
            print(f"Remaining heights (unique): {heights_unique[:30]}{'...' if len(heights_unique) > 30 else ''}")

            raise RuntimeError(
                "Solver returned empty selection. Likely no feasible non-empty packing exists "
                "under current constraints (often because no remaining door-allowed blocks)."
            )

        # Reconstruct y-coordinates (back -> door)
        y_cursor = 0
        rows = []
        for b in chosen_blocks:
            rows.append({
                "block_id": b.block_id,
                "block_type": b.block_type_key,
                "length_cm": b.length_cm,
                "height_cm": b.height_cm,
                "weight_kg": b.weight_kg,
                "pallet_count": b.value,
                "y_start_cm": y_cursor,
                "pallets": b.pallets,
            })
            y_cursor += b.length_cm + gap_cm

        used_len = model.usedLen.value()
        leftover = L_cm - used_len

        container_info = {
            "container_index": container_idx,
            "rows": rows,
            "used_length_cm": used_len,
            "leftover_cm": leftover,
            "loaded_value": model.loadedValue.value(),
            "loaded_weight": model.loadedWeight.value(),
        }
        containers.append(container_info)

        # ----------------------------------------------------
        # 5) Print container summary
        # ----------------------------------------------------
        print(f"Loaded blocks: {len(rows)}")
        print(f"Used length: {used_len} / {L_cm} cm")
        print(f"Leftover length: {leftover} cm")
        print(f"Loaded pallets: {model.loadedValue.value()}")
        print(f"Loaded weight: {model.loadedWeight.value()} kg")

        print("\nRow layout (back → door):")
        for r in rows:
            print(
                f"  y={r['y_start_cm']:>4} cm | "
                f"{r['block_type']:>12} | "
                f"L={r['length_cm']:>3} | "
                f"H={r['height_cm']:>3} | "
                f"pallets={r['pallet_count']}"
            )

        # ----------------------------------------------------
        # 6) Remove used physical blocks
        # ----------------------------------------------------
        remaining_blocks = [b for b in remaining_blocks if b.block_id not in used_block_ids]
        container_idx += 1

    # ------------------------------------------------------------
    # 7) Final output
    # ------------------------------------------------------------
    _log(out_dir, "=== ALL CONTAINERS SOLVED ===")
    print(f"Total containers used: {len(containers)}")

    containers_path = out_dir / "containers.json"
    with containers_path.open("w", encoding="utf-8") as f:
        json.dump(containers, f, ensure_ascii=False, indent=2)
    _log(out_dir, f"Wrote containers: {containers_path}")

    summary_path = out_dir / "summary.txt"
    total_pallets = sum(c["loaded_value"] for c in containers)
    total_weight = sum(c["loaded_weight"] for c in containers)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("CONTAINER PACKING SUMMARY\n")
        f.write("========================\n\n")
        f.write(f"Containers used: {len(containers)}\n")
        f.write(f"Total pallets loaded: {total_pallets}\n")
        f.write(f"Total weight loaded (kg): {total_weight}\n\n")
        for c in containers:
            f.write(
                f"Container {c['container_index']}: pallets={c['loaded_value']}, weight={c['loaded_weight']} kg, used_length={c['used_length_cm']} cm, leftover={c['leftover_cm']} cm\n"
            )
    _log(out_dir, f"Wrote summary: {summary_path}")

    # ------------------------------------------------------------
    # 8) Visualization of all containers
    # ------------------------------------------------------------
    # containers = main("sample_instances/input_large.xlsx")
    if not no_plot:
        plot_all_row_block_containers_pallets(containers, W=235, L=1203, H=270)
        # Keep plot windows open when running as a script
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass

    return containers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Container optimizer (double-click friendly)")
    # Default: in frozen executable expect input.xlsx next to the exe; in dev use sample_instances
    default_excel = "input_final.xlsx" if getattr(sys, "frozen", False) else "sample_instances/input_final.xlsx"
    # In dev, show plots by default; in frozen app, disable plots by default
    default_no_plot = True if getattr(sys, "frozen", False) else False
    parser.add_argument(
        "--excel",
        default=default_excel,
        help="Excel file (frozen: input.xlsx next to exe; dev: sample_instances/input.xlsx)",
    )
    parser.add_argument("--sheet", default=0, help="Sheet index or name")
    parser.add_argument("--no_plot", action="store_true", help="Disable plotting")
    parser.add_argument("--base_dir", default=None, help="Base directory for input/output")

    args = parser.parse_args()

    sheet_val = args.sheet
    try:
        sheet_val = int(sheet_val)
    except Exception:
        pass

    main(
        excel_path=args.excel,
        sheet_name=sheet_val,
        base_dir=args.base_dir,
        no_plot=(args.no_plot or default_no_plot),
    )
