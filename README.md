# Container Packing Optimizer

A rule-based optimization tool for building realistic container loads from palletized orders.
The model groups pallets into full row-blocks and packs them efficiently across one or more containers
while respecting business and physical constraints (stacking, door height, weight, etc.).

---

## Requirements

- Python **3.10+**
- Conda / Anaconda (recommended)
- OR-Tools solver (via CPMpy)

This project was developed and tested using the **`cpmpy-env`** Conda environment.

---

## Environment Setup

```bash
conda create -n cpmpy-env python=3.10
conda activate cpmpy-env
pip install cpmpy ortools pandas matplotlib
```

