# Nissan Leaf Module Cell Datalog Visualizer

High-performance, interactive Python visualization tool for Nissan Leaf module battery cells (Cell 1, 2, 3, 4) using **Python `uv`** and **PyQtGraph / PyQt6**.

---

## ⚡ Quick Start

### 1. Launch the Interactive PyQt GUI:
```bash
uv run plot_cells.py
```
*(Or double-click `run_plot.bat` on Windows)*

### 2. Auto-Export High-Resolution Charts (Offscreen PNGs):
```bash
uv run plot_cells.py --export
```

No manual dependency installation is required; `uv` handles isolated environment setup and all dependencies automatically via PEP 723 inline script metadata.

---

## 📊 Visualized Signals & Features

| Signal | Description | Source Column Variants Handled |
| :--- | :--- | :--- |
| **Time** | Elapsed time (Hours / Mins / Secs) or Datetime | `Date`, `Time` (0.5s sampling rate) |
| **Udc** | Terminal Voltage (V) | `Udc1` (Cell 1) / `Vdc1` (Cells 2, 3, 4) |
| **Idc** | Discharge Current (A) | `Idc1` (Cell 1 is -20A, Cells 2-4 +20A) |
| **IH** | Integrated Capacity (Ah) | `IH1` (Cell 1) / `Ah` (Cells 2, 3, 4) |

### Interactive GUI Capabilities:
1. **Synchronized 3-Plot Stack**: Linked X-axes for Udc, Idc, and IH. Panning and zooming one chart automatically syncs the others.
2. **Interactive Crosshair & Hover Readout**: Real-time cursor coordinates and interpolated values for all cells in the status bar.
3. **Discharge Curve View (Udc vs IH)**: Direct comparison of cell voltage degradation vs discharged capacity (Ah).
4. **Individual Cell Tabs**: Detailed inspection tabs for each cell.
5. **Summary Table**: Side-by-side comparison of initial voltage, cutoff voltage, active discharge current, total capacity, and energy (Wh).
6. **Sign Normalization**: Option to toggle between discharge magnitude (`|Idc|`, `|IH|`) and raw logged signs.
7. **1-Click High-Res Export**: Save active or all plots to PNG at 1920px resolution.

