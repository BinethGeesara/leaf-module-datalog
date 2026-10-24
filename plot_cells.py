# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyqtgraph>=0.13.0",
#     "PyQt6>=6.4.0",
#     "pandas>=2.0.0",
#     "numpy>=1.24.0",
# ]
# ///
"""
Interactive PyQtGraph Data Visualizer for Nissan Leaf Module Cells (1, 2, 3, 4).
Plots Time, Udc (Voltage), Idc (Current), and IH (Capacity Ah) with linked axes,
interactive crosshair, multi-cell overlays, discharge curves, and PNG export.

Run using uv:
    uv run plot_cells.py
Or to automatically export high-res PNG plots without GUI:
    uv run plot_cells.py --export
"""

import sys
import os
import glob
import argparse
import pandas as pd
import numpy as np

# Ensure PyQtGraph uses PyQt6
os.environ["PYQTGRAPH_QT_LIB"] = "PyQt6"

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QCheckBox, QComboBox, QPushButton, QLabel, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox,
    QStatusBar, QFrame
)
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon

import pyqtgraph as pg
import pyqtgraph.exporters

# ---------------------------------------------------------
# Style & Color Configuration
# ---------------------------------------------------------
CELL_CONFIG = {
    "Cell 1": {
        "file_hint": "*Cell 1*.csv",
        "color": "#00E5FF",      # Vibrant Cyan
        "pen_width": 2,
        "name": "Cell 1",
    },
    "Cell 2": {
        "file_hint": "*Cell 2*.csv",
        "color": "#FFAB00",      # Vibrant Amber / Gold
        "pen_width": 2,
        "name": "Cell 2",
    },
    "Cell 3": {
        "file_hint": "*Cell 3*.csv",
        "color": "#00E676",      # Vibrant Neon Green
        "pen_width": 2,
        "name": "Cell 3",
    },
    "Cell 4": {
        "file_hint": "*Cell 4*.csv",
        "color": "#FF4081",      # Vibrant Pink / Magenta
        "pen_width": 2,
        "name": "Cell 4",
    },
}

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def locate_cell_files(base_dir):
    """Find the exact 'Copy of ...' CSV files or fall back to any matching cell file."""
    preferred_names = {
        "Cell 1": "Copy of 10210000 _Cell 1.csv",
        "Cell 2": "Copy of 10210000_Cell 2.csv",
        "Cell 3": "Copy of 10220000_Cell 3.csv",
        "Cell 4": "Copy of 10220000_Cell 4.csv",
    }
    resolved = {}
    for cell_key, filename in preferred_names.items():
        direct_path = os.path.join(base_dir, filename)
        if os.path.exists(direct_path):
            resolved[cell_key] = direct_path
        else:
            # Fallback search
            hint = CELL_CONFIG[cell_key]["file_hint"]
            matches = glob.glob(os.path.join(base_dir, hint))
            if matches:
                # prefer matches containing 'copy' if available
                copy_matches = [m for m in matches if "copy" in os.path.basename(m).lower()]
                resolved[cell_key] = copy_matches[0] if copy_matches else matches[0]
            else:
                resolved[cell_key] = None
    return resolved


def load_cell_data(file_path, cell_name):
    """
    Load and clean cell data from CSV.
    Detects column variants (e.g. Udc1 vs Vdc1, IH1 vs Ah).
    Computes elapsed time in seconds, minutes, and hours at 0.5s sampling rate.
    """
    if not file_path or not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path)

    # Detect Voltage column
    v_col = next((c for c in ['Udc1', 'Vdc1', 'Udc', 'Vdc', 'Voltage'] if c in df.columns), None)
    # Detect Current column
    i_col = next((c for c in ['Idc1', 'Idc', 'Current'] if c in df.columns), None)
    # Detect Capacity / IH column
    ah_col = next((c for c in ['IH1', 'Ah', 'IH', 'Capacity'] if c in df.columns), None)

    if not v_col or not i_col or not ah_col:
        raise ValueError(f"Could not identify required columns in {file_path}. Found: {df.columns.tolist()}")

    # Sampling interval: 0.5 seconds per record
    n_points = len(df)
    elapsed_seconds = np.arange(n_points, dtype=float) * 0.5
    elapsed_minutes = elapsed_seconds / 60.0
    elapsed_hours = elapsed_seconds / 3600.0

    # Parse timestamps
    try:
        start_dt = pd.to_datetime(df['Date'].iloc[0] + ' ' + df['Time'].iloc[0], format='%d/%m/%Y %H:%M:%S')
        timestamps = start_dt + pd.to_timedelta(elapsed_seconds, unit='s')
    except Exception:
        timestamps = pd.date_range(start='2026-10-21 00:00:00', periods=n_points, freq='500ms')

    raw_udc = df[v_col].to_numpy(dtype=float)
    raw_idc = df[i_col].to_numpy(dtype=float)
    raw_ih = df[ah_col].to_numpy(dtype=float)

    # Discharge magnitude (Cell 1 current and Ah were logged as negative, Cell 2-4 positive)
    mag_idc = np.abs(raw_idc)
    # For IH, ensure monotonically increasing discharged capacity
    mag_ih = np.abs(raw_ih)

    # Calculate Energy (Wh) if present or integrate
    wh_col = next((c for c in ['WP1', 'Wh', 'Energy'] if c in df.columns), None)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if wh_col:
        energy_wh = np.abs(df[wh_col].to_numpy(dtype=float))
        final_energy = float(energy_wh[-1])
    elif trapz_fn is not None:
        final_energy = float(trapz_fn(raw_udc * mag_idc, elapsed_hours))
        energy_wh = None
    else:
        final_energy = float(np.sum(raw_udc * mag_idc * 0.5) / 3600.0)
        energy_wh = None

    stats = {
        "cell": cell_name,
        "file": os.path.basename(file_path),
        "count": n_points,
        "start_time": str(timestamps[0]),
        "end_time": str(timestamps[-1]),
        "duration_hrs": elapsed_hours[-1],
        "v_start": raw_udc[0],
        "v_end": raw_udc[-1],
        "v_min": np.min(raw_udc),
        "v_max": np.max(raw_udc),
        "i_active_mean": float(np.mean(mag_idc[mag_idc > 1.0])) if np.any(mag_idc > 1.0) else float(np.mean(mag_idc)),
        "capacity_ah": float(mag_ih[-1]),
        "energy_wh": final_energy,
    }

    return {
        "cell": cell_name,
        "file_path": file_path,
        "stats": stats,
        "n_points": n_points,
        "elapsed_sec": elapsed_seconds,
        "elapsed_min": elapsed_minutes,
        "elapsed_hr": elapsed_hours,
        "timestamps": timestamps,
        "udc": raw_udc,
        "raw_idc": raw_idc,
        "mag_idc": mag_idc,
        "raw_ih": raw_ih,
        "mag_ih": mag_ih,
        "energy_wh": energy_wh,
    }


class LeafModuleVisualizer(QMainWindow):
    def __init__(self, data_dict, parent=None):
        super().__init__(parent)
        self.data_dict = data_dict
        self.setWindowTitle("Nissan Leaf Module Datalog Visualizer — Cell 1, 2, 3, 4 (PyQtGraph)")
        self.resize(1380, 880)

        # Apply Modern Dark Scientific Theme
        self.apply_theme()

        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 8, 10, 8)
        main_layout.setSpacing(6)

        # Top Control Bar
        control_bar = self.create_control_bar()
        main_layout.addWidget(control_bar)

        # Tabs for different plot views
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #374151;
                background-color: #111827;
                border-radius: 6px;
            }
            QTabBar::tab {
                background: #1F2937;
                color: #9CA3AF;
                padding: 8px 18px;
                margin-right: 4px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QTabBar::tab:selected {
                background: #374151;
                color: #00E5FF;
                border-bottom: 2px solid #00E5FF;
            }
            QTabBar::tab:hover {
                background: #2D3748;
                color: #FFFFFF;
            }
        """)
        main_layout.addWidget(self.tabs, stretch=1)

        # Initialize Tabs
        self.init_overview_tab()
        self.init_discharge_curve_tab()
        self.init_individual_tabs()
        self.init_stats_tab()

        # Status Bar with interactive coordinate readout
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("background-color: #1F2937; color: #E5E7EB; font-family: monospace; font-size: 12px;")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Hover over curves to inspect Time, Udc, Idc, and IH.")

        # Update initial plots
        self.update_plots()

    def apply_theme(self):
        pg.setConfigOption('background', '#111827')
        pg.setConfigOption('foreground', '#E5E7EB')
        pg.setConfigOption('antialias', True)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #0F172A;
            }
            QLabel {
                color: #F3F4F6;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #374151;
                border-radius: 6px;
                margin-top: 6px;
                font-weight: bold;
                color: #60A5FA;
                padding: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QCheckBox {
                color: #F3F4F6;
                font-size: 13px;
                font-weight: bold;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #4B5563;
                background: #1F2937;
            }
            QCheckBox::indicator:checked {
                background: #00E5FF;
                border-color: #00E5FF;
            }
            QComboBox {
                background-color: #1F2937;
                color: #F3F4F6;
                border: 1px solid #4B5563;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 12px;
                min-width: 140px;
            }
            QComboBox::drop-down {
                border: 0px;
            }
            QPushButton {
                background-color: #2563EB;
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 6px 14px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1D4ED8;
            }
            QPushButton:pressed {
                background-color: #1E40AF;
            }
            QTableWidget {
                background-color: #111827;
                color: #F3F4F6;
                gridline-color: #374151;
                border: 1px solid #374151;
                border-radius: 6px;
                font-size: 13px;
            }
            QHeaderView::section {
                background-color: #1F2937;
                color: #93C5FD;
                padding: 6px;
                font-weight: bold;
                border: 1px solid #374151;
            }
        """)

    def create_control_bar(self):
        bar = QFrame()
        bar.setStyleSheet("background-color: #1E293B; border-radius: 8px; padding: 4px;")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(14)

        # Title Label
        title = QLabel("⚡ NISSAN LEAF MODULE DATALOG")
        title.setStyleSheet("font-weight: 800; font-size: 15px; color: #38BDF8; letter-spacing: 0.5px;")
        layout.addWidget(title)

        layout.addSpacing(10)

        # Cell Selection Checkboxes
        self.cell_checkboxes = {}
        cells_group = QHBoxLayout()
        cells_group.setSpacing(10)
        for cell_key, cfg in CELL_CONFIG.items():
            cb = QCheckBox(cell_key)
            cb.setChecked(True)
            cb.setStyleSheet(f"QCheckBox {{ color: {cfg['color']}; }}")
            cb.toggled.connect(self.update_plots)
            self.cell_checkboxes[cell_key] = cb
            cells_group.addWidget(cb)
        layout.addLayout(cells_group)

        layout.addSpacing(10)

        # Time Axis Mode Selector
        layout.addWidget(QLabel("X-Axis:"))
        self.time_combo = QComboBox()
        self.time_combo.addItems([
            "Elapsed Time (Hours)",
            "Elapsed Time (Minutes)",
            "Elapsed Time (Seconds)",
            "Data Points Index"
        ])
        self.time_combo.currentIndexChanged.connect(self.update_plots)
        layout.addWidget(self.time_combo)

        # Current & Capacity Sign Selector
        layout.addWidget(QLabel("Sign Mode:"))
        self.sign_combo = QComboBox()
        self.sign_combo.addItems([
            "Discharge Magnitude (|Idc|, |IH|)",
            "Raw Logged Sign"
        ])
        self.sign_combo.currentIndexChanged.connect(self.update_plots)
        layout.addWidget(self.sign_combo)

        layout.addStretch(1)

        # Export Buttons
        btn_export = QPushButton("📷 Save High-Res PNG")
        btn_export.clicked.connect(self.export_current_view)
        layout.addWidget(btn_export)

        btn_export_all = QPushButton("💾 Export All Charts")
        btn_export_all.setStyleSheet("background-color: #059669;")
        btn_export_all.clicked.connect(self.export_all_plots)
        layout.addWidget(btn_export_all)

        return bar

    # ---------------------------------------------------------
    # TAB 1: Multi-Cell Overview (Synchronized 3-Plot Stack)
    # ---------------------------------------------------------
    def init_overview_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)

        # PyQtGraph GraphicsLayoutWidget
        self.overview_glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.overview_glw)

        # 3 Linked Subplots: Voltage, Current, Capacity
        self.plot_v = self.overview_glw.addPlot(row=0, col=0, title="<b style='color:#38BDF8; font-size:13pt;'>Cell Terminal Voltage: Udc (V)</b>")
        self.plot_i = self.overview_glw.addPlot(row=1, col=0, title="<b style='color:#38BDF8; font-size:13pt;'>Discharge Current: Idc (A)</b>")
        self.plot_ah = self.overview_glw.addPlot(row=2, col=0, title="<b style='color:#38BDF8; font-size:13pt;'>Integrated Capacity: IH (Ah)</b>")

        # Link X Axes so zooming/panning any subplot synchronizes all 3
        self.plot_i.setXLink(self.plot_v)
        self.plot_ah.setXLink(self.plot_v)

        for p in [self.plot_v, self.plot_i, self.plot_ah]:
            p.showGrid(x=True, y=True, alpha=0.35)
            p.addLegend(offset=(10, 10), brush=(25, 30, 45, 200), labelTextColor="#FFFFFF")

        self.plot_v.setLabel('left', 'Voltage', units='V')
        self.plot_i.setLabel('left', 'Current', units='A')
        self.plot_ah.setLabel('left', 'Capacity', units='Ah')
        self.plot_ah.setLabel('bottom', 'Elapsed Time', units='hr')

        # Add interactive crosshairs
        self.setup_crosshairs()

        self.tabs.addTab(tab, "📊 All Cells Overview (Time, Udc, Idc, IH)")

    def setup_crosshairs(self):
        self.vLine_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#9CA3AF', width=1, style=Qt.PenStyle.DashLine))
        self.vLine_i = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#9CA3AF', width=1, style=Qt.PenStyle.DashLine))
        self.vLine_ah = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#9CA3AF', width=1, style=Qt.PenStyle.DashLine))

        self.plot_v.addItem(self.vLine_v, ignoreBounds=True)
        self.plot_i.addItem(self.vLine_i, ignoreBounds=True)
        self.plot_ah.addItem(self.vLine_ah, ignoreBounds=True)

        # Mouse hover signal
        self.overview_glw.scene().sigMouseMoved.connect(self.on_mouse_moved)

    def on_mouse_moved(self, evt):
        pos = evt
        active_plot = None
        if self.plot_v.sceneBoundingRect().contains(pos):
            active_plot = self.plot_v
        elif self.plot_i.sceneBoundingRect().contains(pos):
            active_plot = self.plot_i
        elif self.plot_ah.sceneBoundingRect().contains(pos):
            active_plot = self.plot_ah

        if active_plot is not None:
            mouse_point = active_plot.vb.mapSceneToView(pos)
            x_val = mouse_point.x()
            self.vLine_v.setPos(x_val)
            self.vLine_i.setPos(x_val)
            self.vLine_ah.setPos(x_val)

            # Build status bar readout
            time_label = self.time_combo.currentText()
            status_text = f"[{time_label}: {x_val:.3f}]  |  "

            readouts = []
            for cell_key, data in self.data_dict.items():
                if not self.cell_checkboxes[cell_key].isChecked():
                    continue
                x_arr = self.get_x_data(data)
                idx = np.searchsorted(x_arr, x_val)
                idx = np.clip(idx, 0, len(x_arr) - 1)

                v = data["udc"][idx]
                i = data["mag_idc"][idx] if "Magnitude" in self.sign_combo.currentText() else data["raw_idc"][idx]
                ah = data["mag_ih"][idx] if "Magnitude" in self.sign_combo.currentText() else data["raw_ih"][idx]

                readouts.append(f"{cell_key}: Udc={v:.3f}V, Idc={i:.2f}A, IH={ah:.2f}Ah")

            status_text += "   •   ".join(readouts)
            self.status_bar.showMessage(status_text)

    # ---------------------------------------------------------
    # TAB 2: Discharge Curves (Udc vs IH)
    # ---------------------------------------------------------
    def init_discharge_curve_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)

        self.dc_glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.dc_glw)

        self.plot_dc = self.dc_glw.addPlot(title="<b style='color:#38BDF8; font-size:14pt;'>Battery Discharge Characteristics: Voltage (Udc) vs Capacity (IH)</b>")
        self.plot_dc.showGrid(x=True, y=True, alpha=0.35)
        self.plot_dc.setLabel('left', 'Terminal Voltage (Udc)', units='V')
        self.plot_dc.setLabel('bottom', 'Discharged Capacity (IH)', units='Ah')
        self.plot_dc.addLegend(offset=(20, 20), brush=(25, 30, 45, 200), labelTextColor="#FFFFFF")

        self.tabs.addTab(tab, "📈 Discharge Curve (Udc vs IH)")

    # ---------------------------------------------------------
    # TAB 3: Individual Cell Tabs
    # ---------------------------------------------------------
    def init_individual_tabs(self):
        self.indiv_plots = {}
        for cell_key in ["Cell 1", "Cell 2", "Cell 3", "Cell 4"]:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            layout.setContentsMargins(6, 6, 6, 6)

            glw = pg.GraphicsLayoutWidget()
            layout.addWidget(glw)

            p_v = glw.addPlot(row=0, col=0, title=f"<b style='color:{CELL_CONFIG[cell_key]['color']}; font-size:12pt;'>{cell_key} — Terminal Voltage Udc (V)</b>")
            p_i = glw.addPlot(row=1, col=0, title=f"<b style='color:{CELL_CONFIG[cell_key]['color']}; font-size:12pt;'>{cell_key} — Discharge Current Idc (A)</b>")
            p_ah = glw.addPlot(row=2, col=0, title=f"<b style='color:{CELL_CONFIG[cell_key]['color']}; font-size:12pt;'>{cell_key} — Discharged Capacity IH (Ah)</b>")

            p_i.setXLink(p_v)
            p_ah.setXLink(p_v)

            for p in [p_v, p_i, p_ah]:
                p.showGrid(x=True, y=True, alpha=0.35)

            p_v.setLabel('left', 'Udc', units='V')
            p_i.setLabel('left', 'Idc', units='A')
            p_ah.setLabel('left', 'IH', units='Ah')
            p_ah.setLabel('bottom', 'Elapsed Time', units='hr')

            self.indiv_plots[cell_key] = {"glw": glw, "v": p_v, "i": p_i, "ah": p_ah}
            self.tabs.addTab(tab, f"🔍 {cell_key}")

    # ---------------------------------------------------------
    # TAB 4: Summary & Metrics Table
    # ---------------------------------------------------------
    def init_stats_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)

        header = QLabel("Leaf Module Cells — Performance & Datalog Summary")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #38BDF8; margin-bottom: 8px;")
        layout.addWidget(header)

        self.table = QTableWidget()
        columns = [
            "Cell ID", "Source File", "Data Points", "Start Time", "End Time",
            "Duration (hrs)", "Start V", "End V", "Min V", "Avg Current (A)",
            "Capacity (Ah)", "Energy (Wh)"
        ]
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setRowCount(len(self.data_dict))

        for row, (cell_key, data) in enumerate(self.data_dict.items()):
            s = data["stats"]
            vals = [
                cell_key,
                s["file"],
                f"{s['count']:,}",
                s["start_time"].split()[1] if ' ' in s["start_time"] else s["start_time"],
                s["end_time"].split()[1] if ' ' in s["end_time"] else s["end_time"],
                f"{s['duration_hrs']:.2f}",
                f"{s['v_start']:.3f} V",
                f"{s['v_end']:.3f} V",
                f"{s['v_min']:.3f} V",
                f"{s['i_active_mean']:.2f} A",
                f"{s['capacity_ah']:.2f} Ah",
                f"{s['energy_wh']:.1f} Wh",
            ]
            for col, val in enumerate(vals):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == 0:
                    item.setForeground(QColor(CELL_CONFIG[cell_key]["color"]))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.table.setItem(row, col, item)

        layout.addWidget(self.table)
        self.tabs.addTab(tab, "📋 Summary Table")

    # ---------------------------------------------------------
    # Helper to retrieve selected X axis data
    # ---------------------------------------------------------
    def get_x_data(self, data):
        choice = self.time_combo.currentText()
        if "Hours" in choice:
            return data["elapsed_hr"]
        elif "Minutes" in choice:
            return data["elapsed_min"]
        elif "Seconds" in choice:
            return data["elapsed_sec"]
        else:
            return np.arange(data["n_points"])

    def get_x_unit(self):
        choice = self.time_combo.currentText()
        if "Hours" in choice:
            return "hr"
        elif "Minutes" in choice:
            return "min"
        elif "Seconds" in choice:
            return "s"
        else:
            return "points"

    # ---------------------------------------------------------
    # Redraw Plots
    # ---------------------------------------------------------
    def update_plots(self):
        # 1. Clear existing curves in Overview & DC plots
        self.plot_v.clear()
        self.plot_i.clear()
        self.plot_ah.clear()
        self.plot_dc.clear()

        # Re-add crosshair lines to overview
        self.plot_v.addItem(self.vLine_v, ignoreBounds=True)
        self.plot_i.addItem(self.vLine_i, ignoreBounds=True)
        self.plot_ah.addItem(self.vLine_ah, ignoreBounds=True)

        use_magnitude = "Magnitude" in self.sign_combo.currentText()
        x_unit = self.get_x_unit()

        self.plot_ah.setLabel('bottom', 'Elapsed Time', units=x_unit)

        for cell_key, data in self.data_dict.items():
            if data is None:
                continue

            cfg = CELL_CONFIG[cell_key]
            is_visible = self.cell_checkboxes[cell_key].isChecked()

            x = self.get_x_data(data)
            y_v = data["udc"]
            y_i = data["mag_idc"] if use_magnitude else data["raw_idc"]
            y_ah = data["mag_ih"] if use_magnitude else data["raw_ih"]

            pen = pg.mkPen(color=cfg["color"], width=cfg["pen_width"])

            # Overview Plot curves
            if is_visible:
                self.plot_v.plot(x, y_v, pen=pen, name=cell_key)
                self.plot_i.plot(x, y_i, pen=pen, name=cell_key)
                self.plot_ah.plot(x, y_ah, pen=pen, name=cell_key)

                # Discharge Curve Plot (Voltage vs Ah)
                self.plot_dc.plot(data["mag_ih"], y_v, pen=pen, name=f"{cell_key} ({data['stats']['capacity_ah']:.1f} Ah)")

            # Individual Cell Plots
            if cell_key in self.indiv_plots:
                ip = self.indiv_plots[cell_key]
                ip["v"].clear()
                ip["i"].clear()
                ip["ah"].clear()
                ip["v"].plot(x, y_v, pen=pen)
                ip["i"].plot(x, y_i, pen=pen)
                ip["ah"].plot(x, y_ah, pen=pen)
                ip["ah"].setLabel('bottom', 'Elapsed Time', units=x_unit)

    # ---------------------------------------------------------
    # PNG Export Functionality
    # ---------------------------------------------------------
    def export_current_view(self):
        curr_idx = self.tabs.currentIndex()
        tab_name = self.tabs.tabText(curr_idx).replace(" ", "_").replace("📊_", "").replace("📈_", "").replace("🔍_", "")
        default_filename = f"leaf_{tab_name.lower()}.png"

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save High-Resolution Plot", os.path.join(DATA_DIR, default_filename), "PNG Images (*.png)"
        )
        if filepath:
            self.export_widget_to_image(self.tabs.currentWidget(), filepath)
            QMessageBox.information(self, "Export Successful", f"Saved chart to:\n{filepath}")

    def export_widget_to_image(self, widget, filepath):
        # Locate GraphicsLayoutWidget inside widget
        glw = widget.findChild(pg.GraphicsLayoutWidget)
        if glw:
            exporter = pg.exporters.ImageExporter(glw.scene())
            exporter.parameters()['width'] = 1920  # High-definition width
            exporter.export(filepath)
        else:
            # Fallback for table or generic widget
            pixmap = widget.grab()
            pixmap.save(filepath, "PNG")

    def export_all_plots(self):
        export_folder = DATA_DIR
        out_files = [
            os.path.join(export_folder, "leaf_all_cells_overview.png"),
            os.path.join(export_folder, "leaf_discharge_curves_udc_vs_ih.png")
        ]
        exporter1 = pg.exporters.ImageExporter(self.overview_glw.scene())
        exporter1.parameters()['width'] = 1920
        exporter1.export(out_files[0])

        exporter2 = pg.exporters.ImageExporter(self.dc_glw.scene())
        exporter2.parameters()['width'] = 1920
        exporter2.export(out_files[1])

        # Also individual cell charts
        for cell_key, p_dict in self.indiv_plots.items():
            cell_fname = os.path.join(export_folder, f"leaf_{cell_key.lower().replace(' ', '_')}.png")
            exp = pg.exporters.ImageExporter(p_dict["glw"].scene())
            exp.parameters()['width'] = 1920
            exp.export(cell_fname)
            out_files.append(cell_fname)

        msg = "Successfully exported the following charts:\n\n" + "\n".join([f"• {os.path.basename(f)}" for f in out_files])
        QMessageBox.information(self, "Batch Export Complete", msg)


# ---------------------------------------------------------
# CLI & Standalone Export Mode
# ---------------------------------------------------------
def run_cli_export(data_dict):
    """Generates and exports PNG plots without opening a GUI window."""
    print("Generating high-resolution plots via PyQtGraph offscreen exporter...")
    app = QApplication.instance() or QApplication(["--platform", "offscreen"])

    # 1. Overview Chart
    win1 = pg.GraphicsLayoutWidget()
    win1.resize(1920, 1080)
    pg.setConfigOption('background', '#111827')
    pg.setConfigOption('foreground', '#E5E7EB')

    p_v = win1.addPlot(row=0, col=0, title="<b style='color:#38BDF8; font-size:14pt;'>Terminal Voltage: Udc (V) vs Elapsed Time</b>")
    p_i = win1.addPlot(row=1, col=0, title="<b style='color:#38BDF8; font-size:14pt;'>Discharge Current: Idc (A) vs Elapsed Time</b>")
    p_ah = win1.addPlot(row=2, col=0, title="<b style='color:#38BDF8; font-size:14pt;'>Discharged Capacity: IH (Ah) vs Elapsed Time</b>")
    p_i.setXLink(p_v)
    p_ah.setXLink(p_v)

    for p in [p_v, p_i, p_ah]:
        p.showGrid(x=True, y=True, alpha=0.35)
        p.addLegend(offset=(15, 15), brush=(25, 30, 45, 200), labelTextColor="#FFFFFF")

    p_v.setLabel('left', 'Voltage (Udc)', units='V')
    p_i.setLabel('left', 'Current (Idc)', units='A')
    p_ah.setLabel('left', 'Capacity (IH)', units='Ah')
    p_ah.setLabel('bottom', 'Elapsed Time', units='hr')

    for cell_key, data in data_dict.items():
        cfg = CELL_CONFIG[cell_key]
        pen = pg.mkPen(color=cfg["color"], width=cfg["pen_width"])
        x = data["elapsed_hr"]
        p_v.plot(x, data["udc"], pen=pen, name=cell_key)
        p_i.plot(x, data["mag_idc"], pen=pen, name=cell_key)
        p_ah.plot(x, data["mag_ih"], pen=pen, name=cell_key)

    win1.show()
    app.processEvents()
    out1 = os.path.join(DATA_DIR, "leaf_all_cells_overview.png")
    exp1 = pg.exporters.ImageExporter(win1.scene())
    exp1.parameters()['width'] = 1920
    exp1.export(out1)
    print(f"Saved: {out1}")

    # 2. Discharge Curve (Udc vs IH)
    win2 = pg.GraphicsLayoutWidget()
    win2.resize(1920, 1080)
    p_dc = win2.addPlot(title="<b style='color:#38BDF8; font-size:14pt;'>Discharge Characteristics: Voltage (Udc) vs Capacity (IH)</b>")
    p_dc.showGrid(x=True, y=True, alpha=0.35)
    p_dc.setLabel('left', 'Terminal Voltage (Udc)', units='V')
    p_dc.setLabel('bottom', 'Discharged Capacity (IH)', units='Ah')
    p_dc.addLegend(offset=(20, 20), brush=(25, 30, 45, 200), labelTextColor="#FFFFFF")

    for cell_key, data in data_dict.items():
        cfg = CELL_CONFIG[cell_key]
        pen = pg.mkPen(color=cfg["color"], width=cfg["pen_width"])
        p_dc.plot(data["mag_ih"], data["udc"], pen=pen, name=f"{cell_key} ({data['stats']['capacity_ah']:.1f} Ah)")

    win2.show()
    app.processEvents()
    out2 = os.path.join(DATA_DIR, "leaf_discharge_curves_udc_vs_ih.png")
    exp2 = pg.exporters.ImageExporter(win2.scene())
    exp2.parameters()['width'] = 1920
    exp2.export(out2)
    print(f"Saved: {out2}")
    print("\nAll plots generated successfully!")


def main():
    parser = argparse.ArgumentParser(description="Nissan Leaf Module Cell Plotter (PyQtGraph)")
    parser.add_argument("--export", action="store_true", help="Auto-export PNG charts offscreen without opening GUI")
    args = parser.parse_args()

    # Find files
    resolved_files = locate_cell_files(DATA_DIR)
    print("Found Cell Files:")
    for cell_key, fpath in resolved_files.items():
        print(f"  * {cell_key}: {os.path.basename(fpath) if fpath else 'NOT FOUND'}")

    # Load data
    data_dict = {}
    for cell_key, fpath in resolved_files.items():
        if fpath:
            try:
                data_dict[cell_key] = load_cell_data(fpath, cell_key)
            except Exception as e:
                print(f"Error loading {cell_key} ({fpath}): {e}")

    if not data_dict:
        print("Error: No valid cell CSV files loaded.")
        sys.exit(1)

    if args.export:
        run_cli_export(data_dict)
        return

    # Launch GUI
    app = QApplication(sys.argv)
    window = LeafModuleVisualizer(data_dict)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
