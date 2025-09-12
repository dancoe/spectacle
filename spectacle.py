#!/usr/bin/env python3
"""
FITS Spectra Viewer with PyQt

... (docstrings remain the same) ...

**New Feature: External Annotation Script**

You can add custom annotations (lines, text, etc.) to your plots by providing an
external Python script using the `--annotate` command-line argument.

The script must contain a function with the following signature:
    def annotate_plot(viewer):
        # viewer is the main FITSSpectraViewer application window

Example `annotate_plot.py`:
--------------------------
import pyqtgraph as pg

def annotate_plot(viewer):
    # This function is called after data is loaded.
    # 'viewer' gives you access to the application's plots and data.

    # Example 1: Add a label to the 1D plot
    text = pg.TextItem("My Custom Label", color='blue', anchor=(0, 1))
    viewer.plot_1d.addItem(text)
    text.setPos(1000, 5e-6) # Position in data coordinates (index, flux)

    # Example 2: Add a vertical line to both plots at a specific index
    line_pos_index = 1500
    pen = pg.mkPen('purple', width=2, style=pg.QtCore.Qt.DotLine)
    
    line1d = pg.InfiniteLine(pos=line_pos_index, angle=90, pen=pen)
    viewer.plot_1d.addItem(line1d)

    if viewer.plot_2d.isVisible():
        line2d = pg.InfiniteLine(pos=line_pos_index, angle=90, pen=pen)
        viewer.plot_2d.addItem(line2d)
--------------------------
"""
import sys
import os
import argparse
import numpy as np
import json
import importlib.util
import io

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QLabel,
                             QSplitter, QStatusBar, QToolBar, QAction,
                             QComboBox, QSpinBox, QCheckBox, QGroupBox,
                             QSlider, QMessageBox, QDoubleSpinBox, QGraphicsItem,
                             QColorDialog, QDialog, QLineEdit, QRadioButton,
                             QDialogButtonBox)
from PyQt5.QtCore import Qt, QPointF, pyqtSignal, pyqtSlot, QEvent, QMarginsF
from PyQt5.QtGui import QKeySequence, QFont, QColor, QImage, QPainter, QPageSize
from PyQt5.QtPrintSupport import QPrinter
import pyqtgraph as pg
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.visualization import simple_norm
import astropy.units as u
import warnings
import pyqtgraph.exporters

# Import block for SVG handling, which can be used by PDF exporters
try:
    from PyQt5 import QtSvg
    QTSVG_AVAILABLE = True
except ImportError:
    QTSVG_AVAILABLE = False

warnings.filterwarnings('ignore')

# --- Style Configuration ---
pg.setConfigOptions(antialias=True, imageAxisOrder='row-major')
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
AXIS_LABEL_FONT_SIZE = 14
AXIS_TICK_FONT_SIZE = 12

min_bins_default = 5  # minimum zoom in x = wavelength (bins in index space)
min_y_rows_default = 5  # minimum zoom in y for 2D rows

# ------------------------- DEFAULT EMISSION LINES (rest Å) -------------------------
# These are used when no external file is loaded. See the header doc for format.
DEFAULT_EMISSION_LINES = [
    ("Lyα", 1215.6700),
    ("[O II]", 3727.0635),
    ("[O II]", 3729.8472),
    ("Hβ", 4862.6739),
    ("[O III]", 4960.2785),
    ("[O III]", 5008.2236),
    ("Hα", 6564.6237),
]

# ------------------------- AXES THAT WORK IN INDEX SPACE -------------------------
class _NiceTicksMixin:
    @staticmethod
    def _nice_step(x: float) -> float:
        """Return a 'nice' step (1,2,5 × 10^n) ≥ x"""
        if x <= 0 or not np.isfinite(x):
            return 1.0
        exp = np.floor(np.log10(x))
        f = x / (10 ** exp)
        if f <= 1.0:
            nf = 1.0
        elif f <= 2.0:
            nf = 2.0
        elif f <= 5.0:
            nf = 5.0
        else:
            nf = 10.0
        return nf * (10 ** exp)

class ObservedAxisItem(pg.AxisItem, _NiceTicksMixin):
    """
    Bottom axis for *observed* wavelength (μm) while the ViewBox X is **index**.
    We generate ticks at regular wavelength intervals and map them into
    index positions using provided mapping functions.
    """
    def __init__(self, orientation='bottom', idx_to_wave=None, wave_to_idx=None):
        super().__init__(orientation=orientation)
        self.setLabel("Wavelength (µm)")
        self._idx_to_wave = idx_to_wave
        self._wave_to_idx = wave_to_idx

    def set_mappers(self, idx_to_wave, wave_to_idx):
        self._idx_to_wave = idx_to_wave
        self._wave_to_idx = wave_to_idx
        self.picture = None
        self.update()

    def tickValues(self, minVal, maxVal, size):
        if self._idx_to_wave is None or self._wave_to_idx is None:
            return super().tickValues(minVal, maxVal, size)
        wmin = float(self._idx_to_wave(minVal))
        wmax = float(self._idx_to_wave(maxVal))
        if wmax < wmin:
            wmin, wmax = wmax, wmin
        rng = max(wmax - wmin, 1e-12)
        approxNTicks = max(int(size / 90.0), 2)
        step = self._nice_step(rng / approxNTicks)
        first = np.ceil(wmin / step) * step
        majors_wave = [v for v in np.arange(first, wmax + 1e-9, step)]
        minor_step = step / 5.0
        first_m = np.ceil(wmin / minor_step) * minor_step
        minors_wave = [v for v in np.arange(first_m, wmax + 1e-9, minor_step) if abs((v / step) - np.round(v / step)) > 1e-6]
        
        majors_idx = np.array([float(self._wave_to_idx(w)) for w in majors_wave])
        minors_idx = np.array([float(self._wave_to_idx(w)) for w in minors_wave])
        
        try:
            dx = float(self._wave_to_idx(wmin + step) - self._wave_to_idx(wmin))
            dx_m = float(self._wave_to_idx(wmin + minor_step) - self._wave_to_idx(wmin))
        except Exception:
            dx, dx_m = step, minor_step
        return [ (dx, majors_idx), (dx_m, minors_idx) ]

    def tickStrings(self, values, scale, spacing):
        if self._idx_to_wave is None:
            return super().tickStrings(values, scale, spacing)
        waves = np.array([float(self._idx_to_wave(v)) for v in values])
        delta_wave = waves[1] - waves[0] if len(waves) > 1 else 0.1
        
        out = []
        for w in waves:
            if delta_wave >= 0.1: s = f"{w:.1f}"
            elif delta_wave >= 0.01: s = f"{w:.2f}"
            else: s = f"{w:.3f}"
            out.append(s)
        return out

class RestFrameAxisItem(pg.AxisItem, _NiceTicksMixin):
    """
    Top axis that shows rest-frame wavelength in Å while ViewBox X is **index**.
    Requires mapping functions idx->obs μm and obs μm->idx, and a z getter.
    """
    def __init__(self, orientation='top', idx_to_wave=None, wave_to_idx=None, get_z=lambda: 0.0):
        super().__init__(orientation=orientation)
        self._idx_to_wave = idx_to_wave
        self._wave_to_idx = wave_to_idx
        self._get_z = get_z
        self.setLabel("Rest Wavelength (Å)")

    def set_mappers(self, idx_to_wave, wave_to_idx):
        self._idx_to_wave = idx_to_wave
        self._wave_to_idx = wave_to_idx
        self.picture = None
        self.update()

    def tickValues(self, minVal, maxVal, size):
        if self._idx_to_wave is None or self._wave_to_idx is None: return []
        try: z = max(float(self._get_z()), -0.99)
        except Exception: z = 0.0
        
        wmin, wmax = sorted([float(self._idx_to_wave(v)) for v in (minVal, maxVal)])
        rmin, rmax = (wmin / (1.0 + z)) * 1e4, (wmax / (1.0 + z)) * 1e4
        rng = max(rmax - rmin, 1e-12)
        approxNTicks = max(int(size / 90.0), 2)
        step = self._nice_step(rng / approxNTicks)
        
        first = np.ceil(rmin / step) * step
        majors_rest = [v for v in np.arange(first, rmax + 1e-9, step)]
        minor_step = step / 5.0
        first_m = np.ceil(rmin / minor_step) * minor_step
        minors_rest = [v for v in np.arange(first_m, rmax + 1e-9, minor_step) if abs((v / step) - np.round(v / step)) > 1e-6]

        majors_obs = (np.array(majors_rest) * (1.0 + z)) / 1e4
        minors_obs = (np.array(minors_rest) * (1.0 + z)) / 1e4
        majors_idx = np.array([float(self._wave_to_idx(w)) for w in majors_obs])
        minors_idx = np.array([float(self._wave_to_idx(w)) for w in minors_obs])
        
        try:
            dx = float(self._wave_to_idx(wmin + (step / 1e4) * (1.0 + z)) - self._wave_to_idx(wmin))
            dx_m = float(self._wave_to_idx(wmin + (minor_step / 1e4) * (1.0 + z)) - self._wave_to_idx(wmin))
        except Exception:
            dx, dx_m = step, minor_step
        return [ (dx, majors_idx), (dx_m, minors_idx) ]

    def tickStrings(self, values, scale, spacing):
        if self._idx_to_wave is None: return []
        try: z = max(float(self._get_z()), -0.99)
        except Exception: z = 0.0
        
        waves = np.array([float(self._idx_to_wave(v)) for v in values])
        rest = (waves / (1.0 + z)) * 1e4
        
        out = []
        for v in rest:
            av = abs(v)
            if av >= 1000: s = f"{v:.0f}"
            elif av >= 100: s = f"{v:.1f}"
            else: s = f"{v:.2f}"
            out.append(s)
        return out

class PixelRowAxisItem(pg.AxisItem):
    """
    A custom Y-axis for the 2D plot that places ticks in the middle of
    pixel rows and labels them with integer row numbers.
    """
    def __init__(self, orientation='left'):
        super().__init__(orientation=orientation)
        self.setLabel('Pixel Row')

    def tickValues(self, minVal, maxVal, size):
        # Generate ticks only at the center of visible integer rows
        min_row = max(0, int(np.floor(minVal)))
        max_row = int(np.ceil(maxVal))
        
        # Determine a reasonable step to avoid overcrowding labels
        num_visible = max_row - min_row
        if num_visible == 0: return []
        
        approx_ticks = max(int(size / 40.0), 2) # Target ~40 px per tick
        step = 1
        if num_visible > approx_ticks:
            step = int(np.ceil(num_visible / approx_ticks))
        
        ticks = [i + 0.5 for i in range(min_row, max_row, step)]
        
        # Return only a single list for major ticks, no minor ticks
        return [(step, ticks)]

    def tickStrings(self, values, scale, spacing):
        # The values are already at the center (e.g., 7.5, 8.5)
        # We just need to convert them to integer strings
        return [f"{int(v)}" for v in values]


# ------------------------------- MAIN WIDGETS --------------------------------
class SpectrumPlotWidget(pg.PlotWidget):
    """Custom plot widget with trackpad gesture support"""
    # Signal emitted when x range changes (1D plot only)
    x_range_changed = pyqtSignal(float, float)
    # Signal emitted when Alt-drag selects a Y range (ymin, ymax)
    alt_selected = pyqtSignal(float, float)

    def __init__(self, parent=None, is_2d=False, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.is_2d = is_2d

        font_label = QFont()
        font_label.setPointSize(AXIS_LABEL_FONT_SIZE)
        font_tick = QFont()
        font_tick.setPointSize(AXIS_TICK_FONT_SIZE)

        left_axis = self.getAxis('left')
        if not is_2d:
            left_axis.setLabel('Flux', units='Jy')
        # For 2D, the custom axis item sets its own label
        left_axis.label.setFont(font_label)
        left_axis.setTickFont(font_tick)
        
        self.showGrid(x=False, y=False, alpha=0.3)
        self.viewport().setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.setMouseTracking(True)
        try: self.grabGesture(Qt.PinchGesture)
        except Exception: pass
        
        self.data_x_min = None; self.data_x_max = None
        self.data_y_min = None; self.data_y_max = None
        self.min_x_range = None; self.min_y_rows = None
        self.mouse_x_pos = None; self.mouse_y_pos = None
        self._alt_drag_start = None; self._alt_region = None
        
        self.getViewBox().sigRangeChanged.connect(self.on_range_changed)

    def set_data_limits(self, x_min, x_max, y_min=None, y_max=None,
                         min_dx=None, min_bins=min_bins_default, min_y_rows=None):
        """Set the data limits to restrict panning/zooming"""
        self.data_x_min = x_min; self.data_x_max = x_max
        self.data_y_min = y_min; self.data_y_max = y_max
        if min_dx is not None:
            self.min_x_range = float(min_dx) * max(int(min_bins), 1)
        self.min_y_rows = int(min_y_rows) if (min_y_rows is not None) else None
        
        vb = self.getViewBox()
        lim_kwargs = {}
        if x_min is not None: lim_kwargs['xMin'] = x_min
        if x_max is not None: lim_kwargs['xMax'] = x_max
        if y_min is not None: lim_kwargs['yMin'] = y_min
        if y_max is not None: lim_kwargs['yMax'] = y_max
        if self.min_x_range is not None: lim_kwargs['minXRange'] = self.min_x_range
        if self.min_y_rows is not None: lim_kwargs['minYRange'] = float(self.min_y_rows)
        try: vb.setLimits(**lim_kwargs)
        except Exception: pass

    def mouseMoveEvent(self, ev):
        """Track mouse position for zoom centering and live Alt-region updates"""
        if self.sceneBoundingRect().contains(ev.pos()):
            mouse_point = self.getViewBox().mapSceneToView(ev.pos())
            self.mouse_x_pos = mouse_point.x()
            self.mouse_y_pos = mouse_point.y()
            if self._alt_drag_start is not None and self._alt_region is not None:
                vb = self.getViewBox()
                p1 = vb.mapSceneToView(self._alt_drag_start)
                p2 = mouse_point
                self._alt_region.setRegion(sorted((p1.y(), p2.y())))
        super().mouseMoveEvent(ev)

    def mousePressEvent(self, ev):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.AltModifier and ev.button() == Qt.LeftButton:
            if self.sceneBoundingRect().contains(ev.pos()):
                self._alt_drag_start = ev.pos()
                self._alt_region = pg.LinearRegionItem(
                    orientation='horizontal', brush=pg.mkBrush(255, 255, 0, 60),
                    pen=pg.mkPen('y', width=1))
                self.addItem(self._alt_region)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseDragEvent(self, ev):
        super().mouseDragEvent(ev)
        vb = self.getViewBox()
        if self.data_x_min is not None and self.data_x_max is not None:
            xmin, xmax = vb.viewRange()[0]
            xmin = max(xmin, self.data_x_min); xmax = min(xmax, self.data_x_max)
            if self.min_x_range is not None and (xmax - xmin) < self.min_x_range:
                cx = 0.5 * (xmax + xmin)
                xmin = max(self.data_x_min, cx - 0.5 * self.min_x_range)
                xmax = min(self.data_x_max, cx + 0.5 * self.min_x_range)
            vb.setXRange(xmin, xmax, padding=0)
        if self.data_y_min is not None and self.data_y_max is not None:
            ymin, ymax = vb.viewRange()[1]
            vb.setYRange(max(ymin, self.data_y_min), min(ymax, self.data_y_max), padding=0)
        
        if not self.is_2d:
            self.x_range_changed.emit(*vb.viewRange()[0])

    def mouseReleaseEvent(self, ev):
        if self._alt_drag_start is not None:
            if self._alt_region:
                ymin, ymax = self._alt_region.getRegion()
                self.alt_selected.emit(ymin, ymax)
                self.removeItem(self._alt_region)
            self._alt_drag_start = None
            self._alt_region = None
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def wheelEvent(self, ev):
        """
        Two-finger scroll/pan (X) and Ctrl+Scroll Y-scale (fallback for pinch).
        Sensitivity tuned: X zoom slower; Y zoom anchored at cursor with min 3 px.
        """
        modifiers = QApplication.keyboardModifiers()
        delta = ev.angleDelta()
        dx = delta.x() / 120.0; dy = delta.y() / 120.0
        vb = self.getViewBox()
        (xmin, xmax), (ymin, ymax) = vb.viewRange()
        view_h = max(self.viewport().height(), 1)

        if modifiers == Qt.NoModifier:
            # Zoom X
            if abs(dy) > 0:
                scale_factor = 1.02 ** (-dy)
                x_center = self.mouse_x_pos if self.mouse_x_pos is not None else 0.5 * (xmin + xmax)
                left_ratio = (x_center - xmin) / (xmax - xmin)
                new_x_range = (xmax - xmin) * scale_factor
                new_xmin = x_center - new_x_range * left_ratio
                new_xmax = new_xmin + new_x_range

                if self.data_x_min is not None: new_xmin = max(new_xmin, self.data_x_min)
                if self.data_x_max is not None: new_xmax = min(new_xmax, self.data_x_max)
                if self.min_x_range is not None and (new_xmax - new_xmin) < self.min_x_range:
                    cx = 0.5 * (new_xmin + new_xmax)
                    new_xmin = max(self.data_x_min, cx - 0.5 * self.min_x_range)
                    new_xmax = min(self.data_x_max, cx + 0.5 * self.min_x_range)
                vb.setXRange(new_xmin, new_xmax, padding=0)
            # Pan X
            if abs(dx) > 0:
                x_shift = (xmax - xmin) * dx * 0.1
                new_xmin, new_xmax = xmin + x_shift, xmax + x_shift
                if self.data_x_min is not None and new_xmin < self.data_x_min:
                    shift = self.data_x_min - new_xmin
                    new_xmin += shift; new_xmax += shift
                if self.data_x_max is not None and new_xmax > self.data_x_max:
                    shift = new_xmax - self.data_x_max
                    new_xmin -= shift; new_xmax -= shift
                vb.setXRange(new_xmin, new_xmax, padding=0)

        elif modifiers == Qt.ControlModifier and abs(dy) > 0:
            # Zoom Y
            scale_factor = 1.02 ** (-dy)
            y_cursor = self.mouse_y_pos if self.mouse_y_pos is not None else 0.5 * (ymin + ymax)
            y_range = ymax - ymin
            new_y_range = max(y_range * scale_factor, 3.0 * y_range / view_h)
            frac = (y_cursor - ymin) / y_range if y_range != 0 else 0.5
            new_ymin = y_cursor - frac * new_y_range
            new_ymax = new_ymin + new_y_range

            if self.data_y_min is not None and new_ymin < self.data_y_min:
                new_ymin = self.data_y_min
                new_ymax = new_ymin + new_y_range
            if self.data_y_max is not None and new_ymax > self.data_y_max:
                new_ymax = self.data_y_max
                new_ymin = new_ymax - new_y_range
            vb.setYRange(new_ymin, new_ymax, padding=0)
        
        ev.accept()

    def on_range_changed(self):
        if not self.is_2d:
            self.x_range_changed.emit(*self.getViewBox().viewRange()[0])

    def event(self, ev):
        if ev.type() == QEvent.Gesture and (g := ev.gesture(Qt.PinchGesture)):
            scale = g.scaleFactor()
            vb = self.getViewBox()
            (xmin, xmax), (ymin, ymax) = vb.viewRange()
            view_h = max(self.viewport().height(), 1)
            
            center_scene = g.centerPoint()
            y_cursor = vb.mapSceneToView(center_scene).y() if center_scene else 0.5 * (ymin + ymax)
            
            new_range = max((ymax - ymin) / max(scale, 1e-6), 3.0 * (ymax - ymin) / view_h)
            frac = (y_cursor - ymin) / (ymax - ymin) if ymax != ymin else 0.5
            new_ymin = y_cursor - frac * new_range
            new_ymax = new_ymin + new_range

            if self.data_y_min is not None and new_ymin < self.data_y_min:
                new_ymin = self.data_y_min; new_ymax = new_ymin + new_range
            if self.data_y_max is not None and new_ymax > self.data_y_max:
                new_ymax = self.data_y_max; new_ymin = new_ymax - new_range
                
            vb.setYRange(new_ymin, new_ymax, padding=0)
            ev.accept()
            return True
        return super().event(ev)

# ------------------------- MANUAL EXTRACTION REGION ------------------------

class DraggableRegionItem(pg.LinearRegionItem):
    """
    A LinearRegionItem that snaps to integer row boundaries.
    """
    def __init__(self, *args, **kwargs):
        color = kwargs.pop('color', QColor('green'))
        self.color = color if isinstance(color, QColor) else QColor(color)

        pen = pg.mkPen(self.color, width=2)
        brush_color = QColor(self.color); brush_color.setAlpha(25)
        brush = pg.mkBrush(brush_color)

        kwargs.update({'pen': pen, 'hoverPen': pen, 'brush': brush, 'hoverBrush': brush})
        super().__init__(*args, **kwargs)

        try: self.sigRegionChanged.disconnect()
        except TypeError: pass 

        self.lines[0].sigPositionChanged.connect(self._line_moved)
        self.lines[1].sigPositionChanged.connect(self._line_moved)

    def _line_moved(self):
        """Ensures lines snap to integers and don't cross."""
        bottom_val, top_val = self.lines[0].value(), self.lines[1].value()
        ry_bottom, ry_top = round(bottom_val), round(top_val)

        if ry_top <= ry_bottom:
            if abs(bottom_val - ry_bottom) > abs(top_val - ry_top):
                ry_bottom = ry_top - 1
            else:
                ry_top = ry_bottom + 1
        
        self.blockSignals(True)
        self.setRegion((ry_bottom, ry_top))
        self.blockSignals(False)

    def set_color(self, color):
        """Update the color of the region."""
        self.color = color if isinstance(color, QColor) else QColor(color)
        pen = pg.mkPen(self.color, width=2)
        brush_color = QColor(self.color); brush_color.setAlpha(25)
        brush = pg.mkBrush(brush_color)

        for line in self.lines:
            line.setPen(pen); line.setHoverPen(pen)
        self.setBrush(brush); self.setHoverBrush(brush)

    
# ------------------------- MAIN APPLICATION WINDOW -------------------------
class FITSSpectraViewer(QMainWindow):
    """Main application window for FITS spectra viewing"""
    def __init__(self):
        super().__init__()
        self.s2d_data = None; self.x1d_wave = None
        self.x1d_flux = None; self.x1d_fluxerr = None
        self.current_s2d_file = None; self.current_x1d_file = None
        self.splitter = None; self.z_input = None
        self._have_plotted = False
        self._nav_stack = []; self._nav_index = -1
        self._is_restoring = False; self._syncing_x = False
        self.cursor_line = None; self.s2d_label = None; self.x1d_label = None
        self._idx_to_wave = None; self._wave_to_idx = None; self._nx = None
        
        self.emission_lines = list(DEFAULT_EMISSION_LINES)
        self.em_line_items_1d, self.em_label_items_1d, self.em_line_items_2d = [], [], []
        self.show_lines_check = None; self.load_lines_btn = None
        
        self.spectra_sources = []
        self._is_programmatic_change = False

        self.init_ui()

    # --------------------------- Alignment helpers ---------------------------
    def _build_index_wavelength_mappers(self):
        """Create callable mppers index<->wavelength given current x1d_wave."""
        if self.x1d_wave is None:
            self._idx_to_wave, self._wave_to_idx, self._nx = None, None, None
            return
        w = np.asarray(self.x1d_wave, dtype=float)
        
        if not np.all(np.diff(w[np.isfinite(w)]) >= 0):
            order = np.argsort(w)
            w_sorted, idx_sorted = w[order], np.arange(w.size, dtype=float)[order]
            self._idx_to_wave = lambda i: np.interp(i, idx_sorted, w_sorted)
            self._wave_to_idx = lambda x: np.interp(x, w_sorted, idx_sorted)
        else:
            idx = np.arange(w.size, dtype=float)
            self._idx_to_wave = lambda i: np.interp(i, idx, w)
            self._wave_to_idx = lambda x: np.interp(x, w, idx)
        self._nx = int(w.size)

    def _compute_bin_edges(self, n):
        """Return integer bin edges [0..n] for n spectral columns."""
        return np.arange(int(n) + 1, dtype=float)

    def expand_wavelength_gap(self, x1d_wave, x1d_flux, x1d_fluxerr, s2d_data, expand_wavelength_gap=True):
        """Detect large wavelength gaps in 1D and insert NaNs into 1D flux and 2D data to keep alignment."""
        if not expand_wavelength_gap or x1d_wave is None or s2d_data is None:
            return x1d_wave, x1d_flux, x1d_fluxerr, s2d_data
            
        dx1d_wave = np.diff(x1d_wave)
        if len(dx1d_wave) == 0:
             return x1d_wave, x1d_flux, x1d_fluxerr, s2d_data
             
        igap = np.argmax(dx1d_wave)
        dx1d_max = dx1d_wave[igap]
        
        left = dx1d_wave[igap-1] if igap > 0 else dx1d_max
        right = dx1d_wave[igap+1] if igap < len(dx1d_wave) - 1 else dx1d_max
        dx_replace = (left + right) / 2.
        
        if dx_replace <= 0:
            return x1d_wave, x1d_flux, x1d_fluxerr, s2d_data
            
        num_fill = int(np.round(dx1d_max / dx_replace))
        
        if num_fill > 1 and dx1d_max > 1.5 * dx_replace:
            # Create the values to insert
            wave_fill = np.linspace(x1d_wave[igap], x1d_wave[igap+1], num_fill + 1)[1:-1]
            s2d_fill = np.full((s2d_data.shape[0], num_fill - 1), np.nan)
            x1d_fill = np.full(num_fill - 1, np.nan)

            # Use concatenate to build the new arrays
            x1d_wave = np.concatenate([x1d_wave[:igap+1], wave_fill, x1d_wave[igap+1:]])
            x1d_flux = np.concatenate([x1d_flux[:igap+1], x1d_fill, x1d_flux[igap+1:]])
            x1d_fluxerr = np.concatenate([x1d_fluxerr[:igap+1], x1d_fill, x1d_fluxerr[igap+1:]])
            s2d_data = np.concatenate([s2d_data[:, :igap+1], s2d_fill, s2d_data[:, igap+1:]], axis=1)

        return x1d_wave, x1d_flux, x1d_fluxerr, s2d_data

    # --------------------------------- UI ---------------------------------
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('FITS Spectra Viewer')
        self.setGeometry(100, 100, 1400, 900)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        self.create_toolbar()
        layout.addWidget(self.create_control_panel())
        
        self.splitter = QSplitter(Qt.Vertical)

        # 2D spectrum display
        self.plot_2d = SpectrumPlotWidget(is_2d=True, axisItems={'left': PixelRowAxisItem()})
        self.plot_2d.setAspectLocked(False)
        self.image_item = pg.ImageItem(axisOrder='row-major')
        self.image_item.setAutoDownsample(False)
        self.plot_2d.addItem(self.image_item)
        self.plot_2d.getPlotItem().getViewBox().setBorder(pg.mkPen('k'))

        font_label = QFont(); font_label.setPointSize(AXIS_LABEL_FONT_SIZE)
        font_tick = QFont(); font_tick.setPointSize(AXIS_TICK_FONT_SIZE)
        
        pi2 = self.plot_2d.getPlotItem()
        pi2.hideAxis('bottom')
        pi2.showAxis('top', True)
        self.top_axis = RestFrameAxisItem(
            'top',
            idx_to_wave=lambda x: self._idx_to_wave(x) if self._idx_to_wave else x,
            wave_to_idx=lambda w: self._wave_to_idx(w) if self._wave_to_idx else w,
            get_z=lambda: self.z_input.value() if self.z_input else 0.0,
        )
        self.top_axis.label.setFont(font_label)
        self.top_axis.setTickFont(font_tick)

        old_top = pi2.getAxis('top')
        pi2.layout.removeItem(old_top)
        pi2.axes['top']['item'] = self.top_axis
        pi2.layout.addItem(self.top_axis, 1, 1)
        self.top_axis.linkToView(pi2.vb)

        colors = [(0, 0, 180), (100, 150, 255), (255, 255, 255), (255, 150, 100), (180, 0, 0)]
        cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), color=colors)
        self.image_item.setLookupTable(cmap.getLookupTable())
        self.splitter.addWidget(self.plot_2d)

        self.plot_2d.scene().sigMouseMoved.connect(self._on_2d_mouse_moved)
        self.plot_2d.viewport().installEventFilter(self)
        self.plot_2d.getViewBox().sigRangeChanged.connect(self._on_2d_range_changed)

        # 1D spectrum display
        obs_axis = ObservedAxisItem('bottom')
        obs_axis.label.setFont(font_label); obs_axis.setTickFont(font_tick)
        self.plot_1d = SpectrumPlotWidget(axisItems={'bottom': obs_axis})
        self.plot_1d.getPlotItem().getViewBox().setBorder(pg.mkPen('k'))
        self.obs_axis = self.plot_1d.getAxis('bottom')
        self.plot_1d.hideAxis('top')

        self.plot_1d.x_range_changed.connect(self.sync_x_range_to_2d)
        self.plot_1d.getViewBox().sigRangeChanged.connect(self._on_any_range_changed)
        self.plot_2d.getViewBox().sigRangeChanged.connect(self._on_any_range_changed)
        self.plot_1d.getViewBox().sigRangeChanged.connect(self._relayout_emission_labels)

        self.cursor_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('magenta', width=1), movable=False)
        self.plot_1d.addItem(self.cursor_line); self.cursor_line.hide()
        self.splitter.addWidget(self.plot_1d)

        self.plot_1d.alt_selected.connect(lambda ymin, ymax: self.plot_1d.setYRange(ymin, ymax, padding=0))
        self.plot_2d.alt_selected.connect(lambda ymin, ymax: self.plot_1d.setYRange(ymin, ymax, padding=0))
        
        self.plot_1d.scene().sigMouseMoved.connect(self._on_1d_mouse_moved)
        self.plot_1d.viewport().installEventFilter(self)

        self.splitter.setSizes([225, 675])
        layout.addWidget(self.splitter)

        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar)
        self.s2d_label = QLabel(""); self.x1d_label = QLabel("")
        mono_css = "QLabel { background-color: #f0f0f0; padding: 3px 6px; color: black; }"
        self.s2d_label.setStyleSheet(mono_css); self.x1d_label.setStyleSheet(mono_css)
        status_container = QWidget()
        vb = QVBoxLayout(status_container); vb.setContentsMargins(0, 0, 0, 0); vb.setSpacing(0)
        vb.addWidget(self.s2d_label); vb.addWidget(self.x1d_label)
        self.status_bar.addPermanentWidget(status_container, 1)
        self.status_bar.showMessage('Ready. Use File menu to load FITS data.')
        self._setup_spectra_controls()
        self._update_status_clear()

    def create_toolbar(self):
        """Create application toolbar"""
        toolbar = QToolBar(); self.addToolBar(toolbar)
        actions = [
            ('Open FITS...', 'Ctrl+O', self.open_fits_file),
            ('Save Plot...', 'Ctrl+P', self._save_plot),
            None,
            ('Open State...', 'Ctrl+Shift+O', self._open_state_file),
            ('Save State...', 'Ctrl+S', self._save_state),
            None,
            ('Reset View', 'R', self.reset_views),
            ('Autoscale', 'A', self.autoscale),
            None,
            ('Back', 'Alt+Left', self.nav_back),
            ('Forward', 'Alt+Right', self.nav_forward)
        ]
        for item in actions:
            if item is None:
                toolbar.addSeparator()
                continue
            name, shortcut, func = item
            action = QAction(name, self); action.setShortcut(shortcut)
            action.triggered.connect(func); toolbar.addAction(action)

    def create_control_panel(self):
        """Create control panel for display options and spectra controls."""
        panel_container = QWidget()
        main_layout = QHBoxLayout(panel_container); main_layout.setContentsMargins(0, 0, 0, 0)
        
        display_group = QGroupBox("Display Controls")
        display_v_layout = QVBoxLayout(display_group)
        
        row1 = QHBoxLayout(); row1.addWidget(QLabel("2D: Colormap:"))
        self.cmap_combo = QComboBox(); self.cmap_combo.addItems(['RdBu', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'])
        self.cmap_combo.currentTextChanged.connect(self.change_colormap); row1.addWidget(self.cmap_combo)
        row1.addSpacing(14); row1.addWidget(QLabel("Sigma Clip:"))
        self.sigma_spin = QSpinBox(); self.sigma_spin.setRange(1, 10); self.sigma_spin.setValue(5)
        self.sigma_spin.valueChanged.connect(self.update_display); row1.addWidget(self.sigma_spin)
        row1.addStretch(); display_v_layout.addLayout(row1)

        row2 = QHBoxLayout(); row2.addWidget(QLabel("1D:"))
        self.show_errors_check = QCheckBox("Show Uncertainty"); self.show_errors_check.setChecked(True)
        self.show_errors_check.toggled.connect(self.update_display); row2.addWidget(self.show_errors_check)
        row2.addSpacing(14)
        self.autoscale_y_check = QCheckBox("Autoscale Flux"); self.autoscale_y_check.setChecked(False)
        self.autoscale_y_check.toggled.connect(self.on_autoscale_toggled); row2.addWidget(self.autoscale_y_check)
        row2.addStretch(); display_v_layout.addLayout(row2)

        row3 = QHBoxLayout(); row3.addWidget(QLabel("z:"))
        self.z_input = QDoubleSpinBox(); self.z_input.setDecimals(4); self.z_input.setRange(-0.99, 12.0)
        self.z_input.setSingleStep(0.001); self.z_input.setValue(0.0)
        self.z_input.valueChanged.connect(self._refresh_top_axis)
        self.z_input.valueChanged.connect(self._update_emission_lines); row3.addWidget(self.z_input)
        row3.addSpacing(14)
        self.show_lines_check = QCheckBox("Show Emission Lines"); self.show_lines_check.setChecked(True)
        self.show_lines_check.toggled.connect(self._update_emission_lines); row3.addWidget(self.show_lines_check)
        self.load_lines_btn = QPushButton("Load Lines…"); self.load_lines_btn.clicked.connect(self.load_emission_lines_file)
        row3.addWidget(self.load_lines_btn); row3.addStretch(); display_v_layout.addLayout(row3)
        
        main_layout.addWidget(display_group)

        self.spectra_controls_group = QGroupBox("1D Extractions")
        self.extraction_controls_layout = QVBoxLayout(self.spectra_controls_group)
        self.extraction_controls_layout.addStretch(1); main_layout.addWidget(self.spectra_controls_group)
        
        main_layout.addStretch(1)
        
        help_text = ("X Wavelength: Zoom Scroll ⬆/⬇ ; Pan Scroll ⬅/➡\n"
                     "Y Zoom: Pinch / Ctrl+Scroll / Alt/Option+Drag\n"
                     "Alt/Option ⬅/➡: Back/Forward (history)\n" "R: Reset View\n" "A: Autoscale Flux")
        gestures_label = QLabel(help_text); gestures_label.setStyleSheet("QLabel { color: #666; }")
        main_layout.addWidget(gestures_label)

        return panel_container

    # --------------------------- Manual Extraction --------------------------
    def _setup_spectra_controls(self):
        """Create and manage the UI controls for the three spectra sources."""
        while self.extraction_controls_layout.count():
            item = self.extraction_controls_layout.takeAt(0)
            if (widget := item.widget()): widget.deleteLater()
        self.spectra_sources = []
        
        source_defs = [('x1d', QColor(Qt.black), 'x1d.fits'),
                       ('manual', QColor(0, 222, 0), 'Manual 1'), 
                       ('manual', QColor(139, 69, 19), 'Manual 2')]

        for i, (stype, color, label) in enumerate(source_defs):
            source = {'type': stype, 'color': color, 'visible': (i==0)}
            row = QWidget(); row_layout = QHBoxLayout(row); row_layout.setContentsMargins(0, 0, 0, 0)
            
            cb = QCheckBox(); cb.setChecked(i == 0); row_layout.addWidget(cb); source['checkbox'] = cb
            swatch = QLabel(); swatch.setFixedSize(16, 16); self._update_swatch_color(swatch, color)
            row_layout.addWidget(swatch); source['color_swatch'] = swatch

            if stype == 'x1d':
                lbl = QLabel(label); row_layout.addWidget(lbl); source['label'] = lbl
            else:
                row_layout.addWidget(QLabel("Rows:"))
                spin_start = QSpinBox(); spin_start.setRange(0, 9999); row_layout.addWidget(spin_start)
                row_layout.addWidget(QLabel("–"))
                spin_stop = QSpinBox(); spin_stop.setRange(0, 9999); spin_stop.setValue(5); row_layout.addWidget(spin_stop)
                row_layout.addWidget(QLabel("Width:"))
                spin_width = QSpinBox(); spin_width.setRange(1, 999); spin_width.setValue(6); row_layout.addWidget(spin_width)
                
                for s in [spin_start, spin_stop, spin_width]: s.setFixedWidth(50)
                
                btn_center = QPushButton("Center"); btn_center.setFixedWidth(60); btn_center.setCheckable(True)
                btn_peak = QPushButton("Peak"); btn_peak.setFixedWidth(60); btn_peak.setCheckable(True)
                row_layout.addWidget(btn_center); row_layout.addWidget(btn_peak)
                source.update({'spin_width': spin_width, 'spin_start': spin_start, 'spin_stop': spin_stop,
                               'btn_center': btn_center, 'btn_peak': btn_peak})
            
            row_layout.addStretch()
            self.extraction_controls_layout.insertWidget(i, row)
            self.spectra_sources.append(source)

        for i, source in enumerate(self.spectra_sources):
            source['checkbox'].toggled.connect(lambda checked, idx=i: self._on_spectrum_toggled(idx, checked))
            source['color_swatch'].mousePressEvent = lambda event, idx=i: self._on_color_swatch_clicked(idx)
            if source['type'] == 'manual':
                source['spin_start'].valueChanged.connect(lambda val, idx=i: self._update_region_from_spins(idx))
                source['spin_stop'].valueChanged.connect(lambda val, idx=i: self._update_region_from_spins(idx))
                source['spin_start'].editingFinished.connect(lambda idx=i: self._on_start_spin_edited(idx))
                source['spin_stop'].editingFinished.connect(lambda idx=i: self._on_stop_spin_edited(idx))
                source['spin_width'].valueChanged.connect(lambda val, idx=i: self._on_width_spin_changed(idx))
                source['btn_center'].clicked.connect(lambda _, idx=i: self._center_region(idx))
                source['btn_peak'].clicked.connect(lambda _, idx=i: self._find_peak_region(idx))

    def _on_width_spin_changed(self, idx):
        source = self.spectra_sources[idx]
        if self.s2d_data is None or not source['spin_width'].hasFocus(): return
        ny = self.s2d_data.shape[0]
        start, width = source['spin_start'].value(), source['spin_width'].value()
        new_stop = min(start + width - 1, ny - 1)
        source['spin_stop'].setValue(new_stop)

    def _on_start_spin_edited(self, idx):
        source = self.spectra_sources[idx]
        if self.s2d_data is None: return
        ny = self.s2d_data.shape[0]
        start, width = source['spin_start'].value(), source['spin_width'].value()
        new_stop = min(start + width - 1, ny - 1)
        source['spin_stop'].setValue(new_stop)

    def _on_stop_spin_edited(self, idx):
        source = self.spectra_sources[idx]
        if self.s2d_data is None: return
        stop, width = source['spin_stop'].value(), source['spin_width'].value()
        new_start = max(0, stop - width + 1)
        source['spin_start'].setValue(new_start)

    def _update_width_from_rows(self, idx):
        source = self.spectra_sources[idx]
        width = source['spin_stop'].value() - source['spin_start'].value() + 1
        source['spin_width'].blockSignals(True)
        source['spin_width'].setValue(width)
        source['spin_width'].blockSignals(False)

    def _update_region_from_spins(self, idx):
        source = self.spectra_sources[idx]
        start_spin, stop_spin = source['spin_start'], source['spin_stop']
        
        if not self._is_programmatic_change:
            start_spin.blockSignals(True); stop_spin.blockSignals(True)
            start_val, stop_val = start_spin.value(), stop_spin.value()
            if start_val > stop_val:
                if start_spin.hasFocus(): stop_spin.setValue(start_val)
                else: start_spin.setValue(stop_val)
            start_spin.blockSignals(False); stop_spin.blockSignals(False)
        
        if self.s2d_data is not None:
            ny = self.s2d_data.shape[0]
            max_width = ny - start_spin.value()
            source['spin_width'].setMaximum(max_width)
            
        if source.get('region'):
            region_start, region_stop = start_spin.value(), stop_spin.value()
            source['region'].setRegion((region_start, region_stop + 1))
            self._update_extraction(idx)
        
        self._update_width_from_rows(idx)
        if not self._is_programmatic_change:
            self._reset_extraction_button_state(idx)

    def _set_manual_region_rows(self, idx, start, stop):
        source = self.spectra_sources[idx]
        if not source['type'] == 'manual': return
        self._is_programmatic_change = True
        
        start, stop = int(round(start)), int(round(stop))
        source['spin_start'].setValue(start)
        source['spin_stop'].setValue(stop)
        
        self._is_programmatic_change = False
        self._update_region_from_spins(idx)

    def _reset_extraction_button_state(self, idx):
        source = self.spectra_sources[idx]
        if source['type'] == 'manual':
            source['btn_center'].setChecked(False); source['btn_peak'].setChecked(False)

    def _center_region(self, idx):
        if self.s2d_data is None: return
        source = self.spectra_sources[idx]
        ny = self.s2d_data.shape[0]
        width = source['spin_width'].value()
        
        center_y = ny / 2
        half_low = (width - 1) // 2
        start = int(round(center_y - half_low))
        stop = start + width - 1

        if stop >= ny: stop, start = ny - 1, ny - width
        if start < 0: start, stop = 0, width - 1
        
        self._set_manual_region_rows(idx, start, stop)
        if source.get('btn_center'): source['btn_center'].setChecked(True)
        if source.get('btn_peak'): source['btn_peak'].setChecked(False)

    def _find_peak_region(self, idx, use_visible_range=False):
        if self.s2d_data is None: return
        source = self.spectra_sources[idx]
        ny, nx = self.s2d_data.shape; width = source['spin_width'].value()
        if ny < width: return

        # Always search the full data range for peak
        visible_data = self.s2d_data
        
        row_sums = np.nansum(visible_data, axis=1)
        if len(row_sums) == 0: return

        window_sums = np.convolve(row_sums, np.ones(width, dtype=int), 'valid')
        best_y_start = np.nanargmax(window_sums)
        
        self._set_manual_region_rows(idx, best_y_start, best_y_start + width - 1)
        if source.get('btn_peak'): source['btn_peak'].setChecked(True)
        if source.get('btn_center'): source['btn_center'].setChecked(False)

    def _on_spectrum_toggled(self, idx, checked):
        source = self.spectra_sources[idx]
        source['visible'] = checked
        if source['type'] == 'x1d':
            if hasattr(self, 'flux_curve'): self.flux_curve.setVisible(checked)
            if hasattr(self, 'err_curve'): self.err_curve.setVisible(checked and self.show_errors_check.isChecked())
        else:
            if checked and source.get('region') is None and self.s2d_data is not None:
                self._create_manual_extraction(idx)
            if source.get('region'): source['region'].setVisible(checked)
            if source.get('curve'): source['curve'].setVisible(checked)
        self._update_cursor_dots()

    def _on_color_swatch_clicked(self, idx):
        source = self.spectra_sources[idx]
        if (color := QColorDialog.getColor(source['color'], self)).isValid():
            source['color'] = color
            self._update_swatch_color(source['color_swatch'], color)
            if source['type'] == 'x1d' and hasattr(self, 'flux_curve'):
                self.flux_curve.setPen(color)
            elif source['type'] == 'manual':
                if source.get('curve'): source['curve'].setPen(color)
                if source.get('region'): source['region'].set_color(color)
            if source.get('cursor_dot'): source['cursor_dot'].setBrush(color)

    def _on_region_dragged(self, region_item, idx):
        source = self.spectra_sources[idx]
        ymin, ymax = region_item.getRegion()
        
        start, stop = int(round(ymin)), int(round(ymax)) - 1
        
        self._is_programmatic_change = True
        source['spin_start'].setValue(start)
        source['spin_stop'].setValue(stop)
        self._is_programmatic_change = False
        
        self._update_extraction(idx)
        self._update_width_from_rows(idx)
        self._reset_extraction_button_state(idx)

    def _update_swatch_color(self, swatch, color):
        swatch.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")

    def _create_manual_extraction(self, idx):
        if self.s2d_data is None: return
        source = self.spectra_sources[idx]; ny = self.s2d_data.shape[0]
        
        start, stop = source['spin_start'].value(), source['spin_stop'].value()

        region = DraggableRegionItem(values=[start, stop + 1], orientation='horizontal',
            color=source['color'], movable=True, bounds=[0, ny])
        region.sigRegionChangeFinished.connect(lambda r=region: self._on_region_dragged(r, idx))
            
        curve = self.plot_1d.plot(pen=pg.mkPen(source['color'], width=2))
        dot = pg.ScatterPlotItem(size=8, brush=source['color'], pen='k')
        self.plot_1d.addItem(dot); dot.hide()

        source.update({'region': region, 'curve': curve, 'cursor_dot': dot})
        self.plot_2d.addItem(region)
        self._update_extraction(idx)

    def _update_extraction(self, idx):
        if self.s2d_data is None: return
        source = self.spectra_sources[idx]
        if not all(k in source for k in ['curve', 'region']): return

        ymin, ymax = source['region'].getRegion()
        ystart, ystop = int(round(ymin)), int(round(ymax)) - 1
        
        if ystart > ystop:
            source['curve'].setData([], []); return

        extracted_flux = np.nansum(self.s2d_data[ystart:ystop+1, :], axis=0)
        num_rows = ystop - ystart + 1
        extracted_flux *= 6.0 / max(1, num_rows)

        pixel_area = (0.1 * u.arcsec) ** 2
        extracted_flux *= (u.MJy/u.sr * pixel_area).to(u.Jy).value * 2
        
        nx = len(extracted_flux)
        x_edges = self._compute_bin_edges(nx)
        source['curve'].setData(x_edges, extracted_flux, stepMode=True)

    # --------------------------- File I/O & Main Logic ---------------------------
    def open_fits_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open FITS File", "",
            "FITS Files (*.fits *.fit);;All Files (*.*)")
        if file_path:
            try: self.load_fits_pair(file_path)
            except Exception as e: QMessageBox.critical(self, "Error", f"Failed to load FITS file:\n{e}")

    def load_fits_pair(self, file_path):
        base_name = os.path.basename(file_path).lower()
        if 's2d' in base_name:
            s2d_path = file_path
            x1d_path = file_path.replace('s2d', 'x1d').replace('S2D', 'X1D')
            if not os.path.exists(x1d_path): x1d_path = file_path.replace('_s2d', '_x1d').replace('_S2D', '_X1D')
        elif 'x1d' in base_name:
            x1d_path = file_path
            s2d_path = file_path.replace('x1d', 's2d').replace('X1D', 'S2D')
            if not os.path.exists(s2d_path): s2d_path = file_path.replace('_x1d', '_s2d').replace('_X1D', '_S2D')
        else:
            x1d_path, s2d_path = file_path, None
        if not os.path.exists(x1d_path): x1d_path = None
        if not os.path.exists(s2d_path): s2d_path = None

        self.s2d_data, self.x1d_wave, self.x1d_flux, self.x1d_fluxerr = None, None, None, None
        self.current_s2d_file, self.current_x1d_file = s2d_path, x1d_path
        
        if s2d_path: self.load_s2d_data(s2d_path)
        if x1d_path: self.load_x1d_data(x1d_path)
        elif self.s2d_data is not None: self.extract_1d_from_2d()
        if self.x1d_flux is None: raise ValueError("Could not load or extract a 1D spectrum.")

        self.x1d_wave, self.x1d_flux, self.x1d_fluxerr, self.s2d_data = self.expand_wavelength_gap(
            self.x1d_wave, self.x1d_flux, self.x1d_fluxerr, self.s2d_data)
        
        self._build_index_wavelength_mappers()
        if self.obs_axis: self.obs_axis.set_mappers(self._idx_to_wave, self._wave_to_idx)
        if self.top_axis: self.top_axis.set_mappers(self._idx_to_wave, self._wave_to_idx)
        
        self.spectra_sources[0]['label'].setText(os.path.basename(x1d_path) if x1d_path else 'x1d (no file)')
        self.spectra_sources[0]['checkbox'].setEnabled(bool(x1d_path))
        
        is_1d_only = self.s2d_data is None
        self.plot_2d.setVisible(not is_1d_only); self.s2d_label.setVisible(not is_1d_only)
        
        for i, source in enumerate(self.spectra_sources):
            if source['type'] == 'manual':
                for w_name in ['checkbox', 'spin_start', 'spin_stop', 'spin_width', 'btn_center', 'btn_peak']:
                    if w := source.get(w_name): w.setEnabled(not is_1d_only)
                if not is_1d_only:
                    ny = self.s2d_data.shape[0]
                    source['spin_start'].setRange(0, ny - 1)
                    source['spin_stop'].setRange(0, ny - 1)
                    source['spin_width'].setRange(1, ny)
        
        if not is_1d_only:
            self._find_peak_region(1); self._center_region(2)
            
        if is_1d_only: self.splitter.setSizes([0, 1])
        else: self.splitter.setSizes([225, 675])
        
        self.update_display()
        self._update_emission_lines()
        
        if self.s2d_data is not None and not x1d_path:
            self.spectra_sources[0]['checkbox'].setChecked(False)
            self.spectra_sources[1]['checkbox'].setChecked(True)
            self.spectra_sources[2]['checkbox'].setChecked(True)
        else:
            self.spectra_sources[0]['checkbox'].setChecked(True)
            self.spectra_sources[1]['checkbox'].setChecked(False)
            self.spectra_sources[2]['checkbox'].setChecked(False)

        status = [f"S2D: {os.path.basename(f)}" for f in [s2d_path] if f]
        status += [f"X1D: {os.path.basename(f)}" for f in [x1d_path] if f]
        self.status_bar.showMessage(" ".join(status), 5000)
        self._update_status_clear()

    def load_s2d_data(self, file_path):
        with fits.open(file_path) as hdul:
            for hdu in hdul:
                if hdu.data is not None and len(hdu.data.shape) == 2:
                    self.s2d_data = hdu.data.copy()
                    self.s2d_data[self.s2d_data == 0] = np.nan
                    return

    def load_x1d_data(self, file_path):
        with fits.open(file_path) as hdul:
            for hdu in hdul:
                if hasattr(hdu, 'data') and hdu.data is not None:
                    if hasattr(hdu, 'columns') and 'WAVELENGTH' in hdu.columns.names:
                        data = hdu.data
                        self.x1d_wave = data['WAVELENGTH'].copy()
                        self.x1d_flux = data['FLUX'].copy()
                        # --- CORRECTED LINE ---
                        if 'FLUX_ERROR' in hdu.columns.names:
                            self.x1d_fluxerr = data['FLUX_ERROR'].copy()
                        else:
                            self.x1d_fluxerr = np.abs(self.x1d_flux) * 0.1
                        return
                    elif len(hdu.data.shape) == 1:
                        self.x1d_flux = hdu.data.copy()
                        h = hdu.header
                        if 'CRVAL1' in h:
                            nx = len(self.x1d_flux)
                            self.x1d_wave = h['CRVAL1'] + (np.arange(nx) - h.get('CRPIX1', 1.0) + 1) * h.get('CDELT1', 1.0)
                        else:
                            self.x1d_wave = np.linspace(1.0, 5.5, len(self.x1d_flux))
                        self.x1d_fluxerr = np.abs(self.x1d_flux) * 0.1
                        return

    def extract_1d_from_2d(self):
        if self.s2d_data is None: return
        ny, nx = self.s2d_data.shape
        w = 6; y_cen = ny // 2
        ystart, ystop = max(0, y_cen - w // 2), min(ny, y_cen + w // 2)
        self.x1d_flux = np.nansum(self.s2d_data[ystart:ystop, :], axis=0)
        self.x1d_wave = np.linspace(1.0, 5.5, nx)
        self.x1d_fluxerr = np.abs(self.x1d_flux) * 0.1

    # -------------------------------- Display & UI Sync --------------------------------
    def update_display(self):
        if self.x1d_flux is None: return
        x1d_rng = self.plot_1d.getViewBox().viewRange() if self._have_plotted else None
        x2d_rng = self.plot_2d.getViewBox().viewRange() if self._have_plotted else None
        
        if self.s2d_data is not None:
            sigma = self.sigma_spin.value()
            clipped = sigma_clip(self.s2d_data[np.isfinite(self.s2d_data)], sigma=sigma, maxiters=3)
            vmin, vmax = np.min(clipped), np.max(clipped)
            ny, nx = self.s2d_data.shape
            self.image_item.setImage(self.s2d_data, autoLevels=False, levels=(vmin, vmax))
            self.image_item.setRect(pg.QtCore.QRectF(0.0, 0.0, float(nx), float(ny)))
            self.plot_2d.set_data_limits(0.0, float(nx), 0.0, float(ny), min_dx=1.0, 
                                         min_bins=min_bins_default, min_y_rows=min_y_rows_default)
            self.plot_2d.setYRange(0, ny, padding=0)

        if self.x1d_wave is not None:
            nx = self._nx if self._nx else len(self.x1d_flux)
            x_edges = self._compute_bin_edges(nx)
            
            if not hasattr(self, 'flux_curve'):
                pen = pg.mkPen(self.spectra_sources[0]['color'], width=1)
                self.flux_curve = self.plot_1d.plot(stepMode=True, pen=pen, name='Flux')
                self.spectra_sources[0]['curve'] = self.flux_curve
            self.flux_curve.setData(x_edges, self.x1d_flux)
            self.flux_curve.setVisible(self.spectra_sources[0]['visible'])

            show_err = self.show_errors_check.isChecked() and self.x1d_fluxerr is not None
            if not hasattr(self, 'err_curve') and show_err:
                self.err_curve = self.plot_1d.plot(stepMode=True, pen=pg.mkPen('r', width=0.5), name='Error')
                self.spectra_sources[0]['err_curve'] = self.err_curve
            if hasattr(self, 'err_curve'):
                if show_err: self.err_curve.setData(x_edges, self.x1d_fluxerr)
                self.err_curve.setVisible(show_err and self.spectra_sources[0]['visible'])

            if not hasattr(self, 'zero_curve'):
                self.zero_curve = self.plot_1d.plot(stepMode=True, pen=pg.mkPen(0.5, width=0.5, style=Qt.DashLine))
            self.zero_curve.setData(x_edges, np.zeros_like(self.x1d_flux))

            if not self.spectra_sources[0].get('cursor_dot'):
                dot = pg.ScatterPlotItem(size=8, brush=self.spectra_sources[0]['color'], pen='k')
                self.plot_1d.addItem(dot); dot.hide(); self.spectra_sources[0]['cursor_dot'] = dot

            self.plot_1d.set_data_limits(0.0, float(nx), min_dx=1.0, min_bins=min_bins_default)
            if not self._have_plotted:
                self.reset_views()
            else:
                if x1d_rng: self.plot_1d.getViewBox().setRange(xRange=x1d_rng[0], yRange=x1d_rng[1], padding=0)
                if x2d_rng: self.plot_2d.getViewBox().setRange(xRange=x2d_rng[0], yRange=x2d_rng[1], padding=0)            
            self._have_plotted = True
        self._relayout_emission_labels()

    def sync_x_range_to_2d(self, xmin, xmax):
        if self.s2d_data is not None and not self._syncing_x:
            self._syncing_x = True
            self.plot_2d.setXRange(xmin, xmax, padding=0)
            self._syncing_x = False
        
        if self.autoscale_y_check.isChecked() and self._nx:
            mask = (np.arange(self._nx) >= np.floor(xmin)) & (np.arange(self._nx) <= np.ceil(xmax))
            all_flux = [src['curve'].yData[mask] for src in self.spectra_sources 
                        if src['visible'] and src.get('curve') and src['curve'].yData is not None]
            if all_flux:
                try:
                    full_flux = np.concatenate(all_flux)
                    flux_min, flux_max = np.nanmin(full_flux), np.nanmax(full_flux)
                    if np.isfinite(flux_min) and np.isfinite(flux_max):
                        margin = (flux_max - flux_min) * 0.1 if flux_max > flux_min else 1.0
                        self.plot_1d.setYRange(min(flux_min - margin, -margin), flux_max + margin, padding=0)
                except (ValueError, RuntimeError): pass
        self._relayout_emission_labels()

    def _on_2d_range_changed(self):
        if not self._syncing_x:
            self._syncing_x = True
            self.plot_1d.setXRange(*self.plot_2d.getViewBox().viewRange()[0], padding=0)
            self._syncing_x = False

    def _on_any_range_changed(self):
        if not self._is_restoring: self._push_nav_state()

    def _push_nav_state(self):
        x1d_r, x2d_r = self.plot_1d.getViewBox().viewRange(), self.plot_2d.getViewBox().viewRange()
        state = {'x1d_x': tuple(x1d_r[0]), 'x1d_y': tuple(x1d_r[1]),
                 'x2d_x': tuple(x2d_r[0]), 'x2d_y': tuple(x2d_r[1])}
        if self._nav_index < len(self._nav_stack) - 1:
            self._nav_stack = self._nav_stack[:self._nav_index+1]
        self._nav_stack.append(state); self._nav_index = len(self._nav_stack) - 1

    def _apply_nav_state(self, idx):
        if not (0 <= idx < len(self._nav_stack)): return
        st = self._nav_stack[idx]; self._is_restoring = True
        self.plot_1d.getViewBox().setRange(x=st['x1d_x'], y=st['x1d_y'], padding=0)
        self.plot_2d.getViewBox().setRange(x=st['x2d_x'], y=st['x2d_y'], padding=0)
        self._is_restoring = False

    def nav_back(self):
        if self._nav_index > 0: self._nav_index -= 1; self._apply_nav_state(self._nav_index)

    def nav_forward(self):
        if self._nav_index < len(self._nav_stack) - 1: self._nav_index += 1; self._apply_nav_state(self._nav_index)

    def on_autoscale_toggled(self):
        if self.autoscale_y_check.isChecked():
            self.sync_x_range_to_2d(*self.plot_1d.getViewBox().viewRange()[0])

    # ----------------------- Status & Hover -----------------------
    def _basename_or_dash(self, path): return os.path.basename(path) if path else "—"

    def _rest_from_obs(self, lam_obs_um):
        z = self.z_input.value() if self.z_input else 0.0
        return (lam_obs_um / (1.0 + max(z, -0.99))) * 1e4

    def _update_status(self, obs_um=None, y=None, x=None, val2d=None):
        rest_ang_str = f"{self._rest_from_obs(obs_um):.1f} Å" if obs_um else "—"
        s2d_val_str = f"[{y:d}, {int(x):d}]: {val2d:.4g} MJy/sr" if all(v is not None for v in [y, x, val2d]) and np.isfinite(val2d) else "—"
        obs_str = f"{obs_um:.4f} μm" if obs_um else "—"
        
        flux_str = ""
        ix = self._col_from_x(x)
        if ix is not None:
            for i, src in enumerate(self.spectra_sources):
                if src['visible'] and src.get('curve') and (curve := src['curve'].yData) is not None and ix < len(curve):
                    flux = curve[ix]
                    if np.isfinite(flux):
                        label = "1D" if src['type'] == 'x1d' else f"Ext{i}"
                        flux_str += f"&nbsp;&nbsp;<font color='{src['color'].name()}'>{label}: {1e6*flux:.4g} µJy</font>"
        
        self.s2d_label.setText(f"S2D: {self._basename_or_dash(self.current_s2d_file)}   rest λ: {rest_ang_str}     2D flux[y,x]: {s2d_val_str}")
        self.x1d_label.setText(f"X1D: {self._basename_or_dash(self.current_x1d_file)} <html>&nbsp;&nbsp;</html>  obs λ: {obs_str}{flux_str}")

    def _update_status_clear(self): self._update_status()

    def _update_cursor_dots(self, x_idx=None):
        if x_idx is None: self._hide_cursor(); return
        ix = self._col_from_x(x_idx)
        if ix is None: self._hide_cursor(); return
        center_x = float(ix) + 0.5
        self.cursor_line.setPos(center_x); self.cursor_line.show()
        
        for source in self.spectra_sources:
            if not (dot := source.get('cursor_dot')): continue
            flux = np.nan
            if source['visible'] and source.get('curve') and (curve := source['curve'].yData) is not None and ix < len(curve):
                flux = curve[ix]
            if np.isfinite(flux):
                dot.setData([center_x], [flux]); dot.show()
            else: dot.hide()

    def _hide_cursor(self):
        self.cursor_line.hide()
        for source in self.spectra_sources:
            if (dot := source.get('cursor_dot')): dot.hide()

    def _col_from_x(self, x):
        nx = self._nx if self._nx else getattr(self.x1d_flux, 'size', 0)
        if nx > 0 and x is not None and np.isfinite(x):
            return int(np.clip(np.floor(float(x)), 0, nx - 1))
        return None

    def _on_1d_mouse_moved(self, pos):
        if not self.plot_1d.sceneBoundingRect().contains(pos):
            self._hide_cursor(); self._update_status_clear(); return
        x = self.plot_1d.getViewBox().mapSceneToView(pos).x()
        ix = self._col_from_x(x)
        if ix is not None and self.x1d_wave is not None and ix < len(self.x1d_wave):
            self._update_cursor_dots(x)
            self._update_status(obs_um=float(self.x1d_wave[ix]), x=x)
        else:
            self._hide_cursor(); self._update_status_clear()

    def _on_2d_mouse_moved(self, pos):
        if not self.plot_2d.sceneBoundingRect().contains(pos):
            self._hide_cursor(); self._update_status_clear(); return
        pt = self.plot_2d.getViewBox().mapSceneToView(pos)
        x, y = pt.x(), pt.y()
        ix = self._col_from_x(x)
        if ix is not None and self.x1d_wave is not None and ix < len(self.x1d_wave) and self.s2d_data is not None:
            ny = self.s2d_data.shape[0]
            if 0 <= y < ny:
                iy = int(np.clip(np.floor(y), 0, ny - 1))
                val = self.s2d_data[iy, ix]
                self._update_cursor_dots(x)
                self._update_status(obs_um=float(self.x1d_wave[ix]), y=iy, x=x, val2d=val)
                return
        self._hide_cursor(); self._update_status_clear()

    def eventFilter(self, obj, event):
        if obj in [self.plot_1d.viewport(), self.plot_2d.viewport()] and event.type() == QEvent.Leave:
            self._hide_cursor(); self._update_status_clear()
        return super().eventFilter(obj, event)

    # ------------------------------ Misc & Plotting ------------------------------
    def change_colormap(self, cmap_name):
        if cmap_name == 'RdBu':
            colors = [(0,0,180), (100,150,255), (255,255,255), (255,150,100), (180,0,0)]
            cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 5), color=colors)
        else:
            cmap = pg.colormap.get(cmap_name)
        self.image_item.setLookupTable(cmap.getLookupTable())

    def reset_views(self):
        nx = self._nx if self._nx else getattr(self.x1d_flux, 'size', None)
        if nx:
            self.plot_1d.setXRange(0.0, float(nx), padding=0)
            self.plot_2d.setXRange(0.0, float(nx), padding=0)
        if self.s2d_data is not None:
            self.plot_2d.setYRange(0, self.s2d_data.shape[0], padding=0)
        if self.x1d_flux is not None:
            flux_min, flux_max = np.nanmin(self.x1d_flux), np.nanmax(self.x1d_flux)
            margin = (flux_max - flux_min) * 0.1 if flux_max > flux_min else 1.0
            self.plot_1d.setYRange(min(flux_min - margin, -margin), flux_max + margin, padding=0)
        self._push_nav_state()
        self._hide_cursor(); self._update_status_clear()
        self._relayout_emission_labels()

    def autoscale(self): self.reset_views()

    def _refresh_top_axis(self, *_):
        if self.top_axis: self.top_axis.picture = None; self.top_axis.update()
        if self.obs_axis: self.obs_axis.picture = None; self.obs_axis.update()

    def _parse_emission_lines_file(self, path):
        lines = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith('#') or ('name' in s.lower() and ('wave' in s.lower() or 'λ' in s.lower())):
                        continue
                    parts = s.split('\t') if '\t' in s else s.rsplit(None, 1)
                    if len(parts) >= 2:
                        try: lines.append((parts[0].strip(), float(parts[-1])))
                        except ValueError: continue
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to read lines file:\n{e}')
            return None
        if not lines: QMessageBox.warning(self, 'Empty file', 'No emission lines found in the file.')
        return lines or None

    def load_emission_lines_file(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load Emission Lines', '', 'Text Files (*.txt *.dat *.tsv *.csv);;All Files (*.*)')
        if path and (parsed := self._parse_emission_lines_file(path)):
            self.emission_lines = parsed
            self._update_emission_lines()

    def _clear_emission_overlays(self):
        for item in self.em_line_items_1d + self.em_label_items_1d + self.em_line_items_2d:
            if item.scene(): item.scene().removeItem(item)
        self.em_line_items_1d, self.em_label_items_1d, self.em_line_items_2d = [], [], []

    def _update_emission_lines(self):
        self._clear_emission_overlays()
        if not (self.show_lines_check.isChecked() and self.emission_lines and self._wave_to_idx and self._nx):
            return
        z = max(self.z_input.value(), -0.99)
        pen = pg.mkPen('gray', width=0.5, style=Qt.DashLine)
        
        for name, rest_A in self.emission_lines:
            try:
                obs_um = (rest_A * (1.0 + z)) / 1e4
                idx = float(self._wave_to_idx(obs_um))
                if 0.0 < idx < float(self._nx) - 1:
                    line = pg.InfiniteLine(pos=idx, angle=90, pen=pen); self.plot_1d.addItem(line)
                    self.em_line_items_1d.append(line)
                    if self.s2d_data is not None:
                        line2d = pg.InfiniteLine(pos=idx, angle=90, pen=pen); self.plot_2d.addItem(line2d)
                        self.em_line_items_2d.append(line2d)
                    
                    html = f"<span style='color:black; font-size:10pt;'>{name}</span>"
                    label = pg.TextItem(html=html, anchor=(0.5, 1.0)); self.plot_1d.addItem(label)
                    label.setPos(idx, 0.0); self.em_label_items_1d.append(label)
            except Exception: continue
        self._relayout_emission_labels()

    def _relayout_emission_labels(self, *args):
        if not self.em_label_items_1d: return
        vb = self.plot_1d.getViewBox()
        (xmin, xmax), (ymin, ymax) = vb.viewRange()
        px_w = max(self.plot_1d.viewport().width(), 1)
        yr = max(ymax - ymin, 1e-9)
        
        visible_labels = []
        for ti in self.em_label_items_1d:
            x = ti.pos().x()
            is_visible = xmin < x < xmax
            ti.setVisible(is_visible)
            if is_visible:
                px = (x - xmin) / max(xmax - xmin, 1e-9) * px_w
                visible_labels.append({'px': px, 'item': ti, 'x': x})
        if not visible_labels: return
        
        visible_labels.sort(key=lambda d: d['px'])
        min_sep_px, levels_px = 60.0, []
        base_y, step = ymax - 0.06 * yr, 0.055 * yr
        
        for lbl in visible_labels:
            level, placed = 0, False
            for i, last_px in enumerate(levels_px):
                if abs(lbl['px'] - last_px) >= min_sep_px:
                    level, levels_px[i], placed = i, lbl['px'], True; break
            if not placed:
                levels_px.append(lbl['px']); level = len(levels_px) - 1
            lbl['item'].setPos(float(lbl['x']), float(base_y - level * step))

    # --------------------------- Save/Load State & Plots ----------------------------
    def _save_state(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save View State", "spectacle.json", "JSON Files (*.json)")
        if not path: return
        try:
            x1d_r, x2d_r = self.plot_1d.getViewBox().viewRange(), self.plot_2d.getViewBox().viewRange()
            state = {
                'files': {'s2d': self.current_s2d_file, 'x1d': self.current_x1d_file},
                'redshift': self.z_input.value(), 'sigma_clip': self.sigma_spin.value(),
                'colormap': self.cmap_combo.currentText(), 'show_errors': self.show_errors_check.isChecked(),
                'autoscale_y': self.autoscale_y_check.isChecked(), 'show_lines': self.show_lines_check.isChecked(),
                'view_ranges': {'1d': {'x': list(x1d_r[0]), 'y': list(x1d_r[1])},
                                '2d': {'x': list(x2d_r[0]), 'y': list(x2d_r[1])}},
                'spectra_sources': [{'type': s['type'], 'visible': s['checkbox'].isChecked(), 'color': s['color'].name(),
                                     **({'rows': [s['spin_start'].value(), s['spin_stop'].value()],
                                         'width': s['spin_width'].value()} if s['type'] == 'manual' else {})}
                                    for s in self.spectra_sources]}
            with open(path, 'w') as f: json.dump(state, f, indent=2)
            self.status_bar.showMessage(f"State saved to {os.path.basename(path)}", 3000)
        except Exception as e: QMessageBox.critical(self, "Error Saving State", f"Could not save state:\n{e}")

    def _open_state_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open View State", "", "JSON Files (*.json)")
        if not path: return
        try:
            with open(path, 'r') as f: state = json.load(f)
            file_to_load = state.get('files', {}).get('x1d') or state.get('files', {}).get('s2d')
            if not (file_to_load and os.path.exists(file_to_load)):
                raise ValueError("No valid FITS file path found in the state file.")
            self.load_fits_pair(file_to_load)
            self.load_state(state)
        except Exception as e: QMessageBox.critical(self, "Error Opening State", f"Could not open state file:\n{e}")

    def load_state(self, state):
        self.z_input.setValue(state.get('redshift', 0.0))
        self.sigma_spin.setValue(state.get('sigma_clip', 5))
        self.cmap_combo.setCurrentText(state.get('colormap', 'RdBu'))
        self.show_errors_check.setChecked(state.get('show_errors', True))
        self.autoscale_y_check.setChecked(state.get('autoscale_y', False))
        self.show_lines_check.setChecked(state.get('show_lines', True))

        for i, src_state in enumerate(state.get('spectra_sources', [])):
            if i < len(self.spectra_sources):
                src = self.spectra_sources[i]
                color = QColor(src_state.get('color', '#000000'))
                src['color'] = color
                self._update_swatch_color(src['color_swatch'], color)
                if src['type'] == 'manual':
                    src['spin_width'].setValue(src_state.get('width', 6))
                    rows = src_state.get('rows', [0, 5])
                    self._set_manual_region_rows(i, rows[0], rows[1])
                src['checkbox'].setChecked(src_state.get('visible', True))

        self._is_restoring = True
        if vr := state.get('view_ranges'):
            if '1d' in vr: self.plot_1d.setRange(xRange=vr['1d']['x'], yRange=vr['1d']['y'], padding=0)
            if '2d' in vr: self.plot_2d.setRange(xRange=vr['2d']['x'], yRange=vr['2d']['y'], padding=0)
        self._is_restoring = False
        
        self.update_display(); self._update_emission_lines()
        self.status_bar.showMessage(f"Loaded state successfully", 3000)

    def _save_plot(self):
        if self.plot_1d is None: return
        dialog = SavePlotDialog(parent=self)
        if self.s2d_data is None:
            dialog.radio_2d.setEnabled(False); dialog.radio_both.setEnabled(False)
            dialog.radio_1d.setChecked(True)

        if dialog.exec_() == QDialog.Accepted:
            options = dialog.get_options()
            base_path, _ = os.path.splitext(options['filename'])
            
            try:
                items = {'1d': [self.plot_1d.getPlotItem()], '2d': [self.plot_2d.getPlotItem()],
                         'both': [self.plot_2d.getPlotItem(), self.plot_1d.getPlotItem()]}[options['plot_choice']]
                if options['save_png']: self._render_items_to_image(items).save(f"{base_path}.png")
                if options['save_pdf']: self._save_items_as_pdf(items, f"{base_path}.pdf")
                self.status_bar.showMessage(f"Plot saved to {os.path.basename(base_path)}.*", 3000)
            except Exception as e: QMessageBox.critical(self, "Error Saving Plot", f"Could not save the plot:\n{e}")

    def _render_items_to_image(self, plot_items, width=1200):
        heights = [width * (item.boundingRect().height() / item.boundingRect().width() if item.boundingRect().width() > 0 else 0.25) for item in plot_items]
        total_height = int(sum(heights))
        image = QImage(width, total_height, QImage.Format_ARGB32); image.fill(Qt.white)
        painter = QPainter(image); painter.setRenderHint(QPainter.Antialiasing)
        y_offset = 0
        for i, item in enumerate(plot_items):
            target_rect = pg.QtCore.QRectF(0, y_offset, width, int(heights[i]))
            item.scene().render(painter, target_rect, item.boundingRect())
            y_offset += int(heights[i])
        painter.end()
        return image


    def _save_items_as_pdf(self, plot_items, filename, width=1200):
        """Saves one or more plot items to a single vectorized PDF file."""
        if not QTSVG_AVAILABLE:
            QMessageBox.critical(self, "Dependency Missing",
                                 "Saving to PDF requires the PyQt5.QtSvg module, which could not be imported.")
            return

        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(filename)

        heights = []
        for item in plot_items:
            br = item.boundingRect()
            aspect = (br.height() / br.width()) if br.width() > 0 else 0.25
            heights.append(width * aspect)

        total_height = sum(heights)

        page_size_points = pg.QtCore.QSizeF(width, total_height)
        page_size = QPageSize(page_size_points, QPageSize.Point, "Custom", QPageSize.ExactMatch)
        printer.setPageSize(page_size)
        printer.setPageMargins(0.0, 0.0, 0.0, 0.0, QPrinter.Point)

        painter = QPainter(printer)
        page_rect_pixels = printer.pageRect(QPrinter.DevicePixel)
        painter.setViewport(page_rect_pixels.toRect())
        painter.setWindow(0, 0, width, int(total_height))

        y_offset = 0
        for i, item in enumerate(plot_items):
            target_rect = pg.QtCore.QRectF(0, y_offset, width, heights[i])
            original_pens = {}

            try:
                # --- Export based on plot type ---
                if item == self.plot_2d.getPlotItem():
                    # For the 2D plot, render manually to a QImage for robustness
                    img_width = width * 2 
                    img_height = int(heights[i] * 2)
                    
                    qimage = QImage(img_width, img_height, QImage.Format_ARGB32)
                    qimage.fill(Qt.white)
                    
                    img_painter = QPainter(qimage)
                    img_painter.setRenderHint(QPainter.Antialiasing)
                    source_rect = item.boundingRect()
                    target_img_rect = pg.QtCore.QRectF(0, 0, img_width, img_height)
                    item.scene().render(img_painter, target_img_rect, source_rect)
                    img_painter.end()

                    if qimage:
                        painter.drawImage(target_rect, qimage)
                else:
                    # For the 1D plot, thicken lines and use vector SVGExporter
                    items_to_thicken = [pi for pi in item.items if isinstance(pi, (pg.PlotCurveItem, pg.InfiniteLine))]
                    for line_item in items_to_thicken:
                        pen = None
                        if isinstance(line_item, pg.PlotCurveItem):
                            pen = line_item.opts.get('pen')
                        elif isinstance(line_item, pg.InfiniteLine):
                            pen = line_item.pen

                        if pen:
                            original_pens[line_item] = pen
                            new_pen = pg.mkPen(pen)
                            # --- START OF CORRECTION ---
                            # Be more aggressive with thickening for clear visibility in PDFs
                            new_width = max(new_pen.widthF() * 4.0, 2.5)
                            # --- END OF CORRECTION ---
                            new_pen.setWidthF(new_width)
                            line_item.setPen(new_pen)

                    exporter = pg.exporters.SVGExporter(item)
                    svg_bytes = exporter.export(toBytes=True)
                    renderer = QtSvg.QSvgRenderer(svg_bytes)
                    renderer.render(painter, target_rect)
            finally:
                # --- Restore on-screen appearance ---
                for line_item, pen in original_pens.items():
                    line_item.setPen(pen)
            
            y_offset += heights[i]

        painter.end()

# ---------------------------- HELPER DIALOGS & MAIN --------------------------------
class SavePlotDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Plot Options")
        layout = QVBoxLayout(self)
        
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("File Name:"))
        self.filename_edit = QLineEdit("spectrum_plot")
        file_layout.addWidget(self.filename_edit)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse)
        file_layout.addWidget(browse_btn)
        layout.addLayout(file_layout)
        
        plot_group = QGroupBox("Content to Save")
        plot_layout = QVBoxLayout()
        self.radio_both = QRadioButton("Save Both (Combined)")
        self.radio_2d = QRadioButton("Save 2D Plot Only")
        self.radio_1d = QRadioButton("Save 1D Plot Only")
        self.radio_both.setChecked(True)
        plot_layout.addWidget(self.radio_both)
        plot_layout.addWidget(self.radio_2d)
        plot_layout.addWidget(self.radio_1d)
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)
        
        format_group = QGroupBox("Output Format(s)")
        format_layout = QHBoxLayout()
        self.check_png = QCheckBox("PNG (raster)")
        self.check_pdf = QCheckBox("PDF (vector)")
        self.check_png.setChecked(True)
        self.check_pdf.setChecked(True)
        format_layout.addWidget(self.check_png)
        format_layout.addWidget(self.check_pdf)
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self):
        """Override to check for existing files before closing."""
        filename = self.filename_edit.text()
        if not filename:
            QMessageBox.warning(self, "No Filename", "Please enter a file name.")
            return

        base, _ = os.path.splitext(filename)
        png_path = f"{base}.png"
        pdf_path = f"{base}.pdf"

        # Check for which files will actually be written
        png_exists = self.check_png.isChecked() and os.path.exists(png_path)
        pdf_exists = self.check_pdf.isChecked() and os.path.exists(pdf_path)

        if png_exists or pdf_exists:
            files_to_clobber = []
            if png_exists: files_to_clobber.append(os.path.basename(png_path))
            if pdf_exists: files_to_clobber.append(os.path.basename(pdf_path))
            
            reply = QMessageBox.question(self, 'File Exists',
                                         f"The file(s) '{', '.join(files_to_clobber)}' already exist.\nDo you want to overwrite?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.No:
                return  # Stop and leave the dialog open

        super().accept()  # Proceed to close the dialog

    def browse(self):
        path, _ = QFileDialog.getSaveFileName(self, "Select Output File", self.filename_edit.text(), "All Files (*)")
        if path:
            self.filename_edit.setText(path)

    def get_options(self):
        plot_choice = '1d' if self.radio_1d.isChecked() else '2d' if self.radio_2d.isChecked() else 'both'
        return {'filename': self.filename_edit.text(), 'plot_choice': plot_choice,
                'save_png': self.check_png.isChecked(), 'save_pdf': self.check_pdf.isChecked()}


def main():
    parser = argparse.ArgumentParser(description="FITS Spectra Viewer")
    parser.add_argument("input_file", nargs='?', default=None, help="Optional: Path to a FITS file (s2d or x1d) or a .json state file")
    parser.add_argument("-z", "--redshift", type=float, help="Initial redshift (only used with FITS files)")
    parser.add_argument("--lines", type=str, help="Path to a custom emission lines file (only used with FITS files)")
    parser.add_argument("--annotate", type=str, help="Path to a Python script with an annotate_plot(viewer) function")
    args = parser.parse_args()

    app = QApplication(sys.argv); app.setStyle('Fusion')
    win = FITSSpectraViewer()

    if args.input_file:
        try:
            if args.input_file.lower().endswith('.json'):
                with open(args.input_file, 'r') as f: state = json.load(f)
                file_to_load = state.get('files', {}).get('x1d') or state.get('files', {}).get('s2d')
                if not (file_to_load and os.path.exists(file_to_load)):
                    raise ValueError("No valid FITS file path found in state file.")
                win.load_fits_pair(file_to_load); win.load_state(state)
            else:
                win.load_fits_pair(args.input_file)
                if args.redshift is not None: win.z_input.setValue(args.redshift)
                if args.lines and (parsed := win._parse_emission_lines_file(args.lines)):
                    win.emission_lines = parsed; win._update_emission_lines()
            
            if args.annotate:
                if not os.path.exists(args.annotate): raise FileNotFoundError(f"Annotation script not found: {args.annotate}")
                spec = importlib.util.spec_from_file_location("annotation_module", args.annotate)
                anno_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(anno_module)
                if hasattr(anno_module, 'annotate_plot') and callable(anno_module.annotate_plot):
                    anno_module.annotate_plot(win)
                else: raise AttributeError("Annotation script must contain a function 'annotate_plot(viewer)'.")

        except Exception as e:
            QMessageBox.critical(win, 'Error', f'Failed during startup:\n{e}')

    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()