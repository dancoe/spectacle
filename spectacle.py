#!/usr/bin/env python3
"""
FITS Spectra Viewer with PyQt
Trackpad gestures:
- Zoom x (wavelength): 2 fingers up/down
- Pan x (wavelength): 2 fingers left/right
- Scale y (flux): 2 fingers pinch
Update (2025-09-08):
- **Pixel-perfect wavelength alignment** of 2D and 1D:
  We now render both plots in **index space** (x = spectral column index),
  and only the tick labels show wavelength. Each 1D horizontal step spans
  **exactly one 2D pixel column**, matching the method from `show_mos_spectrum.py`.

Update (2025-09-08-b):
- **Emission line overlays (rest-frame)**: Load a simple text file of rest-frame
  wavelengths to draw vertical markers and labels. Labels auto-deconflict by
  bumping down slightly to avoid overlap as you zoom.

  File format (whitespace or tab between columns; header optional):

      name    wavelength
      Lya    1215.6700
      [O II]    3727.0635
      [O II]    3729.8472
      HB    4862.6739
      [O III]    4960.2785
      [O III]    5008.2236
      Ha    6564.6237

  Notes:
  - Wavelengths are **rest-frame Angstroms (Å)**. They are converted to observed
    wavelengths using the current redshift *z* (observed = rest × (1+z)).
  - Markers are shown on both 1D and 2D panels; labels are drawn on the 1D panel.
  - Use **Display Controls → Load Lines…** to load a custom file. If not provided,
    the defaults above are used automatically.
"""
import sys
import os
import argparse
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QLabel,
                             QSplitter, QStatusBar, QToolBar, QAction,
                             QComboBox, QSpinBox, QCheckBox, QGroupBox,
                             QSlider, QMessageBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QPointF, pyqtSignal, pyqtSlot, QEvent
from PyQt5.QtGui import QKeySequence
import pyqtgraph as pg
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.visualization import simple_norm
import warnings
warnings.filterwarnings('ignore')
pg.setConfigOptions(antialias=True, imageAxisOrder='row-major')
min_bins_default = 5  # minimum zoom in x = wavelength (bins in index space)
min_y_rows_default = 5  # minimum zoom in y for 2D rows

# ------------------------- DEFAULT EMISSION LINES (rest Å) -------------------------
# These are used when no external file is loaded. See the header doc for format.
DEFAULT_EMISSION_LINES = [
    ("Lya", 1215.6700),
    ("[O II]", 3727.0635),
    ("[O II]", 3729.8472),
    ("HB", 4862.6739),
    ("[O III]", 4960.2785),
    ("[O III]", 5008.2236),
    ("Ha", 6564.6237),
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
        self.setLabel("Wavelength (µm)")  # , units='μm' likes to say (kµm)
        self._idx_to_wave = idx_to_wave  # callable: idx(float) -> μm
        self._wave_to_idx = wave_to_idx  # callable: μm -> idx(float)

    def set_mappers(self, idx_to_wave, wave_to_idx):
        self._idx_to_wave = idx_to_wave
        self._wave_to_idx = wave_to_idx
        # force a refresh
        self.picture = None
        self.update()

    def tickValues(self, minVal, maxVal, size):
        if self._idx_to_wave is None or self._wave_to_idx is None:
            return super().tickValues(minVal, maxVal, size)
        # Map index range -> observed wavelength (μm)
        wmin = float(self._idx_to_wave(minVal))
        wmax = float(self._idx_to_wave(maxVal))
        if wmax < wmin:
            wmin, wmax = wmax, wmin
        rng = max(wmax - wmin, 1e-12)
        approxNTicks = max(int(size / 90.0), 2)  # target ~90 px per major tick
        step = self._nice_step(rng / approxNTicks)
        first = np.ceil(wmin / step) * step
        majors_wave = []
        v = first
        for _ in range(2000):
            if v > wmax + 1e-9:
                break
            majors_wave.append(v)
            v += step
        minor_step = step / 5.0
        first_m = np.ceil(wmin / minor_step) * minor_step
        minors_wave = []
        v = first_m
        for _ in range(10000):
            if v > wmax + 1e-9:
                break
            if (abs((v / step) - np.round(v / step)) > 1e-6):
                minors_wave.append(v)
            v += minor_step
        # Map to index positions for the view
        majors_idx = np.array([float(self._wave_to_idx(w)) for w in majors_wave])
        minors_idx = np.array([float(self._wave_to_idx(w)) for w in minors_wave])
        # Report spacing approximately in index units
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
        out = []
        delta_wave = waves[1] - waves[0]
        for w in waves:
            if delta_wave >= 0.1:
                s = f"{w:.1f}"
            elif delta_wave >= 0.01:
                s = f"{w:.2f}"
            else:
                s = f"{w:.3f}"
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
        self.setLabel("Rest Wavelength (Å)")  # , units='Å' likes to say (kÅ)

    def set_mappers(self, idx_to_wave, wave_to_idx):
        self._idx_to_wave = idx_to_wave
        self._wave_to_idx = wave_to_idx
        self.picture = None
        self.update()

    def tickValues(self, minVal, maxVal, size):
        if self._idx_to_wave is None or self._wave_to_idx is None:
            return []
        try:
            z = float(self._get_z()) if callable(self._get_z) else float(self._get_z)
        except Exception:
            z = 0.0
        z = max(z, -0.99)  # avoid 1+z <= 0
        # Index -> observed μm -> rest Å range
        wmin = float(self._idx_to_wave(minVal))
        wmax = float(self._idx_to_wave(maxVal))
        if wmax < wmin:
            wmin, wmax = wmax, wmin
        rmin = (wmin / (1.0 + z)) * 1e4
        rmax = (wmax / (1.0 + z)) * 1e4
        rng = max(rmax - rmin, 1e-12)
        approxNTicks = max(int(size / 90.0), 2)
        step = self._nice_step(rng / approxNTicks)
        first = np.ceil(rmin / step) * step
        majors_rest = []
        v = first
        for _ in range(2000):
            if v > rmax + 1e-9:
                break
            majors_rest.append(v)
            v += step
        minor_step = step / 5.0
        first_m = np.ceil(rmin / minor_step) * minor_step
        minors_rest = []
        v = first_m
        for _ in range(10000):
            if v > rmax + 1e-9:
                break
            if (abs((v / step) - np.round(v / step)) > 1e-6):
                minors_rest.append(v)
            v += minor_step
        # Map rest Å -> observed μm -> index
        majors_obs = (np.array(majors_rest) * (1.0 + z)) / 1e4
        minors_obs = (np.array(minors_rest) * (1.0 + z)) / 1e4
        majors_idx = np.array([float(self._wave_to_idx(w)) for w in majors_obs])
        minors_idx = np.array([float(self._wave_to_idx(w)) for w in minors_obs])
        # Spacing in index units (approx)
        try:
            dx = float(self._wave_to_idx(wmin + (step / 1e4) * (1.0 + z)) - self._wave_to_idx(wmin))
            dx_m = float(self._wave_to_idx(wmin + (minor_step / 1e4) * (1.0 + z)) - self._wave_to_idx(wmin))
        except Exception:
            dx, dx_m = step, minor_step
        return [ (dx, majors_idx), (dx_m, minors_idx) ]

    def tickStrings(self, values, scale, spacing):
        if self._idx_to_wave is None:
            return []
        try:
            z = float(self._get_z()) if callable(self._get_z) else float(self._get_z)
        except Exception:
            z = 0.0
        z = max(z, -0.99)
        waves = np.array([float(self._idx_to_wave(v)) for v in values])  # μm
        rest = (waves / (1.0 + z)) * 1e4  # Å
        out = []
        for v in rest:
            av = abs(v)
            if av >= 1000:
                s = f"{v:.0f}"
            elif av >= 100:
                s = f"{v:.1f}"
            else:
                s = f"{v:.2f}"
            out.append(s)
        return out

# ------------------------------- MAIN WIDGETS --------------------------------
class SpectrumPlotWidget(pg.PlotWidget):
    """Custom plot widget with trackpad gesture support"""
    # Signal emitted when x range changes (1D plot only)
    x_range_changed = pyqtSignal(float, float)
    # Signal emitted when Alt-drag selects a Y range (ymin, ymax)
    alt_selected = pyqtSignal(float, float)

    def __init__(self, parent=None, is_2d=False):
        super().__init__(parent)
        self.is_2d = is_2d
        if not is_2d:
            self.setLabel('left', 'Flux', units='Jy')
        else:
            self.setLabel('left', 'Pixel Row')
        # NOTE: bottom axis label is handled by a custom AxisItem for 1D plot
        self.showGrid(x=True, y=True, alpha=0.3)
        # Enable mouse tracking and gestures
        self.viewport().setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.setMouseTracking(True)
        try:
            self.grabGesture(Qt.PinchGesture)  # native pinch for Y-zoom
        except Exception:
            pass
        # Data limits
        self.data_x_min = None
        self.data_x_max = None
        self.data_y_min = None
        self.data_y_max = None
        self.min_x_range = None  # minimum allowed x-range (~few bins)
        self.min_y_rows = None  # minimum allowed y-range in rows (2D only)
        # Track mouse position for zoom centering
        self.mouse_x_pos = None
        self.mouse_y_pos = None
        # Alt-drag state
        self._alt_drag_start = None
        self._alt_region = None
        # Connect range change signal
        self.getViewBox().sigRangeChanged.connect(self.on_range_changed)

    def set_data_limits(self, x_min, x_max, y_min=None, y_max=None,
                         min_dx=None, min_bins=min_bins_default, min_y_rows=None):
        """Set the data limits to restrict panning/zooming"""
        self.data_x_min = x_min
        self.data_x_max = x_max
        self.data_y_min = y_min
        self.data_y_max = y_max
        if min_dx is not None:
            self.min_x_range = float(min_dx) * max(int(min_bins), 1)
        self.min_y_rows = int(min_y_rows) if (min_y_rows is not None) else None
        # Apply hard limits at the ViewBox level so panning can't escape
        vb = self.getViewBox()
        lim_kwargs = {}
        if x_min is not None:
            lim_kwargs['xMin'] = x_min
        if x_max is not None:
            lim_kwargs['xMax'] = x_max
        if y_min is not None:
            lim_kwargs['yMin'] = y_min
        if y_max is not None:
            lim_kwargs['yMax'] = y_max
        if self.min_x_range is not None:
            lim_kwargs['minXRange'] = self.min_x_range
        if self.min_y_rows is not None:
            lim_kwargs['minYRange'] = float(self.min_y_rows)
        try:
            vb.setLimits(**lim_kwargs)
        except Exception:
            pass

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
                ymin = min(p1.y(), p2.y())
                ymax = max(p1.y(), p2.y())
                self._alt_region.setRegion((ymin, ymax))
        super().mouseMoveEvent(ev)

    def mousePressEvent(self, ev):
        # Alt/Option + Left-drag => select a Y interval (visual highlight)
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.AltModifier and ev.button() == Qt.LeftButton:
            if self.sceneBoundingRect().contains(ev.pos()):
                self._alt_drag_start = ev.pos()
                try:
                    self._alt_region = pg.LinearRegionItem(
                        values=(0, 0),
                        orientation='horizontal',
                        brush=pg.mkBrush(255, 255, 0, 60),
                        pen=pg.mkPen('y', width=1),
                    )
                    self.addItem(self._alt_region)
                except Exception:
                    self._alt_region = None
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseDragEvent(self, ev):
        """Default pan, then clamp to data limits and emit x_range_changed."""
        try:
            super().mouseDragEvent(ev)
        except Exception:
            pass
        # Clamp after drag
        try:
            vb = self.getViewBox()
            if self.data_x_min is not None and self.data_x_max is not None:
                xmin, xmax = vb.viewRange()[0]
                xmin = max(xmin, self.data_x_min)
                xmax = min(xmax, self.data_x_max)
                if self.min_x_range is not None and (xmax - xmin) < self.min_x_range:
                    cx = 0.5 * (xmax + xmin)
                    xmin = max(self.data_x_min, cx - 0.5 * self.min_x_range)
                    xmax = min(self.data_x_max, cx + 0.5 * self.min_x_range)
                vb.setXRange(xmin, xmax, padding=0)
            if self.data_y_min is not None and self.data_y_max is not None:
                ymin, ymax = vb.viewRange()[1]
                ymin = max(ymin, self.data_y_min)
                ymax = min(ymax, self.data_y_max)
                vb.setYRange(ymin, ymax, padding=0)
            xmin, xmax = vb.viewRange()[0]
            if not self.is_2d:
                self.x_range_changed.emit(xmin, xmax)
        except Exception:
            pass

    def mouseReleaseEvent(self, ev):
        if self._alt_drag_start is not None:
            start = self._alt_drag_start
            end = ev.pos()
            if start != end:
                vb = self.getViewBox()
                p1 = vb.mapSceneToView(start)
                p2 = vb.mapSceneToView(end)
                ymin = min(p1.y(), p2.y())
                ymax = max(p1.y(), p2.y())
                try:
                    self.alt_selected.emit(ymin, ymax)
                except Exception:
                    pass
            # remove visual region
            try:
                if self._alt_region is not None:
                    self.removeItem(self._alt_region)
            except Exception:
                pass
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
        dx = delta.x() / 120.0  # Horizontal scroll
        dy = delta.y() / 120.0  # Vertical scroll
        vb = self.getViewBox()
        xmin, xmax = vb.viewRange()[0]
        ymin, ymax = vb.viewRange()[1]
        view_h = max(self.viewport().height(), 1)
        if modifiers == Qt.NoModifier:
            # Up/down => Zoom X (both 1D and 2D), slower sensitivity
            if abs(dy) > 0:
                scale_factor = 1.02 ** (-dy)  # gentle zoom
                if self.mouse_x_pos is not None:
                    x_center = self.mouse_x_pos
                    left_ratio = (x_center - xmin) / (xmax - xmin)
                    right_ratio = 1 - left_ratio
                    new_x_range = (xmax - xmin) * scale_factor
                    new_xmin = x_center - new_x_range * left_ratio
                    new_xmax = x_center + new_x_range * right_ratio
                else:
                    x_center = 0.5 * (xmin + xmax)
                    new_x_range = (xmax - xmin) * scale_factor
                    new_xmin = x_center - new_x_range / 2
                    new_xmax = x_center + new_x_range / 2
                if self.data_x_min is not None and self.data_x_max is not None:
                    new_xmin = max(new_xmin, self.data_x_min)
                    new_xmax = min(new_xmax, self.data_x_max)
                if self.min_x_range is not None and (new_xmax - new_xmin) < self.min_x_range:
                    cx = 0.5 * (new_xmin + new_xmax)
                    new_xmin = max(self.data_x_min, cx - 0.5 * self.min_x_range)
                    new_xmax = min(self.data_x_max, cx + 0.5 * self.min_x_range)
                vb.setXRange(new_xmin, new_xmax, padding=0)
            # Left/right => Pan X (both 1D and 2D)
            if abs(dx) > 0:
                x_shift = (xmax - xmin) * dx * 0.1
                new_xmin = xmin + x_shift
                new_xmax = xmax + x_shift
                if self.data_x_min is not None and self.data_x_max is not None:
                    if new_xmin < self.data_x_min:
                        shift = self.data_x_min - new_xmin
                        new_xmin += shift
                        new_xmax += shift
                    if new_xmax > self.data_x_max:
                        shift = new_xmax - self.data_x_max
                        new_xmin -= shift
                        new_xmax -= shift
                    new_xmin = max(new_xmin, self.data_x_min)
                    new_xmax = min(new_xmax, self.data_x_max)
                vb.setXRange(new_xmin, new_xmax, padding=0)
        elif modifiers == Qt.ControlModifier:
            # Ctrl + Scroll => Scale Y — both 1D & 2D, cursor-anchored
            if abs(dy) > 0:
                scale_factor = 1.02 ** (-dy)  # gentle
                y_cursor = self.mouse_y_pos if self.mouse_y_pos is not None else 0.5 * (ymin + ymax)
                y_range = ymax - ymin
                new_y_range = y_range * scale_factor
                # Min 3 screen px in data units
                px2data = y_range / view_h
                min_range = max(3.0 * px2data, 1e-12)
                if new_y_range < min_range:
                    new_y_range = min_range
                # Keep cursor value fixed in screen by preserving its relative fraction
                frac = 0.0 if y_range == 0 else (y_cursor - ymin) / y_range
                new_ymin = y_cursor - frac * new_y_range
                new_ymax = new_ymin + new_y_range
                # Clamp to data limits if present
                if self.data_y_min is not None and self.data_y_max is not None:
                    if new_ymin < self.data_y_min:
                        new_ymin = self.data_y_min
                        new_ymax = new_ymin + new_y_range
                    if new_ymax > self.data_y_max:
                        new_ymax = self.data_y_max
                        new_ymin = new_ymax - new_y_range
                vb.setYRange(new_ymin, new_ymax, padding=0)
        ev.accept()

    def on_range_changed(self):
        """Emit signal when x range changes (1D plot)."""
        if not self.is_2d:
            xmin, xmax = self.getViewBox().viewRange()[0]
            self.x_range_changed.emit(xmin, xmax)

    def constrain_x_range(self):
        """Constrain x range to data limits."""
        if self.data_x_min is not None and self.data_x_max is not None:
            xmin, xmax = self.getViewBox().viewRange()[0]
            xmin = max(xmin, self.data_x_min)
            xmax = min(xmax, self.data_x_max)
            self.getViewBox().setXRange(xmin, xmax, padding=0)

    # Native gesture handling: pinch to scale Y only, cursor-anchored, min 3 px
    def event(self, ev):
        if ev.type() == QEvent.Gesture:
            g = ev.gesture(Qt.PinchGesture)
            if g is not None:
                try:
                    scale = g.scaleFactor()
                    vb = self.getViewBox()
                    (xmin, xmax), (ymin, ymax) = vb.viewRange()
                    view_h = max(self.viewport().height(), 1)
                    # Pinch center in scene coords -> view coords
                    center_scene = g.centerPoint()
                    if center_scene is not None:
                        center_view = vb.mapSceneToView(center_scene)
                        y_cursor = center_view.y()
                    else:
                        y_cursor = 0.5 * (ymin + ymax)
                    # Compute new Y range (inverse scaling)
                    new_range = (ymax - ymin) / max(scale, 1e-6)
                    # Enforce min 3 px in data units
                    px2data = (ymax - ymin) / view_h
                    min_range = max(3.0 * px2data, 1e-12)
                    if new_range < min_range:
                        new_range = min_range
                    # Preserve the cursor's relative fraction in screen
                    frac = 0.0 if (ymax - ymin) == 0 else (y_cursor - ymin) / (ymax - ymin)
                    new_ymin = y_cursor - frac * new_range
                    new_ymax = new_ymin + new_range
                    # Clamp to data limits
                    if self.data_y_min is not None and self.data_y_max is not None:
                        if new_ymin < self.data_y_min:
                            new_ymin = self.data_y_min
                            new_ymax = new_ymin + new_range
                        if new_ymax > self.data_y_max:
                            new_ymax = self.data_y_max
                            new_ymin = new_ymax - new_range
                    vb.setYRange(new_ymin, new_ymax, padding=0)
                    ev.accept()
                    return True
                except Exception:
                    pass
        return super().event(ev)

class FITSSpectraViewer(QMainWindow):
    """Main application window for FITS spectra viewing"""
    def __init__(self):
        super().__init__()
        self.s2d_data = None
        self.x1d_wave = None
        self.x1d_flux = None
        self.x1d_fluxerr = None
        self.current_s2d_file = None
        self.current_x1d_file = None
        self.splitter = None
        self.s2d_controls_widget = None
        # default redshift for rest-frame display
        self.z_input = None
        # first-time draw flag
        self._have_plotted = False
        # navigation stacks (Back/Forward)
        self._nav_stack = []
        self._nav_index = -1
        self._is_restoring = False
        # sync guard
        self._syncing_x = False
        # For cursor tracking
        self.cursor_line = None
        self.cursor_dot = None
        # Two-row status (labels)
        self.s2d_label = None
        self.x1d_label = None
        # Mapping functions for axes (index <-> wavelength)
        self._idx_to_wave = None  # callable idx->μm
        self._wave_to_idx = None  # callable μm->idx
        # number of spectral columns
        self._nx = None
        # -------- Emission line state --------
        self.emission_lines = list(DEFAULT_EMISSION_LINES)  # list of (name, rest_A)
        self.em_line_items_1d = []  # InfiniteLine items on 1D
        self.em_label_items_1d = []  # TextItem labels on 1D
        self.em_line_items_2d = []  # InfiniteLine items on 2D
        self.show_lines_check = None
        self.load_lines_btn = None

        self.init_ui()

    # --------------------------- Alignment helpers ---------------------------
    def _build_index_wavelength_mappers(self):
        """Create callable mappers index<->wavelength given current x1d_wave."""
        if self.x1d_wave is None:
            self._idx_to_wave = None
            self._wave_to_idx = None
            self._nx = None
            return
        w = np.asarray(self.x1d_wave, dtype=float)
        # Ensure monotonic increasing for interpolation
        # If not monotonic, fall back to argsort mapping
        if not np.all(np.diff(w[np.isfinite(w)]) >= 0):
            order = np.argsort(w)
            w_sorted = w[order]
            idx_sorted = np.arange(w.size, dtype=float)[order]
            def idx_to_wave(i):
                return np.interp(i, idx_sorted, w_sorted)
            def wave_to_idx(x):
                return np.interp(x, w_sorted, idx_sorted)
        else:
            def idx_to_wave(i):
                # clamp to valid domain for interp stability
                return np.interp(i, np.arange(w.size, dtype=float), w)
            def wave_to_idx(x):
                return np.interp(x, w, np.arange(w.size, dtype=float))
        self._idx_to_wave = idx_to_wave
        self._wave_to_idx = wave_to_idx
        self._nx = int(w.size)

    def _compute_bin_edges(self, n):
        """Return integer bin edges [0..n] for n spectral columns."""
        return np.arange(int(n) + 1, dtype=float)

    def expand_wavelength_gap(self, x1d_wave, x1d_flux, x1d_fluxerr, s2d_data, expand_wavelength_gap=True):
        """Detect large wavelength gaps in 1D and insert NaNs into 1D flux and 2D data to keep alignment."""
        if not expand_wavelength_gap or x1d_wave is None or s2d_data is None:
            return x1d_wave, x1d_flux, x1d_fluxerr, s2d_data
        dx1d_wave = x1d_wave[1:] - x1d_wave[:-1]
        igap = np.argmax(dx1d_wave)
        dx1d_max = np.max(dx1d_wave)
        # heuristic neighbor dx
        left = dx1d_wave[igap-1] if igap-1 >= 0 else dx1d_wave[igap]
        right = dx1d_wave[igap+1] if igap+1 < len(dx1d_wave) else dx1d_wave[igap]
        dx_replace = (left + right) / 2.
        num_fill = int(np.round(dx1d_max / dx_replace))
        if num_fill > 1 and dx1d_max > 1.5 * dx_replace:
            wave_fill = np.mgrid[x1d_wave[igap]: x1d_wave[igap+1]: (num_fill+1)*1j]
            x1d_wave = np.concatenate([x1d_wave[:igap+1], wave_fill[1:-1], x1d_wave[igap+1:]])
            num_rows, num_waves = s2d_data.shape
            s2d_fill = np.zeros(shape=(num_rows, num_fill-1)) * np.nan
            s2d_data = np.concatenate([s2d_data[:, :igap+1], s2d_fill, s2d_data[:, igap+1:]], axis=1)
            x1d_fill = np.zeros(shape=(num_fill-1,)) * np.nan
            x1d_flux = np.concatenate([x1d_flux[:igap+1], x1d_fill, x1d_flux[igap+1:]])
            x1d_fluxerr = np.concatenate([x1d_fluxerr[:igap+1], x1d_fill, x1d_fluxerr[igap+1:]])
        return x1d_wave, x1d_flux, x1d_fluxerr, s2d_data
    # --------------------------------- UI ---------------------------------
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('FITS Spectra Viewer')
        self.setGeometry(100, 100, 1400, 900)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        # Toolbar
        self.create_toolbar()
        # Control panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        # Splitter for 2D and 1D plots
        self.splitter = QSplitter(Qt.Vertical)

        # 2D spectrum display (index domain)
        self.plot_2d = SpectrumPlotWidget(is_2d=True)
        self.plot_2d.setAspectLocked(False)
        self.image_item = pg.ImageItem()
        # Disable any image smoothing to keep pixel alignment crisp
        try:
            self.image_item.setAutoDownsample(False)
            self.image_item.setOpts(axisOrder='row-major')
        except Exception:
            pass
        self.plot_2d.addItem(self.image_item)

        # Hide bottom axis on the 2D plot; add top rest-frame axis in Å (index domain)
        try:
            pi2 = self.plot_2d.getPlotItem()
            pi2.showAxis('bottom', False)
            pi2.showAxis('top', True)
            self.top_axis = RestFrameAxisItem(
                'top',
                idx_to_wave=lambda x: self._idx_to_wave(x) if self._idx_to_wave else x,
                wave_to_idx=lambda w: self._wave_to_idx(w) if self._wave_to_idx else w,
                get_z=lambda: self.z_input.value() if self.z_input else 0.0,
            )
            # replace default top axis
            old_top = pi2.getAxis('top')
            pi2.layout.removeItem(old_top)
            pi2.axes['top']['item'] = self.top_axis
            pi2.layout.addItem(self.top_axis, 1, 1)
            self.top_axis.linkToView(pi2.vb)
            pi2.vb.sigResized.connect(lambda: self.top_axis.linkToView(pi2.vb))
        except Exception:
            self.top_axis = None

        # Set a diverging colormap similar to RdBu
        colors = [
            (0, 0, 180),  # Dark blue
            (100, 150, 255),  # Light blue
            (255, 255, 255),  # White
            (255, 150, 100),  # Light red
            (180, 0, 0)  # Dark red
        ]
        cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), color=colors)
        self.image_item.setLookupTable(cmap.getLookupTable())
        self.splitter.addWidget(self.plot_2d)

        # connect 2D hover to update coordinates
        try:
            self.plot_2d.scene().sigMouseMoved.connect(self._on_2d_mouse_moved)
        except Exception:
            pass
        # Also clear indicators when leaving the plot widgets
        try:
            self.plot_2d.viewport().installEventFilter(self)
        except Exception:
            pass
        # 2D -> 1D X sync on any range change
        try:
            self.plot_2d.getViewBox().sigRangeChanged.connect(self._on_2d_range_changed)
        except Exception:
            pass

        # 1D spectrum display (index domain)
        # Replace bottom axis with ObservedAxisItem that shows μm labels
        obs_axis = ObservedAxisItem('bottom')
        self.plot_1d = SpectrumPlotWidget()
        try:
            pi1 = self.plot_1d.getPlotItem()
            old_bottom = pi1.getAxis('bottom')
            pi1.layout.removeItem(old_bottom)
            pi1.axes['bottom']['item'] = obs_axis
            pi1.layout.addItem(obs_axis, 3, 1)
            obs_axis.linkToView(pi1.vb)
            pi1.vb.sigResized.connect(lambda: obs_axis.linkToView(pi1.vb))
        except Exception:
            pass
        self.obs_axis = obs_axis

        # Sync x axes: 1D -> 2D
        self.plot_1d.x_range_changed.connect(self.sync_x_range_to_2d)
        # Track ranges for nav stack
        try:
            self.plot_1d.getViewBox().sigRangeChanged.connect(self._on_any_range_changed)
            self.plot_2d.getViewBox().sigRangeChanged.connect(self._on_any_range_changed)
        except Exception:
            pass
        # Also rerender emission-line labels on 1D range changes (x or y)
        try:
            self.plot_1d.getViewBox().sigRangeChanged.connect(self._relayout_emission_labels)
        except Exception:
            pass

        # Cursor line and dot (green)
        self.cursor_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('g', width=1))
        self.cursor_dot = pg.ScatterPlotItem(size=8, brush='g', pen='w')
        self.plot_1d.addItem(self.cursor_line)
        self.plot_1d.addItem(self.cursor_dot)
        self.cursor_line.hide()
        self.cursor_dot.hide()

        self.splitter.addWidget(self.plot_1d)

        # Alt-selection: set Y-range on 1D
        try:
            self.plot_1d.alt_selected.connect(lambda ymin, ymax: self.plot_1d.setYRange(ymin, ymax, padding=0))
            self.plot_2d.alt_selected.connect(lambda ymin, ymax: self.plot_1d.setYRange(ymin, ymax, padding=0))
        except Exception:
            pass

        # Connect 1D hover
        try:
            self.plot_1d.scene().sigMouseMoved.connect(self._on_1d_mouse_moved)
        except Exception:
            pass
        try:
            self.plot_1d.viewport().installEventFilter(self)
        except Exception:
            pass

        # Set splitter sizes (1:3 ratio)
        self.splitter.setSizes([225, 675])
        layout.addWidget(self.splitter)

        # Status bar with two-row values
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        # Two labels in a small container
        self.s2d_label = QLabel("")  # top row
        self.x1d_label = QLabel("")  # bottom row
        mono_css = "QLabel { background-color: #f0f0f0; padding: 3px 6px; }"
        self.s2d_label.setStyleSheet(mono_css)
        self.x1d_label.setStyleSheet(mono_css)
        status_container = QWidget()
        vb = QVBoxLayout(status_container)
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setSpacing(0)
        vb.addWidget(self.s2d_label)
        vb.addWidget(self.x1d_label)
        self.status_bar.addPermanentWidget(status_container, 1)
        self.status_bar.showMessage('Ready. Use File menu to load FITS data.')

        # Initialize empty display
        self._update_status_clear()

    def create_toolbar(self):
        """Create application toolbar"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        # File actions
        open_action = QAction('Open FITS', self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_fits_file)
        toolbar.addAction(open_action)
        toolbar.addSeparator()
        # View actions
        reset_action = QAction('Reset View', self)
        reset_action.setShortcut('R')
        reset_action.triggered.connect(self.reset_views)
        toolbar.addAction(reset_action)
        autoscale_action = QAction('Autoscale', self)
        autoscale_action.setShortcut('A')
        autoscale_action.triggered.connect(self.autoscale)
        toolbar.addAction(autoscale_action)
        toolbar.addSeparator()
        back_action = QAction('Back', self)
        back_action.setShortcut('Alt+Left')
        back_action.triggered.connect(self.nav_back)
        toolbar.addAction(back_action)
        fwd_action = QAction('Forward', self)
        fwd_action.setShortcut('Alt+Right')
        fwd_action.triggered.connect(self.nav_forward)
        toolbar.addAction(fwd_action)

    def create_control_panel(self):
        """Create control panel for display options"""
        panel = QGroupBox("Display Controls")
        main_layout = QHBoxLayout()
        # Left side controls
        left_layout = QHBoxLayout()

        # 2D Controls
        self.s2d_controls_widget = QWidget()
        s2d_layout = QHBoxLayout()
        s2d_layout.setContentsMargins(0, 0, 0, 0)
        s2d_layout.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(['RdBu', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'])
        self.cmap_combo.currentTextChanged.connect(self.change_colormap)
        s2d_layout.addWidget(self.cmap_combo)
        s2d_layout.addSpacing(14)
        s2d_layout.addWidget(QLabel("Sigma Clip:"))
        self.sigma_spin = QSpinBox()
        self.sigma_spin.setRange(1, 10)
        self.sigma_spin.setValue(5)
        self.sigma_spin.valueChanged.connect(self.update_display)
        s2d_layout.addWidget(self.sigma_spin)
        self.s2d_controls_widget.setLayout(s2d_layout)
        left_layout.addWidget(self.s2d_controls_widget)
        left_layout.addSpacing(14)

        # Show error bars
        self.show_errors_check = QCheckBox("Show Uncertainty")
        self.show_errors_check.setChecked(True)
        self.show_errors_check.toggled.connect(self.update_display)
        left_layout.addWidget(self.show_errors_check)
        left_layout.addSpacing(14)
        # Autoscale Y on zoom
        self.autoscale_y_check = QCheckBox("Autoscale 1D Flux")
        self.autoscale_y_check.setChecked(False)
        self.autoscale_y_check.toggled.connect(self.on_autoscale_toggled)
        left_layout.addWidget(self.autoscale_y_check)
        left_layout.addSpacing(14)
        # Redshift input for rest-frame axis
        left_layout.addWidget(QLabel("z:"))
        self.z_input = QDoubleSpinBox()
        self.z_input.setDecimals(4)
        # avoid 1+z <= 0
        self.z_input.setRange(-0.99, 12.0)
        self.z_input.setSingleStep(0.001)
        self.z_input.setValue(0.0)
        # Immediate axis refresh & emission-line update on z change
        self.z_input.valueChanged.connect(self._refresh_top_axis)
        self.z_input.valueChanged.connect(lambda *_: self._update_emission_lines())
        left_layout.addWidget(self.z_input)
        left_layout.addSpacing(14)
        # Emission-line toggles & loader
        self.show_lines_check = QCheckBox("Show Emission Lines")
        self.show_lines_check.setChecked(True)
        self.show_lines_check.toggled.connect(lambda *_: self._update_emission_lines())
        left_layout.addWidget(self.show_lines_check)
        self.load_lines_btn = QPushButton("Load Lines…")
        self.load_lines_btn.clicked.connect(self.load_emission_lines_file)
        left_layout.addWidget(self.load_lines_btn)

        main_layout.addLayout(left_layout)
        main_layout.addStretch()
        # Right side - Gestures / Controls help
        help_text = (
            "X Wavelength: Zoom Scroll ⬆/⬇ ; Pan Scroll ⬅/➡\n"
            "Y Zoom: Pinch / Ctrl+Scroll / Alt/Option+Drag\n"
            "Alt/Option ⬅/➡: Back/Forward (history)"
        )
        gestures_label = QLabel(help_text)
        gestures_label.setStyleSheet("QLabel { color: #666; }")
        main_layout.addWidget(gestures_label)
        panel.setLayout(main_layout)
        return panel

    # --------------------------- File I/O ---------------------------
    def open_fits_file(self):
        """Open and load FITS file, looking for associated s2d/x1d pair"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open FITS File",
            "",
            "FITS Files (*.fits *.fit);;All Files (*.*)"
        )
        if file_path:
            try:
                self.load_fits_pair(file_path)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load FITS file:\n{str(e)}")

    def load_fits_pair(self, file_path):
        """Load s2d/x1d FITS pair, or a single 1D FITS file if no pair is found."""
        base_name = os.path.basename(file_path)
        s2d_path, x1d_path = None, None

        # Determine if this is s2d or x1d and find the pair
        if 's2d' in base_name.lower():
            s2d_path = file_path
            x1d_path_try = file_path.replace('s2d', 'x1d').replace('S2D', 'X1D')
            if not os.path.exists(x1d_path_try):
                x1d_path_try = file_path.replace('_s2d', '_x1d').replace('_S2D', '_X1D')
            if os.path.exists(x1d_path_try):
                x1d_path = x1d_path_try
        elif 'x1d' in base_name.lower():
            x1d_path = file_path
            s2d_path_try = file_path.replace('x1d', 's2d').replace('X1D', 'S2D')
            if not os.path.exists(s2d_path_try):
                s2d_path_try = file_path.replace('_x1d', '_s2d').replace('_X1D', '_S2D')
            if os.path.exists(s2d_path_try):
                s2d_path = s2d_path_try

        # If no pair was found, assume the input file is a 1D spectrum
        if not s2d_path and not x1d_path:
            x1d_path = file_path
            s2d_path = None

        # Reset data state
        self.s2d_data = None
        self.x1d_wave = None
        self.x1d_flux = None
        self.x1d_fluxerr = None
        self.current_s2d_file = None
        self.current_x1d_file = None

        # Load s2d data if a path exists for it
        if s2d_path and os.path.exists(s2d_path):
            self.load_s2d_data(s2d_path)
            self.current_s2d_file = s2d_path

        # Load x1d data if a path exists, otherwise try to extract from 2D
        if x1d_path and os.path.exists(x1d_path):
            self.load_x1d_data(x1d_path)
            self.current_x1d_file = x1d_path
        elif self.s2d_data is not None:
            self.extract_1d_from_2d()

        # If after all that we have no 1D data, it's an error.
        if self.x1d_flux is None:
            raise ValueError("Could not load a 1D spectrum from the input file.")

        # --- UI Adjustments for 1D/2D mode ---
        is_1d_only = self.s2d_data is None
        self.plot_2d.setVisible(not is_1d_only)
        self.s2d_controls_widget.setVisible(not is_1d_only)
        self.s2d_label.setVisible(not is_1d_only)
        if is_1d_only:
            self.splitter.setSizes([0, 1])  # Collapse the top panel
            self.resize(self.width(), 550)  # Shorter window
        else:
            self.splitter.setSizes([225, 675])  # Restore 1:3 ratio
            self.resize(self.width(), 900)  # Restore original height

        # Try to expand wavelength gaps for alignment
        if self.x1d_wave is not None and self.s2d_data is not None:
            self.x1d_wave, self.x1d_flux, self.x1d_fluxerr, self.s2d_data = \
                self.expand_wavelength_gap(self.x1d_wave, self.x1d_flux, self.x1d_fluxerr, \
                                           self.s2d_data, expand_wavelength_gap=True)
        # Build the mappers and update axes
        self._build_index_wavelength_mappers()
        if self.obs_axis is not None and self._idx_to_wave is not None:
            self.obs_axis.set_mappers(self._idx_to_wave, self._wave_to_idx)
        if self.top_axis is not None and self._idx_to_wave is not None:
            self.top_axis.set_mappers(self._idx_to_wave, self._wave_to_idx)
        self.update_display()
        # Emission-line overlays
        self._update_emission_lines()
        # Update status (filenames)
        status_msg = []
        if self.current_s2d_file:
            status_msg.append(f"S2D: {os.path.basename(self.current_s2d_file)}")
        if self.current_x1d_file:
            status_msg.append(f"X1D: {os.path.basename(self.current_x1d_file)}")
        self.status_bar.showMessage(" ".join(status_msg), 5000)
        # Reset the value rows to placeholders until hover
        self._update_status_clear()

    def load_s2d_data(self, file_path):
        """Load 2D spectrum data"""
        with fits.open(file_path) as hdul:
            # Look for SCI extension or primary data
            for hdu in hdul:
                if hdu.data is not None and len(hdu.data.shape) == 2:
                    self.s2d_data = hdu.data.copy()
                    # Replace zeros with NaN
                    self.s2d_data[self.s2d_data == 0] = np.nan
                    break

    def load_x1d_data(self, file_path):
        """Load 1D extracted spectrum data"""
        with fits.open(file_path) as hdul:
            # Look for spectral data - adapt based on FITS structure
            for hdu in hdul:
                if hasattr(hdu, 'data') and hdu.data is not None:
                    if hasattr(hdu, 'columns') and 'WAVELENGTH' in getattr(hdu, 'columns').names:
                        # Table format
                        self.x1d_wave = hdu.data['WAVELENGTH'].copy()
                        self.x1d_flux = hdu.data['FLUX'].copy()
                        if 'FLUX_ERROR' in hdu.columns.names:
                            self.x1d_fluxerr = hdu.data['FLUX_ERROR'].copy()
                        else:
                            self.x1d_fluxerr = np.abs(self.x1d_flux) * 0.1
                        break
                    elif len(hdu.data.shape) == 1:
                        # Simple 1D array
                        self.x1d_flux = hdu.data.copy()
                        header = dict(hdu.header)
                        # Generate wavelength from header if available
                        if 'CRVAL1' in header:
                            crval = header['CRVAL1']
                            cdelt = header.get('CDELT1', 1.0)
                            crpix = header.get('CRPIX1', 1.0)
                            nx = len(self.x1d_flux)
                            self.x1d_wave = crval + (np.arange(nx) - crpix + 1) * cdelt
                        else:
                            self.x1d_wave = np.linspace(1.0, 5.5, len(self.x1d_flux))
                        self.x1d_fluxerr = np.abs(self.x1d_flux) * 0.1
                        break

    def extract_1d_from_2d(self):
        """Extract 1D spectrum from 2D data"""
        if self.s2d_data is None:
            return
        ny, nx = self.s2d_data.shape
        # Simple extraction from middle rows
        extract_width = 6
        y_center = ny // 2
        ystart = max(0, y_center - extract_width // 2)
        ystop = min(ny, y_center + extract_width // 2)
        self.x1d_flux = np.nansum(self.s2d_data[ystart:ystop, :], axis=0)
        self.x1d_wave = np.linspace(1.0, 5.5, nx)  # Default wavelength range
        self.x1d_fluxerr = np.abs(self.x1d_flux) * 0.1  # 10% error estimate
        # Expand wavelength gaps so 1D and 2D align
        try:
            self.x1d_wave, self.x1d_flux, self.x1d_fluxerr, self.s2d_data = \
                self.expand_wavelength_gap(self.x1d_wave, self.x1d_flux, self.x1d_fluxerr,
                                           self.s2d_data, expand_wavelength_gap=True)
            ny, nx = self.s2d_data.shape
        except Exception:
            pass

    # -------------------------------- Display --------------------------------
    def update_display(self):
        """Update both 2D and 1D displays (now in index domain)."""
        if self.s2d_data is None and self.x1d_flux is None:
            return
        # Preserve current view ranges to avoid resetting on sigma updates
        x1d_rng = self.plot_1d.getViewBox().viewRange() if self._have_plotted else None
        x2d_rng = self.plot_2d.getViewBox().viewRange() if self._have_plotted else None
        # Update 2D display
        if self.s2d_data is not None:
            # Sigma clipping for display range
            sigma_val = self.sigma_spin.value()
            try:
                clipped = sigma_clip(self.s2d_data[~np.isnan(self.s2d_data)],
                                     sigma=sigma_val, maxiters=3)
                vmin, vmax = np.min(clipped), np.max(clipped)
            except Exception:
                vmin, vmax = np.nanmin(self.s2d_data), np.nanmax(self.s2d_data)
            ny, nx = self.s2d_data.shape
            # Render as (nx, ny) with rect spanning x=[0..nx], y=[0..ny] so columns map 1:1 to index
            self.image_item.setImage(self.s2d_data, autoLevels=False)
            self.image_item.setLevels((vmin, vmax))
            rect = pg.QtCore.QRectF(0.0, 0.0, float(nx), float(ny))
            try:
                self.image_item.setRect(rect)
            except Exception:
                # Fallback: setTransform if setRect unavailable
                try:
                    tr = pg.QtGui.QTransform()
                    tr.scale(float(nx) / self.s2d_data.T.shape[0], float(ny) / self.s2d_data.T.shape[1])
                    self.image_item.setTransform(tr)
                except Exception:
                    pass
            # Set data limits for 2D (minXRange ~ 3 bins, minYRange ~ 3 rows)
            self.plot_2d.set_data_limits(0.0, float(nx), 0.0, float(ny),
                                         min_dx=1.0, min_bins=min_bins_default, min_y_rows=min_y_rows_default)
            self.plot_2d.setYRange(0, ny, padding=0)

        # Update 1D display (as steps using integer edges)
        if self.x1d_flux is not None and self.x1d_wave is not None:
            self._build_index_wavelength_mappers()
            nx = self._nx if self._nx is not None else len(self.x1d_flux)
            x_edges = self._compute_bin_edges(nx)  # [0..nx]
            # Flux step curve
            if not hasattr(self, 'flux_curve') or self.flux_curve is None:
                self.flux_curve = self.plot_1d.plot(x_edges, self.x1d_flux,
                                                   stepMode=True, pen='w', name='Flux')
            else:
                self.flux_curve.setData(x_edges, self.x1d_flux, stepMode=True)
            # Error step curve
            if self.show_errors_check.isChecked() and self.x1d_fluxerr is not None:
                if not hasattr(self, 'err_curve') or self.err_curve is None:
                    self.err_curve = self.plot_1d.plot(x_edges, self.x1d_fluxerr,
                                                       stepMode=True,
                                                       pen=pg.mkPen('r', width=0.5), name='Error')
                else:
                    self.err_curve.setData(x_edges, self.x1d_fluxerr, stepMode=True)
            else:
                if hasattr(self, 'err_curve') and self.err_curve is not None:
                    try:
                        self.plot_1d.removeItem(self.err_curve)
                    except Exception:
                        pass
                    self.err_curve = None
            # Zero line as step
            zeros = np.zeros_like(self.x1d_flux)
            if not hasattr(self, 'zero_curve') or self.zero_curve is None:
                self.zero_curve = self.plot_1d.plot(x_edges, zeros,
                                                    stepMode=True,
                                                    pen=pg.mkPen('gray', width=0.5, style=Qt.DashLine))
            else:
                self.zero_curve.setData(x_edges, zeros, stepMode=True)
            # Set data limits (min X range ~ 3 bins)
            self.plot_1d.set_data_limits(0.0, float(nx), min_dx=1.0,
                                         min_bins=min_bins_default, min_y_rows=None)
            # Ensure observed-axis tick mappers are set
            if self.obs_axis is not None and self._idx_to_wave is not None:
                self.obs_axis.set_mappers(self._idx_to_wave, self._wave_to_idx)
            # Initial view only on the first draw
            if not self._have_plotted:
                self.plot_1d.setXRange(0.0, float(nx), padding=0)
                flux_min = np.nanmin(self.x1d_flux)
                flux_max = np.nanmax(self.x1d_flux)
                margin = (flux_max - flux_min) * 0.1
                self.plot_1d.setYRange(min(flux_min - margin, -margin),
                                       flux_max + margin, padding=0)
            # Restore previous view ranges if already plotted
            if self._have_plotted:
                try:
                    if x1d_rng is not None:
                        self.plot_1d.getViewBox().setXRange(*x1d_rng[0], padding=0)
                        self.plot_1d.getViewBox().setYRange(*x1d_rng[1], padding=0)
                    if x2d_rng is not None:
                        self.plot_2d.getViewBox().setXRange(*x2d_rng[0], padding=0)
                        self.plot_2d.getViewBox().setYRange(*x2d_rng[1], padding=0)
                except Exception:
                    pass
            else:
                self._have_plotted = True
                self._push_nav_state()

        # After any redraw, ensure emission line labels are laid out
        self._relayout_emission_labels()

    def sync_x_range_to_2d(self, xmin, xmax):
        """Synchronize 2D plot x range with 1D plot (index domain)."""
        if self.s2d_data is not None:
            if self._syncing_x:
                return
            self._syncing_x = True
            try:
                self.plot_2d.setXRange(xmin, xmax, padding=0)
            finally:
                self._syncing_x = False
        # Autoscale Y if enabled (use indices to pick visible bins)
        if self.autoscale_y_check.isChecked() and self.x1d_flux is not None:
            nx = len(self.x1d_flux)
            i = np.arange(nx)
            mask = (i >= np.floor(xmin)) & (i <= np.ceil(xmax))
            if np.any(mask):
                visible_flux = self.x1d_flux[mask]
                flux_min = np.nanmin(visible_flux)
                flux_max = np.nanmax(visible_flux)
                margin = (flux_max - flux_min) * 0.1
                self.plot_1d.setYRange(min(flux_min - margin, -margin),
                                       flux_max + margin, padding=0)
        # Relayout labels when x-range changes
        self._relayout_emission_labels()

    def _on_2d_range_changed(self):
        """When 2D viewbox changes, keep 1D X in sync."""
        if self._syncing_x:
            return
        self._syncing_x = True
        try:
            xmin, xmax = self.plot_2d.getViewBox().viewRange()[0]
            self.plot_1d.setXRange(xmin, xmax, padding=0)
        finally:
            self._syncing_x = False

    def _on_any_range_changed(self):
        if self._is_restoring:
            return
        self._push_nav_state()

    def _push_nav_state(self):
        try:
            x1d = self.plot_1d.getViewBox().viewRange()
            x2d = self.plot_2d.getViewBox().viewRange()
        except Exception:
            return
        state = dict(
            x1d_x=tuple(x1d[0]), x1d_y=tuple(x1d[1]),
            x2d_x=tuple(x2d[0]), x2d_y=tuple(x2d[1]),
        )
        # truncate redo branch
        if self._nav_index < len(self._nav_stack) - 1:
            self._nav_stack = self._nav_stack[:self._nav_index+1]
        self._nav_stack.append(state)
        self._nav_index = len(self._nav_stack) - 1

    def _apply_nav_state(self, idx):
        if idx < 0 or idx >= len(self._nav_stack):
            return
        st = self._nav_stack[idx]
        self._is_restoring = True
        try:
            self.plot_1d.getViewBox().setXRange(*st['x1d_x'], padding=0)
            self.plot_1d.getViewBox().setYRange(*st['x1d_y'], padding=0)
            self.plot_2d.getViewBox().setXRange(*st['x2d_x'], padding=0)
            self.plot_2d.getViewBox().setYRange(*st['x2d_y'], padding=0)
        finally:
            self._is_restoring = False

    def nav_back(self):
        if self._nav_index > 0:
            self._nav_index -= 1
            self._apply_nav_state(self._nav_index)

    def nav_forward(self):
        if self._nav_index < len(self._nav_stack) - 1:
            self._nav_index += 1
            self._apply_nav_state(self._nav_index)

    def on_autoscale_toggled(self):
        """Handle autoscale Y checkbox toggle"""
        if self.autoscale_y_check.isChecked():
            xmin, xmax = self.plot_1d.getViewBox().viewRange()[0]
            self.sync_x_range_to_2d(xmin, xmax)

    # ----------------------- HOVER & STATUS ROWS -----------------------
    def _basename_or_dash(self, path):
        return os.path.basename(path) if path else "—"

    def _rest_from_obs(self, lam_obs_um):
        """Convert observed μm to rest Å given current z."""
        try:
            z = float(self.z_input.value()) if self.z_input else 0.0
        except Exception:
            z = 0.0
        z = max(z, -0.99)
        return (lam_obs_um / (1.0 + z)) * 1e4

    def _update_status(self, obs_um=None, flux=None, y=None, x=None, val2d=None):
        """Set two-line status text. Use '—' for any missing field."""
        s2d_name = self._basename_or_dash(self.current_s2d_file)
        x1d_name = self._basename_or_dash(self.current_x1d_file)
        if obs_um is None:
            rest_ang_str = "—"
        else:
            rest_ang = self._rest_from_obs(obs_um)
            # formatting: Å integer if large, else 1 decimal
            if abs(rest_ang) >= 1000:
                rest_ang_str = f"{rest_ang:.0f} Å"
            elif abs(rest_ang) >= 100:
                rest_ang_str = f"{rest_ang:.1f} Å"
            else:
                rest_ang_str = f"{rest_ang:.2f} Å"
        if (flux is None) or (not np.isfinite(flux)):
            s2d_val_str = "     2D flux"
        else:
            s2d_val_str = "         2D flux"
        if (y is None) or (x is None) or (val2d is None) or (not np.isfinite(val2d)):
            s2d_val_str += "[y,x]: —"
        else:
            s2d_val_str += f"[{y:d}, {x:d}]: {val2d:.4g} MJy/sr"
        rest_str = "   rest λ:"
        obs_str  = "    obs λ: "
        if obs_um is None:
            obs_str += "—"
        else:
            obs_str += f"{obs_um:.4f} μm"
        flux_str = "     1D flux: "
        if (flux is None) or (not np.isfinite(flux)):
            flux_str += "—"
        else:
            flux_str += f"{1e6*flux:.4g} µJy"
        self.s2d_label.setText(f"S2D: {s2d_name} {rest_str} {rest_ang_str} {s2d_val_str}")
        self.x1d_label.setText(f"X1D: {x1d_name} {obs_str} {flux_str}")

    def _update_status_clear(self):
        """Clear values but keep filenames visible."""
        self._update_status(obs_um=None, flux=None, y=None, x=None, val2d=None)

    def _show_cursor_at(self, x_idx, flux_val):
        """Move & show the green cursor line+dot on the 1D plot (x in index)."""
        try:
            if x_idx is None or not np.isfinite(x_idx):
                return
            self.cursor_line.setPos(float(x_idx))
            if (flux_val is not None) and np.isfinite(flux_val):
                self.cursor_dot.setData([float(x_idx)], [float(flux_val)])
                self.cursor_dot.show()
            else:
                # still show the line, hide dot if flux is NaN
                self.cursor_dot.hide()
            self.cursor_line.show()
        except Exception:
            pass

    def _hide_cursor(self):
        try:
            self.cursor_line.hide()
            self.cursor_dot.hide()
        except Exception:
            pass

    def _in_index_edges(self, x):
        try:
            nx = self._nx if self._nx is not None else len(self.x1d_flux)
            return (x >= 0.0) and (x <= float(nx))
        except Exception:
            return False

    def _col_from_x(self, x):
        """Return integer column index using floor(), clamped to [0, nx-1]."""
        nx = self._nx if self._nx is not None else (len(self.x1d_flux) if self.x1d_flux is not None else None)
        if nx is None or not np.isfinite(x): return None
        return int(np.clip(int(np.floor(float(x))), 0, int(nx) - 1))

    def _bin_center(self, ix):
        return float(ix) + 0.5

    def _on_1d_mouse_moved(self, pos):
        """Hover handler for the 1D plot (index domain)."""
        if self.x1d_wave is None or self.x1d_flux is None:
            return
        try:
            vb = self.plot_1d.getViewBox()
            if not self.plot_1d.sceneBoundingRect().contains(pos):
                # outside widget -> clear
                self._hide_cursor()
                self._update_status_clear()
                return
            mp = vb.mapSceneToView(pos)
            x = mp.x()  # index
            # Only when hovering over the data range [0..nx]
            if (not np.isfinite(x)) or (not self._in_index_edges(x)):
                self._hide_cursor()
                self._update_status_clear()
                return
            nx = len(self.x1d_wave)
            ix = self._col_from_x(x)
            wave = float(self.x1d_wave[ix])
            flux = float(self.x1d_flux[ix]) if ix < len(self.x1d_flux) else np.nan
            # Update cursor on 1D (at index)
            self._show_cursor_at(self._bin_center(ix), flux)
            # Update two-row status
            self._update_status(obs_um=wave, flux=flux, y=None, x=ix, val2d=None)
        except Exception:
            # On error, clear
            self._hide_cursor()
            self._update_status_clear()

    def _on_2d_mouse_moved(self, pos):
        """Hover handler for the 2D image (index domain)."""
        try:
            vb = self.plot_2d.getViewBox()
            if not self.plot_2d.sceneBoundingRect().contains(pos):
                self._hide_cursor()
                self._update_status_clear()
                return
            mouse_point = vb.mapSceneToView(pos)
            x = mouse_point.x()  # index
            y = mouse_point.y()
            if (self.x1d_wave is None) or (self.s2d_data is None):
                self._hide_cursor()
                self._update_status_clear()
                return
            ny, nx = self.s2d_data.shape
            # Must be inside x edges and inside [0, ny) rows
            if (not np.isfinite(x)) or (not np.isfinite(y)) or (not self._in_index_edges(x)) or (y < 0) or (y >= ny):
                self._hide_cursor()
                self._update_status_clear()
                return
            ix = self._col_from_x(x)
            iy = int(np.clip(int(round(y)), 0, ny - 1))
            val = self.s2d_data[iy, ix]
            flux = None
            wave = None
            if self.x1d_flux is not None and ix < len(self.x1d_flux):
                flux = float(self.x1d_flux[ix])
            if self.x1d_wave is not None:
                wave = float(self.x1d_wave[ix])
            # Update cursor on 1D at that index
            self._show_cursor_at(self._bin_center(ix), flux)
            # Update the two-row status (both rows populated)
            self._update_status(obs_um=wave, flux=flux, y=iy, x=ix, val2d=val)
        except Exception:
            self._hide_cursor()
            self._update_status_clear()

    def eventFilter(self, obj, event):
        """Clear cursor and values when leaving either plot."""
        try:
            if event.type() == QEvent.Leave:
                self._hide_cursor()
                self._update_status_clear()
        except Exception:
            pass
        # continue normal processing
        return False

    # ------------------------------ Misc ------------------------------
    def change_colormap(self, cmap_name):
        """Change 2D colormap"""
        if cmap_name == 'RdBu':
            colors = [
                (0, 0, 180),
                (100, 150, 255),
                (255, 255, 255),
                (255, 150, 100),
                (180, 0, 0)
            ]
            cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)),
                               color=colors)
        else:
            cmap = pg.colormap.get(cmap_name)
        self.image_item.setLookupTable(cmap.getLookupTable())

    def reset_views(self):
        """Reset all views to default (index domain)"""
        if self._nx is not None:
            nx = float(self._nx)
        elif self.x1d_wave is not None:
            nx = float(len(self.x1d_wave))
        elif self.s2d_data is not None:
            nx = float(self.s2d_data.shape[1])
        else:
            nx = None
        if nx is not None:
            self.plot_1d.setXRange(0.0, nx, padding=0)
            self.plot_2d.setXRange(0.0, nx, padding=0)
        if self.s2d_data is not None:
            ny, _ = self.s2d_data.shape
            self.plot_2d.setYRange(0, ny, padding=0)
        if self.x1d_flux is not None:
            flux_min = np.nanmin(self.x1d_flux)
            flux_max = np.nanmax(self.x1d_flux)
            margin = (flux_max - flux_min) * 0.1
            self.plot_1d.setYRange(min(flux_min - margin, -margin),
                                   flux_max + margin, padding=0)
        self._push_nav_state()
        # also clear hover indicators after reset
        self._hide_cursor()
        self._update_status_clear()
        # Relayout labels after reset
        self._relayout_emission_labels()

    def autoscale(self):
        """Autoscale displays"""
        self.reset_views()

    def _refresh_top_axis(self, *_):
        try:
            if hasattr(self, 'top_axis') and self.top_axis is not None:
                # Force tick recomputation and repaint immediately
                self.top_axis.picture = None
                self.top_axis.update()
            if hasattr(self, 'obs_axis') and self.obs_axis is not None:
                self.obs_axis.picture = None
                self.obs_axis.update()
        except Exception:
            pass

    # ---------------------- Emission Line Overlays ----------------------
    def _parse_emission_lines_file(self, path):
        """Parse a simple two-column text file: name [TAB/space] wavelength(Å)."""
        lines = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for raw in f:
                    s = raw.strip()
                    if not s:
                        continue
                    if s.startswith('#'):
                        continue
                    # Skip header if present
                    low = s.lower()
                    if ('name' in low) and ('wave' in low or 'λ' in low):
                        continue
                    if '\t' in s:
                        parts = s.split('\t')
                    else:
                        # split by whitespace, but allow spaces inside the name by taking last token as wavelength
                        parts = s.rsplit(None, 1)
                    if len(parts) < 2:
                        continue
                    name = parts[0].strip()
                    # If there were more than 2 parts (tabbed file with extra tabs), recombine all but last
                    if '\t' in s and len(parts) > 2:
                        name = '\t'.join(parts[:-1]).strip()
                    try:
                        wav = float(parts[-1])
                    except Exception:
                        continue
                    lines.append((name, wav))
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to read lines file:\n{e}')
            return None
        if not lines:
            QMessageBox.warning(self, 'Empty file', 'No emission lines found in the selected file.')
            return None
        return lines

    def load_emission_lines_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Load Emission Lines',
            '',
            'Text Files (*.txt *.dat *.tsv *.csv);;All Files (*.*)')
        if not path:
            return
        parsed = self._parse_emission_lines_file(path)
        if parsed is None:
            return
        self.emission_lines = parsed
        self._update_emission_lines()

    def _clear_emission_overlays(self):
        # Remove existing items from plots
        for it in self.em_line_items_1d:
            try:
                self.plot_1d.removeItem(it)
            except Exception:
                pass
        for it in self.em_label_items_1d:
            try:
                self.plot_1d.removeItem(it)
            except Exception:
                pass
        for it in self.em_line_items_2d:
            try:
                self.plot_2d.removeItem(it)
            except Exception:
                pass
        self.em_line_items_1d = []
        self.em_label_items_1d = []
        self.em_line_items_2d = []

    def _update_emission_lines(self):
        """Recreate emission line markers and labels according to current z and mapping."""
        # Clear prior overlays
        self._clear_emission_overlays()
        # Pre-conditions
        if not self.show_lines_check or not self.show_lines_check.isChecked():
            return
        if self.emission_lines is None or len(self.emission_lines) == 0:
            return
        if self._wave_to_idx is None or self._idx_to_wave is None:
            return
        # Compute observed positions in μm and then index
        try:
            z = float(self.z_input.value()) if self.z_input else 0.0
        except Exception:
            z = 0.0
        z = max(z, -0.99)
        nx = self._nx if self._nx is not None else (len(self.x1d_wave) if self.x1d_wave is not None else 0)
        if nx == 0:
            return

        # Build list of visible line positions (idx) inside data range
        line_positions = []  # list of (name, rest_A, idx_float)
        for name, rest_A in self.emission_lines:
            if "Fe" in name or "Ar" in name:
                continue  # Skip iron and argon lines for clarity
            try:
                obs_um = (rest_A * (1.0 + z)) / 1e4
                idx = float(self._wave_to_idx(obs_um))
                if np.isfinite(idx) and (0.0 < idx < float(nx)-1):
                    line_positions.append((name, rest_A, idx))
            except Exception:
                continue
        if not line_positions:
            return
        # Sort by x index for stable labeling
        line_positions.sort(key=lambda t: t[2])

        # Draw vertical lines on 1D and 2D
        pen = pg.mkPen('#ffd400', width=1, style=Qt.DashLine)  # yellow
        for name, rest_A, idx in line_positions:
            ln1 = pg.InfiniteLine(pos=float(idx), angle=90, pen=pen)
            self.plot_1d.addItem(ln1)
            self.em_line_items_1d.append(ln1)
            if self.s2d_data is not None:
                ln2 = pg.InfiniteLine(pos=float(idx), angle=90, pen=pen)
                self.plot_2d.addItem(ln2)
                self.em_line_items_2d.append(ln2)

        # Create labels on 1D; positions will be laid out in _relayout_emission_labels
        for name, rest_A, idx in line_positions:
            # Label shows the name; you could also show rest if desired
            html = f"<span style='color:#000; background-color: rgba(255,212,0,0.85); padding:1px 3px; border-radius:2px;'>{name}</span>"
            ti = pg.TextItem(html=html, anchor=(0.5, 1.0))  # center above the x position
            self.plot_1d.addItem(ti)
            # Temporarily position; final layout handled below
            ti.setPos(float(idx), 0.0)
            self.em_label_items_1d.append(ti)

        # Final layout based on current viewbox
        self._relayout_emission_labels()

    def _relayout_emission_labels(self, *args):
        """Place emission-line labels near the top of the current view and stagger
        them vertically if their x positions would overlap in screen pixels."""
        if not self.em_label_items_1d:
            return
        try:
            vb = self.plot_1d.getViewBox()
            (xmin, xmax), (ymin, ymax) = vb.viewRange()
            px_w = max(self.plot_1d.viewport().width(), 1)
            yr = max(ymax - ymin, 1e-9)
            # Compute x positions for labels from their attached InfiniteLines/known order
            # Build list of (px, TextItem)
            xs = []
            for ti in self.em_label_items_1d:
                # TextItem position .pos() gives a QPointF
                x = ti.pos().x()
                # Skip labels that are off-screen to reduce clutter
                if x < xmin + 0.5 or x > xmax - 0.5:
                    ti.setVisible(False)
                    continue
                ti.setVisible(True)
                px = (x - xmin) / max(xmax - xmin, 1e-9) * px_w
                xs.append((px, ti, x))
            if not xs:
                return
            xs.sort(key=lambda t: t[0])
            # Greedy stacking: maintain last px per level
            min_sep_px = 60.0  # minimum horizontal separation to keep same row
            levels_px = []  # track last px used per level
            # Base and step as fractions of y-range
            base_y = ymax - 0.06 * yr
            step = 0.055 * yr  # bump down ~5.5% of y-range per level
            for px, ti, x in xs:
                # Find the first level where we have enough horizontal separation
                level = 0
                placed = False
                for li, last_px in enumerate(levels_px):
                    if abs(px - last_px) >= min_sep_px:
                        level = li
                        levels_px[li] = px
                        placed = True
                        break
                if not placed:
                    levels_px.append(px)
                    level = len(levels_px) - 1
                y = base_y - level * step
                ti.setPos(float(x), float(y))
        except Exception:
            pass

# ------------------------------ MAIN ------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    viewer = FITSSpectraViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    import sys
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication(sys.argv)
    win = FITSSpectraViewer()

    # Load spectra files from command line arguments if provided
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        try:
            win.load_fits_pair(filename)
        except Exception as e:
            QMessageBox.critical(win, 'Error', f'Failed to load file:\n{e}')

    # Set redshift value z if provided
    if len(sys.argv) > 2:
        try:
            z_val = float(sys.argv[2])
            win.z_input.setValue(z_val)
            win._update_emission_lines()
        except Exception:
            QMessageBox.critical(win, 'Error', f'Failed to load redshift value:\n{e}')

    # Load emission lines file if provided
    if len(sys.argv) > 3:
        emission_lines_file = sys.argv[3]
        try:
            parsed = win._parse_emission_lines_file(emission_lines_file)
            if parsed is not None:
                win.emission_lines = parsed
                win._update_emission_lines()
        except Exception as e:
            QMessageBox.critical(win, 'Error', f'Failed to load emission lines file:\n{e}')

    win.show()
    sys.exit(app.exec_())
