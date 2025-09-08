#!/usr/bin/env python3
"""
FITS Spectra Viewer with PyQt
Trackpad gestures:
- Zoom x (wavelength): 2 fingers up/down
- Pan x (wavelength): 2 fingers left/right
- Scale y (flux): 2 fingers pinch
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
pg.setConfigOptions(antialias=True)

min_bins_default   = 5  # minimum zoom in x = wavelength
min_y_rows_default = 5  # minimum zoom in y for 2D rows

class RestFrameAxisItem(pg.AxisItem):
    """
    Top axis that shows rest-frame wavelength in Å from observed μm.
    It computes its own 'nice' Å tick spacing at regular intervals,
    independent of the bottom axis tick alignment.
    """
    def __init__(self, orientation='top', get_z=lambda: 0.0):
        super().__init__(orientation=orientation)
        self._get_z = get_z
        self.setLabel("Rest-frame λ", units='Å')

    @staticmethod
    def _nice_step(x):
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

    def tickValues(self, minVal, maxVal, size):
        """
        Compute tick positions at regular rest-frame Å intervals and
        return them mapped into observed μm coordinates for the view.
        """
        try:
            z = float(self._get_z()) if callable(self._get_z) else float(self._get_z)
        except Exception:
            z = 0.0
        z = max(z, -0.99)  # avoid 1+z <= 0

        # observed μm -> rest Å
        lam_rest_min = (minVal / (1.0 + z)) * 1e4
        lam_rest_max = (maxVal / (1.0 + z)) * 1e4
        if lam_rest_max < lam_rest_min:
            lam_rest_min, lam_rest_max = lam_rest_max, lam_rest_min

        rng = max(lam_rest_max - lam_rest_min, 1e-12)
        # Choose target ~80 px per major tick
        approxNTicks = max(int(size / 80.0), 2)
        step = self._nice_step(rng / approxNTicks)

        # Build major ticks in rest Å
        first = np.ceil(lam_rest_min / step) * step
        majors_rest = []
        v = first
        # guard against floating step accumulation
        for _ in range(2000):
            if v > lam_rest_max + 1e-9:
                break
            majors_rest.append(v)
            v += step

        # Minor ticks at 1/5 step
        minor_step = step / 5.0
        first_m = np.ceil(lam_rest_min / minor_step) * minor_step
        minors_rest = []
        v = first_m
        for _ in range(10000):
            if v > lam_rest_max + 1e-9:
                break
            # avoid duplicates at major positions
            if (abs((v / step) - np.round(v / step)) > 1e-6):
                minors_rest.append(v)
            v += minor_step

        # Map to observed μm positions for the viewbox
        majors_obs = (np.array(majors_rest) * (1.0 + z)) / 1e4
        minors_obs = (np.array(minors_rest) * (1.0 + z)) / 1e4

        return [
            (float(step * (1.0 + z) / 1e4), majors_obs),
            (float(minor_step * (1.0 + z) / 1e4), minors_obs)
        ]

    def tickStrings(self, values, scale, spacing):
        # observed μm -> rest Å : λ_rest = λ_obs/(1+z) * 1e4
        try:
            z = float(self._get_z()) if callable(self._get_z) else float(self._get_z)
        except Exception:
            z = 0.0
        z = max(z, -0.99)  # guard against 1+z <= 0
        rest = (np.array(values, dtype=float) / (1.0 + z)) * 1e4
        out = []
        for v in rest:
            av = abs(v)
            if av >= 1:
                s = f"{v:.0f}"
            elif av >= 0.1:
                s = f"{v:.1f}"
            else:
                s = f"{v:.2f}"
            out.append(s)
        return out


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
        self.setLabel('bottom', 'Wavelength', units='μm')
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
        self.min_y_rows = None   # minimum allowed y-range in rows (2D only)

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
            # For 2D we can enforce minYRange in data rows; for 1D we clamp in handlers
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
                # Slower base (gentler zoom)
                scale_factor = 1.02 ** (-dy)
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
            # Ctrl + Scroll => Scale Y (fallback for pinch) — both 1D & 2D, cursor-anchored
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
                    # Compute new Y range (inverse scaling for typical pinch semantics)
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
        self.coord_label = None

        self.init_ui()

    def _compute_bin_edges(self, w):
        """Given bin centers w (monotonic, possibly irregular), return bin edges of length len(w)+1."""
        w = np.asarray(w, dtype=float)
        n = w.size
        if n == 0:
            return np.array([], dtype=float)
        if n == 1:
            return np.array([w[0] - 0.5, w[0] + 0.5], dtype=float)
        edges = np.empty(n + 1, dtype=float)
        edges[1:-1] = 0.5 * (w[:-1] + w[1:])
        edges[0] = w[0] - 0.5 * (w[1] - w[0])
        edges[-1] = w[-1] + 0.5 * (w[-1] - w[-2])
        return edges

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
            x1d_fill = np.zeros(shape=(num_fill-1)) * np.nan
            x1d_flux = np.concatenate([x1d_flux[:igap+1], x1d_fill, x1d_flux[igap+1:]])
            x1d_fluxerr = np.concatenate([x1d_fluxerr[:igap+1], x1d_fill, x1d_fluxerr[igap+1:]])
        return x1d_wave, x1d_flux, x1d_fluxerr, s2d_data

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
        splitter = QSplitter(Qt.Vertical)

        # 2D spectrum display
        self.plot_2d = SpectrumPlotWidget(is_2d=True)
        self.plot_2d.setAspectLocked(False)
        self.image_item = pg.ImageItem()
        self.plot_2d.addItem(self.image_item)

        # Hide bottom axis; add top rest-frame axis in Å
        try:
            pi2 = self.plot_2d.getPlotItem()
            pi2.showAxis('bottom', False)
            pi2.showAxis('top', True)
            self.top_axis = RestFrameAxisItem('top', get_z=lambda: self.z_input.value() if self.z_input else 0.0)
            # replace default top axis
            old_top = pi2.getAxis('top')
            pi2.layout.removeItem(old_top)
            pi2.axes['top']['item'] = self.top_axis
            pi2.layout.addItem(self.top_axis, 1, 1)
            self.top_axis.linkToView(pi2.vb)
            pi2.vb.sigResized.connect(lambda: self.top_axis.linkToView(pi2.vb))
        except Exception:
            self.top_axis = None

        # Set colormap
        colors = [
            (0, 0, 180),   # Dark blue
            (100, 150, 255),  # Light blue
            (255, 255, 255),  # White
            (255, 150, 100),  # Light red
            (180, 0, 0)    # Dark red
        ]
        cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)),
                           color=colors)
        self.image_item.setLookupTable(cmap.getLookupTable())
        splitter.addWidget(self.plot_2d)

        # connect 2D hover to update coordinates
        try:
            self.plot_2d.scene().sigMouseMoved.connect(self._on_2d_mouse_moved)
        except Exception:
            pass

        # 2D -> 1D X sync on any range change
        try:
            self.plot_2d.getViewBox().sigRangeChanged.connect(self._on_2d_range_changed)
        except Exception:
            pass

        # 1D spectrum display
        self.plot_1d = SpectrumPlotWidget()
        # Sync x axes: 1D -> 2D
        self.plot_1d.x_range_changed.connect(self.sync_x_range_to_2d)

        # Track ranges for nav stack
        try:
            self.plot_1d.getViewBox().sigRangeChanged.connect(self._on_any_range_changed)
            self.plot_2d.getViewBox().sigRangeChanged.connect(self._on_any_range_changed)
        except Exception:
            pass

        # Cursor line and dot
        self.cursor_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('y', width=1))
        self.cursor_dot = pg.ScatterPlotItem(size=8, brush='y', pen='w')
        self.plot_1d.addItem(self.cursor_line)
        self.plot_1d.addItem(self.cursor_dot)
        self.cursor_line.hide()
        self.cursor_dot.hide()
        splitter.addWidget(self.plot_1d)

        # Alt-selection: set Y-range on 1D
        try:
            self.plot_1d.alt_selected.connect(lambda ymin, ymax: self.plot_1d.setYRange(ymin, ymax, padding=0))
            self.plot_2d.alt_selected.connect(lambda ymin, ymax: self.plot_1d.setYRange(ymin, ymax, padding=0))
        except Exception:
            pass

        # Set splitter sizes (1:3 ratio)
        splitter.setSizes([225, 675])
        layout.addWidget(splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Ready. Use File menu to load FITS data.')

        # Coordinate label
        self.coord_label = QLabel("Wavelength: --- μm, Flux: --- Jy")
        self.coord_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; }")
        self.coord_label.setMinimumWidth(250)
        self.status_bar.addPermanentWidget(self.coord_label)

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

        # Colormap selection for 2D
        left_layout.addWidget(QLabel("2D Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(['RdBu', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'])
        self.cmap_combo.currentTextChanged.connect(self.change_colormap)
        left_layout.addWidget(self.cmap_combo)
        left_layout.addSpacing(14)

        # Sigma clipping
        left_layout.addWidget(QLabel("Sigma Clip:"))
        self.sigma_spin = QSpinBox()
        self.sigma_spin.setRange(1, 10)
        self.sigma_spin.setValue(5)
        self.sigma_spin.valueChanged.connect(self.update_display)
        left_layout.addWidget(self.sigma_spin)
        left_layout.addSpacing(14)

        # Show error bars
        self.show_errors_check = QCheckBox("Show Error Bars")
        self.show_errors_check.setChecked(True)
        self.show_errors_check.toggled.connect(self.update_display)
        left_layout.addWidget(self.show_errors_check)
        left_layout.addSpacing(14)

        # Autoscale Y on zoom
        self.autoscale_y_check = QCheckBox("Autoscale Y on Zoom")
        self.autoscale_y_check.setChecked(False)
        self.autoscale_y_check.toggled.connect(self.on_autoscale_toggled)
        left_layout.addWidget(self.autoscale_y_check)
        left_layout.addSpacing(14)

        # Redshift input for rest-frame axis
        left_layout.addWidget(QLabel("z:"))
        self.z_input = QDoubleSpinBox()
        self.z_input.setDecimals(4)
        self.z_input.setRange(-0.99, 12.0)
        self.z_input.setSingleStep(0.001)
        self.z_input.setValue(0.0)
        # Immediate axis refresh on z change
        self.z_input.valueChanged.connect(self._refresh_top_axis)
        left_layout.addWidget(self.z_input)

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
        """Load s2d/x1d FITS pair"""
        base_dir = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)

        # Determine if this is s2d or x1d and find the pair
        if 's2d' in base_name.lower():
            s2d_path = file_path
            x1d_path = file_path.replace('s2d', 'x1d').replace('S2D', 'X1D')
            if not os.path.exists(x1d_path):
                x1d_path = file_path.replace('_s2d', '_x1d').replace('_S2D', '_X1D')
        elif 'x1d' in base_name.lower():
            x1d_path = file_path
            s2d_path = file_path.replace('x1d', 's2d').replace('X1D', 'S2D')
            if not os.path.exists(s2d_path):
                s2d_path = file_path.replace('_x1d', '_s2d').replace('_X1D', '_S2D')
        else:
            # Try to load single file
            s2d_path = file_path
            x1d_path = None

        # Load s2d data
        if os.path.exists(s2d_path):
            self.load_s2d_data(s2d_path)
            self.current_s2d_file = s2d_path

        # Load x1d data if available
        if x1d_path and os.path.exists(x1d_path):
            self.load_x1d_data(x1d_path)
            self.current_x1d_file = x1d_path
        elif self.s2d_data is not None:
            # Extract 1D from 2D if no x1d file
            self.extract_1d_from_2d()

        # Try to expand wavelength gaps for alignment
        if self.x1d_wave is not None and self.s2d_data is not None:
            self.x1d_wave, self.x1d_flux, self.x1d_fluxerr, self.s2d_data = \
                self.expand_wavelength_gap(self.x1d_wave, self.x1d_flux, self.x1d_fluxerr,
                                           self.s2d_data, expand_wavelength_gap=True)

        self.update_display()

        # Update status
        status_msg = []
        if self.current_s2d_file:
            status_msg.append(f"S2D: {os.path.basename(self.current_s2d_file)}")
        if self.current_x1d_file:
            status_msg.append(f"X1D: {os.path.basename(self.current_x1d_file)}")
        self.status_bar.showMessage("  ".join(status_msg))

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

    def update_display(self):
        """Update both 2D and 1D displays"""
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
            # Only reset image data; hold view range
            self.image_item.setImage(self.s2d_data.T, autoLevels=False)
            self.image_item.setLevels((vmin, vmax))

            # Position the image using spectral edges so 2D columns align with 1D step centers
            if self.x1d_wave is not None:
                w = np.array(self.x1d_wave, dtype=float)
                self._x_edges = self._compute_bin_edges(w)
                rect = pg.QtCore.QRectF(self._x_edges[0], 0, self._x_edges[-1] - self._x_edges[0], ny)
                self.image_item.setRect(rect)

            # Set data limits for 2D (with minXRange ~ 3 bins, minYRange ~ 3 rows)
            if self.x1d_wave is not None and len(self.x1d_wave) > 1:
                med_dx = float(np.median(np.diff(self.x1d_wave)))
            else:
                med_dx = None
            self.plot_2d.set_data_limits(self._x_edges[0], self._x_edges[-1], 0, ny,
                                         min_dx=med_dx, min_bins=min_bins_default, min_y_rows=min_y_rows_default)
            self.plot_2d.setYRange(0, ny, padding=0)  # Fixed initial Y range

        # Update 1D display (as steps using edges, no rebin of flux)
        if self.x1d_flux is not None and self.x1d_wave is not None:
            if not hasattr(self, "_x_edges") or self._x_edges is None or len(self._x_edges) != len(self.x1d_wave) + 1:
                self._x_edges = self._compute_bin_edges(self.x1d_wave)

            # Flux step curve
            if not hasattr(self, 'flux_curve') or self.flux_curve is None:
                self.flux_curve = self.plot_1d.plot(self._x_edges, self.x1d_flux,
                                                    stepMode=True, pen='w', name='Flux')
            else:
                self.flux_curve.setData(self._x_edges, self.x1d_flux, stepMode=True)

            # Error step curve
            if self.show_errors_check.isChecked() and self.x1d_fluxerr is not None:
                if not hasattr(self, 'err_curve') or self.err_curve is None:
                    self.err_curve = self.plot_1d.plot(self._x_edges, self.x1d_fluxerr,
                                                       stepMode=True,
                                                       pen=pg.mkPen('r', width=0.5), name='Error')
                else:
                    self.err_curve.setData(self._x_edges, self.x1d_fluxerr, stepMode=True)
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
                self.zero_curve = self.plot_1d.plot(self._x_edges, zeros,
                                                    stepMode=True,
                                                    pen=pg.mkPen('gray', width=0.5, style=Qt.DashLine))
            else:
                self.zero_curve.setData(self._x_edges, zeros, stepMode=True)

            # Set data limits (min X range ~ 3 bins)
            med_dx1d = float(np.median(np.diff(self.x1d_wave))) if len(self.x1d_wave) > 1 else None
            self.plot_1d.set_data_limits(self._x_edges[0], self._x_edges[-1],
                                         min_dx=med_dx1d, min_bins=min_bins_default, min_y_rows=None)

            # Initial view only on the first draw
            if not self._have_plotted:
                self.plot_1d.setXRange(self._x_edges[0], self._x_edges[-1], padding=0)
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

    def sync_x_range_to_2d(self, xmin, xmax):
        """Synchronize 2D plot x range with 1D plot"""
        if self.s2d_data is not None:
            if self._syncing_x:
                return
            self._syncing_x = True
            try:
                self.plot_2d.setXRange(xmin, xmax, padding=0)
            finally:
                self._syncing_x = False
        # Autoscale Y if enabled
        if self.autoscale_y_check.isChecked() and self.x1d_flux is not None:
            mask = (self.x1d_wave >= xmin) & (self.x1d_wave <= xmax)
            if np.any(mask):
                visible_flux = self.x1d_flux[mask]
                flux_min = np.nanmin(visible_flux)
                flux_max = np.nanmax(visible_flux)
                margin = (flux_max - flux_min) * 0.1
                self.plot_1d.setYRange(min(flux_min - margin, -margin),
                                       flux_max + margin, padding=0)

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

    def on_mouse_moved(self, pos):
        """Update cursor position and coordinates"""
        if self.x1d_wave is None or self.x1d_flux is None:
            return
        mouse_point = self.plot_1d.getViewBox().mapSceneToView(pos)
        x = mouse_point.x()
        if x >= self.x1d_wave.min() and x <= self.x1d_wave.max():
            idx = np.argmin(np.abs(self.x1d_wave - x))
            wave = self.x1d_wave[idx]
            flux = self.x1d_flux[idx]
            # Center yellow line on the step center
            self.cursor_line.setPos(wave)
            self.cursor_dot.setData([wave], [flux])
            self.cursor_line.show()
            self.cursor_dot.show()
            if not np.isnan(flux):
                self.coord_label.setText(f"Wavelength: {wave:.3f} μm, Flux: {flux:.3e} Jy")
            else:
                self.coord_label.setText(f"Wavelength: {wave:.3f} μm, Flux: ---")
        else:
            self.cursor_line.hide()
            self.cursor_dot.hide()
            self.coord_label.setText("Wavelength: --- μm, Flux: --- Jy")

    def change_colormap(self, cmap_name):
        """Change 2D colormap"""
        if cmap_name == 'RdBu':
            colors = [
                (0, 0, 180),  # Dark blue
                (100, 150, 255),  # Light blue
                (255, 255, 255),  # White
                (255, 150, 100),  # Light red
                (180, 0, 0)  # Dark red
            ]
            cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)),
                               color=colors)
        else:
            cmap = pg.colormap.get(cmap_name)
        self.image_item.setLookupTable(cmap.getLookupTable())

    def reset_views(self):
        """Reset all views to default"""
        if self.x1d_wave is not None:
            self.plot_1d.setXRange(self._x_edges[0], self._x_edges[-1], padding=0)
            self.plot_2d.setXRange(self._x_edges[0], self._x_edges[-1], padding=0)
        if self.s2d_data is not None:
            ny, nx = self.s2d_data.shape
            self.plot_2d.setYRange(0, ny, padding=0)
        if self.x1d_flux is not None:
            flux_min = np.nanmin(self.x1d_flux)
            flux_max = np.nanmax(self.x1d_flux)
            margin = (flux_max - flux_min) * 0.1
            self.plot_1d.setYRange(min(flux_min - margin, -margin),
                                   flux_max + margin, padding=0)
        self._push_nav_state()

    def _on_2d_mouse_moved(self, pos):
        # Map scene to view
        try:
            vb = self.plot_2d.getViewBox()
            mouse_point = vb.mapSceneToView(pos)
            x = mouse_point.x()
            y = mouse_point.y()
            if self.x1d_wave is not None and self.s2d_data is not None:
                ix = np.argmin(np.abs(self.x1d_wave - x))
                iy = int(round(y))
                ny, nx = self.s2d_data.shape
                if 0 <= ix < nx and 0 <= iy < ny:
                    val = self.s2d_data[iy, ix]
                    try:
                        f1 = self.x1d_flux[ix] if self.x1d_flux is not None else np.nan
                        self.coord_label.setText(f"λ: {x:.6f} Flux1D: {f1:.4g} 2D[{iy},{ix}]: {val:.4g}")
                    except Exception:
                        self.coord_label.setText(f"λ: {x:.6f}")
                    # Move cursor on 1D to center of step
                    try:
                        if self.cursor_line is not None and self.cursor_dot is not None:
                            cx = self.x1d_wave[ix]
                            self.cursor_line.setPos(cx)
                            self.cursor_line.show()
                            if self.x1d_flux is not None and ix < len(self.x1d_flux):
                                self.cursor_dot.setData([cx], [self.x1d_flux[ix]])
                                self.cursor_dot.show()
                    except Exception:
                        pass
                    return
            # fallback
            self.coord_label.setText(f"λ: {x:.6f}")
        except Exception:
            pass

    def autoscale(self):
        """Autoscale displays"""
        self.reset_views()

    def _refresh_top_axis(self, *_):
        try:
            if hasattr(self, 'top_axis') and self.top_axis is not None:
                # Force tick recomputation and repaint immediately
                self.top_axis.picture = None
                self.top_axis.update()
        except Exception:
            pass


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
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        try:
            win.load_fits_pair(filename)
        except Exception as e:
            QMessageBox.critical(win, 'Error', f'Failed to load file:\n{e}')
    win.show()
    sys.exit(app.exec_())
