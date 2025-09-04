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
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QLabel,
                             QSplitter, QStatusBar, QToolBar, QAction,
                             QComboBox, QSpinBox, QCheckBox, QGroupBox,
                             QSlider, QMessageBox)
from PyQt5.QtCore import Qt, QPointF, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QKeySequence
import pyqtgraph as pg
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.visualization import simple_norm
import warnings
warnings.filterwarnings('ignore')

pg.setConfigOptions(antialias=True)


class SpectrumPlotWidget(pg.PlotWidget):
    """Custom plot widget with trackpad gesture support"""
    
    # Signal emitted when x range changes
    x_range_changed = pyqtSignal(float, float)
    
    def __init__(self, parent=None, is_2d=False):
        super().__init__(parent)
        
        self.is_2d = is_2d
        
        if not is_2d:
            self.setLabel('left', 'Flux', units='Jy')
        else:
            self.setLabel('left', 'Pixel Row')
        self.setLabel('bottom', 'Wavelength', units='μm')
        self.showGrid(x=True, y=True, alpha=0.3)
        
        # Enable mouse tracking for gestures
        self.viewport().setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.setMouseTracking(True)
        
        # Store data limits
        self.data_x_min = None
        self.data_x_max = None
        self.data_y_min = None
        self.data_y_max = None
        
        # Track mouse position for zoom center
        self.mouse_x_pos = None
        
        # Connect range change signal
        self.getViewBox().sigRangeChanged.connect(self.on_range_changed)
        
    def set_data_limits(self, x_min, x_max, y_min=None, y_max=None):
        """Set the data limits to restrict panning/zooming"""
        self.data_x_min = x_min
        self.data_x_max = x_max
        self.data_y_min = y_min
        self.data_y_max = y_max
        
    def mouseMoveEvent(self, ev):
        """Track mouse position for zoom centering"""
        if self.sceneBoundingRect().contains(ev.pos()):
            mouse_point = self.getViewBox().mapSceneToView(ev.pos())
            self.mouse_x_pos = mouse_point.x()
        super().mouseMoveEvent(ev)
        
    def wheelEvent(self, ev):
        """Handle trackpad/mouse wheel events for zoom and pan"""
        modifiers = QApplication.keyboardModifiers()
        
        # Get scroll delta
        delta = ev.angleDelta()
        dx = delta.x() / 120.0  # Horizontal scroll
        dy = delta.y() / 120.0  # Vertical scroll
        
        # Get current view range
        vb = self.getViewBox()
        xmin, xmax = vb.viewRange()[0]
        ymin, ymax = vb.viewRange()[1]
        
        if modifiers == Qt.NoModifier:
            # Two finger scroll up/down: Zoom X (wavelength)
            if abs(dy) > 0 and not self.is_2d:  # Only for 1D plot
                scale_factor = 1.1 ** (-dy)
                
                # Use mouse position as zoom center if available
                if self.mouse_x_pos is not None:
                    x_center = self.mouse_x_pos
                    # Keep the wavelength at cursor fixed
                    left_ratio = (x_center - xmin) / (xmax - xmin)
                    right_ratio = 1 - left_ratio
                    
                    new_x_range = (xmax - xmin) * scale_factor
                    new_xmin = x_center - new_x_range * left_ratio
                    new_xmax = x_center + new_x_range * right_ratio
                else:
                    # Fallback to center zoom
                    x_center = (xmin + xmax) / 2
                    x_range = xmax - xmin
                    new_x_range = x_range * scale_factor
                    new_xmin = x_center - new_x_range/2
                    new_xmax = x_center + new_x_range/2
                
                # Apply limits
                if self.data_x_min is not None and self.data_x_max is not None:
                    new_xmin = max(new_xmin, self.data_x_min)
                    new_xmax = min(new_xmax, self.data_x_max)
                
                vb.setXRange(new_xmin, new_xmax, padding=0)
            
            # Two finger scroll left/right: Pan X (wavelength)
            if abs(dx) > 0 and not self.is_2d:  # Only for 1D plot
                x_shift = (xmax - xmin) * dx * 0.1
                new_xmin = xmin + x_shift
                new_xmax = xmax + x_shift
                
                # Apply limits
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
                
        elif modifiers == Qt.ControlModifier and not self.is_2d:
            # Ctrl + scroll: Scale Y (flux) - simulating pinch (only for 1D)
            if abs(dy) > 0:
                scale_factor = 1.1 ** (-dy)
                y_center = (ymin + ymax) / 2
                y_range = ymax - ymin
                new_y_range = y_range * scale_factor
                vb.setYRange(y_center - new_y_range/2, y_center + new_y_range/2, padding=0)
        
        ev.accept()
        
    def on_range_changed(self):
        """Emit signal when x range changes"""
        if not self.is_2d:  # Only emit from 1D plot
            xmin, xmax = self.getViewBox().viewRange()[0]
            self.x_range_changed.emit(xmin, xmax)
            
    def constrain_x_range(self):
        """Constrain x range to data limits"""
        if self.data_x_min is not None and self.data_x_max is not None:
            xmin, xmax = self.getViewBox().viewRange()[0]
            xmin = max(xmin, self.data_x_min)
            xmax = min(xmax, self.data_x_max)
            self.getViewBox().setXRange(xmin, xmax, padding=0)


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
        
        # For cursor tracking
        self.cursor_line = None
        self.cursor_dot = None
        self.coord_label = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('FITS Spectra Viewer')
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create toolbar
        self.create_toolbar()
        
        # Create control panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # Create splitter for 2D and 1D plots
        splitter = QSplitter(Qt.Vertical)
        
        # 2D spectrum display (using PlotWidget with ImageItem)
        self.plot_2d = SpectrumPlotWidget(is_2d=True)
        self.plot_2d.setAspectLocked(False)
        self.image_item = pg.ImageItem()
        self.plot_2d.addItem(self.image_item)
        
        # Set colormap
        colors = [
            (0, 0, 180),    # Dark blue
            (100, 150, 255), # Light blue  
            (255, 255, 255), # White
            (255, 150, 100), # Light red
            (180, 0, 0)      # Dark red
        ]
        cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), 
                          color=colors)
        self.image_item.setLookupTable(cmap.getLookupTable())
        
        splitter.addWidget(self.plot_2d)
        
        # 1D spectrum display
        self.plot_1d = SpectrumPlotWidget()
        
        # Connect range change signal to sync x axes
        self.plot_1d.x_range_changed.connect(self.sync_x_range_to_2d)
        
        # Add cursor tracking
        self.plot_1d.scene().sigMouseMoved.connect(self.on_mouse_moved)
        
        # Add cursor line and dot
        self.cursor_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('y', width=1))
        self.cursor_dot = pg.ScatterPlotItem(size=8, brush='y', pen='w')
        self.plot_1d.addItem(self.cursor_line)
        self.plot_1d.addItem(self.cursor_dot)
        self.cursor_line.hide()
        self.cursor_dot.hide()
        
        splitter.addWidget(self.plot_1d)
        
        # Set splitter sizes (1:3 ratio as in original)
        splitter.setSizes([225, 675])
        layout.addWidget(splitter)
        
        # Status bar with coordinate display
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Ready. Use File menu to load FITS data.')
        
        # Add coordinate label
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
        
        left_layout.addSpacing(20)
        
        # Sigma clipping
        left_layout.addWidget(QLabel("Sigma Clip:"))
        self.sigma_spin = QSpinBox()
        self.sigma_spin.setRange(1, 10)
        self.sigma_spin.setValue(5)
        self.sigma_spin.valueChanged.connect(self.update_display)
        left_layout.addWidget(self.sigma_spin)
        
        left_layout.addSpacing(20)
        
        # Show error bars
        self.show_errors_check = QCheckBox("Show Error Bars")
        self.show_errors_check.setChecked(True)
        self.show_errors_check.toggled.connect(self.update_display)
        left_layout.addWidget(self.show_errors_check)
        
        left_layout.addSpacing(20)
        
        # Autoscale Y on zoom
        self.autoscale_y_check = QCheckBox("Autoscale Y on Zoom")
        self.autoscale_y_check.setChecked(False)
        self.autoscale_y_check.toggled.connect(self.on_autoscale_toggled)
        left_layout.addWidget(self.autoscale_y_check)
        
        main_layout.addLayout(left_layout)
        main_layout.addStretch()
        
        # Right side - Gestures text
        gestures_label = QLabel(
            "Gestures: Scroll up/down = Zoom X | Scroll left/right = Pan X | "
            "Ctrl+Scroll = Scale Y"
        )
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
            
        self.update_display()
        
        # Update status
        status_msg = []
        if self.current_s2d_file:
            status_msg.append(f"S2D: {os.path.basename(self.current_s2d_file)}")
        if self.current_x1d_file:
            status_msg.append(f"X1D: {os.path.basename(self.current_x1d_file)}")
        self.status_bar.showMessage(" | ".join(status_msg))
        
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
            # Look for spectral data - adapt this based on your FITS structure
            for hdu in hdul:
                if hasattr(hdu, 'data') and hdu.data is not None:
                    if 'WAVELENGTH' in hdu.columns.names:
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
                        header = dict(hdu.header) # Convert header to dict to avoid method name collisions
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
        
    def update_display(self):
        """Update both 2D and 1D displays"""
        if self.s2d_data is None and self.x1d_flux is None:
            return
            
        # Update 2D display
        if self.s2d_data is not None:
            # Sigma clipping for display range
            sigma_val = self.sigma_spin.value()
            try:
                clipped = sigma_clip(self.s2d_data[~np.isnan(self.s2d_data)], 
                                   sigma=sigma_val, maxiters=3)
                vmin, vmax = np.min(clipped), np.max(clipped)
            except:
                vmin, vmax = np.nanmin(self.s2d_data), np.nanmax(self.s2d_data)
            
            # Set image with proper origin (lower)
            ny, nx = self.s2d_data.shape
            self.image_item.setImage(self.s2d_data.T, levels=(vmin, vmax))
            
            # Set up transform to match wavelength scale if we have it
            if self.x1d_wave is not None:
                # Use setRect to define the image's position and scale in one step.
                # This is more robust and avoids the problematic .scale() call.
                rect = pg.QtCore.QRectF(self.x1d_wave[0], 0, self.x1d_wave[-1] - self.x1d_wave[0], ny)
                self.image_item.setRect(rect)

                # Set data limits for 2D plot
                self.plot_2d.set_data_limits(self.x1d_wave[0], self.x1d_wave[-1], 0, ny)
                self.plot_2d.setYRange(0, ny, padding=0)  # Fixed Y range
            
        # Update 1D display
        if self.x1d_flux is not None and self.x1d_wave is not None:
            self.plot_1d.clear()
            
            # Keep cursor items
            self.plot_1d.addItem(self.cursor_line)
            self.plot_1d.addItem(self.cursor_dot)
            
            # Plot flux in step mode
            self.flux_curve = self.plot_1d.plot(self.x1d_wave, self.x1d_flux[:-1], 
                                               stepMode='center', pen='w', name='Flux')
            
            # Plot error bars if available and requested
            if self.show_errors_check.isChecked() and self.x1d_fluxerr is not None:
                self.plot_1d.plot(self.x1d_wave, self.x1d_fluxerr[:-1], 
                                stepMode='center', pen=pg.mkPen('r', width=0.5), 
                                name='Error')
            
            # Add zero line
            self.plot_1d.plot(self.x1d_wave, np.zeros_like(self.x1d_wave), 
                            pen=pg.mkPen('gray', width=0.5, style=Qt.DashLine))
            
            # Set data limits
            self.plot_1d.set_data_limits(self.x1d_wave.min(), self.x1d_wave.max())
            
            # Set initial view
            self.plot_1d.setXRange(self.x1d_wave.min(), self.x1d_wave.max(), padding=0)
            
            # Set Y range with margin below zero
            flux_min = np.nanmin(self.x1d_flux)
            flux_max = np.nanmax(self.x1d_flux)
            margin = (flux_max - flux_min) * 0.1
            self.plot_1d.setYRange(min(flux_min - margin, -margin), 
                                  flux_max + margin, padding=0)
                                  
    def sync_x_range_to_2d(self, xmin, xmax):
        """Synchronize 2D plot x range with 1D plot"""
        if self.s2d_data is not None:
            self.plot_2d.setXRange(xmin, xmax, padding=0)
            
        # Autoscale Y if enabled
        if self.autoscale_y_check.isChecked() and self.x1d_flux is not None:
            # Find flux range in visible x range
            mask = (self.x1d_wave >= xmin) & (self.x1d_wave <= xmax)
            if np.any(mask):
                visible_flux = self.x1d_flux[mask]
                flux_min = np.nanmin(visible_flux)
                flux_max = np.nanmax(visible_flux)
                margin = (flux_max - flux_min) * 0.1
                # Always include zero with negative margin
                self.plot_1d.setYRange(min(flux_min - margin, -margin), 
                                      flux_max + margin, padding=0)
                                      
    def on_autoscale_toggled(self):
        """Handle autoscale Y checkbox toggle"""
        if self.autoscale_y_check.isChecked():
            # Trigger autoscale with current range
            xmin, xmax = self.plot_1d.getViewBox().viewRange()[0]
            self.sync_x_range_to_2d(xmin, xmax)
            
    def on_mouse_moved(self, pos):
        """Update cursor position and coordinates"""
        if self.x1d_wave is None or self.x1d_flux is None:
            return
            
        # Get mouse position in data coordinates
        mouse_point = self.plot_1d.getViewBox().mapSceneToView(pos)
        x = mouse_point.x()
        
        # Check if within data range
        if x >= self.x1d_wave.min() and x <= self.x1d_wave.max():
            # Find nearest data point
            idx = np.argmin(np.abs(self.x1d_wave - x))
            wave = self.x1d_wave[idx]
            flux = self.x1d_flux[idx]
            
            # Update cursor line and dot
            self.cursor_line.setPos(wave)
            self.cursor_dot.setData([wave], [flux])
            self.cursor_line.show()
            self.cursor_dot.show()
            
            # Update coordinate display
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
                (0, 0, 180),    # Dark blue
                (100, 150, 255), # Light blue
                (255, 255, 255), # White
                (255, 150, 100), # Light red
                (180, 0, 0)      # Dark red
            ]
            cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), 
                              color=colors)
        else:
            cmap = pg.colormap.get(cmap_name)
            
        self.image_item.setLookupTable(cmap.getLookupTable())
        
    def reset_views(self):
        """Reset all views to default"""
        if self.x1d_wave is not None:
            self.plot_1d.setXRange(self.x1d_wave.min(), self.x1d_wave.max(), padding=0)
            self.plot_2d.setXRange(self.x1d_wave.min(), self.x1d_wave.max(), padding=0)
            
        if self.s2d_data is not None:
            ny, nx = self.s2d_data.shape
            self.plot_2d.setYRange(0, ny, padding=0)
            
        if self.x1d_flux is not None:
            flux_min = np.nanmin(self.x1d_flux)
            flux_max = np.nanmax(self.x1d_flux)
            margin = (flux_max - flux_min) * 0.1
            self.plot_1d.setYRange(min(flux_min - margin, -margin), 
                                  flux_max + margin, padding=0)
        
    def autoscale(self):
        """Autoscale displays"""
        self.reset_views()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    viewer = FITSSpectraViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()