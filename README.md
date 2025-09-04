# Spectrum Viewer

A lightweight interactive FITS spectrum viewer built with **PyQt5** and **pyqtgraph**.  
It provides both 2D and 1D spectral visualization with smooth, responsive controls.

---

## Main Features

### Trackpad Gestures
- **Two-finger scroll up/down**: Zoom in/out on wavelength (X-axis)  
- **Two-finger scroll left/right**: Pan along wavelength (X-axis)  
- **Ctrl + scroll**: Scale flux (Y-axis) — simulates pinch gesture  

### Display Components
**2D Spectrum Display (top panel):**
- Shows the rectified 2D spectrum with customizable colormaps  
- Interactive zoom and pan capabilities  
- The Y-axis (pixel row) is fixed to always show the full range.

**1D Spectrum Display (bottom panel):**
- Shows extracted 1D flux vs wavelength  
- Optional error bars display  
- Zero-line reference  

### Controls
**File Operations**
- Open FITS files via menu or toolbar  

**Display Options**
- Colormap selection for 2D display (`RdBu` default, plus `viridis`, `plasma`, etc.)  
- Sigma clipping control for display range  
- Toggle error bars  
- Toggle wavelength gap expansion  

**View Controls**
- Reset view (**R**)  
- Autoscale (**A**)  

### Data Processing
- Automatic sigma clipping for optimal display range  
- Wavelength gap detection and expansion (as in your reference code)  
- NaN handling for missing data  
- Simple 1D extraction from 2D data (center rows)  

---

## Requirements

```bash
pip install PyQt5 pyqtgraph astropy numpy