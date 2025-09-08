# Spectacle Spectrum Viewer

Smooth simple scrolling of 2D and 1D synced spectra.  
What's in your spectra? See it all with spectacle.

A lightweight interactive FITS spectrum viewer built with **PyQt5** and **pyqtgraph**.
Built originally for JWST NIRSpec MOS spectra, showing both 2D rectified spectra (s2d.fits) and 1D extractions (x1d.fits).

---

## Quick Start

```bash
pip install PyQt5 pyqtgraph astropy numpy

python spectacle.py file_s2d.fits

python spectacle.py jw02736-o007_s06355_nirspec_f290lp-g395m_s2d.fits 7.665 emission\ lines.txt
```

Mention either the `s2d` or `x1d` file, and spectacle will find both, load them, and sync them.
Optionally, include the redshift and an external file of emission lines to plot.
(Currently suppressing Iron Fe and Argon Ar for clarity.)

---

<p align="center">
  <img src="spectacle.png" alt="spectacle screenshot" width="600"/>
</p>

---

## Trackpad Gestures
- **Two-finger scroll ⬆/⬇**: Wavelength zoom (X)
- **Two-finger scroll ⬅/➡**: Wavelength pan (X)
- **Pinch / Ctrl+scroll / Option/Alt+drag**: Flux zoom (Y)
- **Drag**: move
- **Alt ⬅/➡**: Back/Forward (history)
- **R**: Reset
- **A**: Autoscale
