"""Dust attenuation tables for simple galactic geometries.

Computes the fraction of starlight escaping a galaxy as a function of
wavelength, inclination, dust optical depth and component size, by running the
Hyperion Monte Carlo radiative transfer code over a grid of simple galactic
geometries, and tabulates the result.

The method is described in `Benson (2018)
<https://ui.adsabs.harvard.edu/abs/2018RNAAS...2..188B>`_.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
