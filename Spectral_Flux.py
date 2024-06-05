"""
SpectralFlux.py
Compute the spectral flux between consecutive spectra
This technique can be for onset detection
rectify - only return positive values
"""
import numpy as np

def spectralFlux(spectra, rectify=False):
    """
    Compute the spectral flux between consecutive spectra
    """
    spectralFlux = []

    # Compute flux for zeroth spectrum
    flux = 0
    for bin in spectra:
        flux = flux + abs(bin)
        spectralFlux.append(flux)
    # Compute flux for subsequent spectra
    spectralFluxs = []
    for s in range(len(spectra)):
        prevSpectrum = spectra[s]
        spectrum = spectra[s]
        flux = 0
        diff = abs(spectrum) - abs(prevSpectrum)
        # If rectify is specified, only return positive values
        if rectify and diff < 0:
            diff = 0
        flux = flux + diff
        spectralFluxs.append(flux)
    return np.mean(spectralFluxs)
