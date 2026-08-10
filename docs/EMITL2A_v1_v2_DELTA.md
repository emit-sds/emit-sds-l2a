# Earth surface Mineral dust source InvesTigation (EMIT)

## EMIT L2A: Surface Reflectance

### Version 1 to Version 2 Transition Document

David R. Thompson<sup>1</sup>, Philip G. Brodrick<sup>1</sup>, Robert O. Green<sup>1</sup>, Olga Kalashnikova<sup>1</sup>, Sarah Lundeen<sup>1</sup>, Gregory Okin<sup>2</sup>, Winston Olson-Duvall<sup>1</sup>, Thomas Painter<sup>2</sup>

<sup>1</sup> Jet Propulsion Laboratory, California Institute of Technology

<sup>2</sup> University of California, Los Angeles

Version 2.0
August 2026

Jet Propulsion Laboratory
California Institute of Technology
Pasadena, California 91109-8099


## Table of Contents
1. [Reflectance comparison](#1-reflectance-comparison)
2. [Summary of changes](#2-summary-of-changes)
    - 2.1. [Updated Radiative Transfer Model](#21-updated-radiative-transfer-model)
    - 2.2. [Pre-cached global look-up tables](#22-pre-cached-luts)
    - 2.3. [Empirical orthogonal functions (EOFs)](#23-empirical-orthogonal-functions)
    - 2.4. [Edited Surface Reflectance Statistical Prior](#24-edited-surface-reflectance-statistical-prior)
    - 2.5. [Variable atmospheric carbon dioxide concentration ($CO_2$)](#25-variable-atmospheric-co2)
    - 2.6. [Constrained Aerosol Optical Depth Prior Variance](#26-constrained-aerosol-optical-depth-prior-variance)
    - 2.7. [Removed pressure elevation from solution state](#27-pressure-elevation)
    - 2.8. [Updated L1B radiometry and wavelength solutions](#28-updated-radiometry-wavelengths)

---

## 1. Reflectance comparison

*TODO: Add in-situ comparison exercise. We have reference reflectance match-ups for V2. Can we compare to V1?*

## 2. Summary of Changes

### 2.1. Updated Radiative Transfer Model (RTM)



<p align="center">
    <img src="img_v1_v2_delta/fig01.png" width="100%", alt="Figure 1">
</p>

*Figure 1. (top row) Modeled photon path-specific transmittances at varying atmospheric water vapor concentration. (bottom row) Coupled atmospheric radiances calculated from the path-separated transmittances.*

<p align="center">
    <img src="img_v1_v2_delta/fig02.png" width="100%", alt="Figure 2">
</p>

*Figure 2. (top row) Modeled photon path-specific transmittances at varying Aerosol optical depth. (bottom row) Coupled atmospheric radiances calculated from the path-separated transmittances.*

<p align="center">
    <img src="img_v1_v2_delta/fig03.png" width="100%", alt="Figure 3">
</p>

*Figure 3. Modeled total transmittance with (left) version 1 sRTMnet and (middle) version 2 sRTMnet at varying atmosphere water vapor concentration. (right) Residual difference between version 2 - version 1.*

<p align="center">
    <img src="img_v1_v2_delta/fig04.png" width="100%", alt="Figure 4">
</p>

*Figure 4. Modeled total transmittance with (left) version 1 sRTMnet and (middle) version 2 sRTMnet at varying aerosol optical depth. (right) Residual difference between version 2 - version 1.*

### 2.2. Pre-cached global look-up tables
### 2.3. Empirical orthogonal functions (EOFs)

<p align="center">
    <img src="img_v1_v2_delta/fig05.png" width="80%", alt="Figure 5">
</p>

*Figure 5.*

<p align="center">
    <img src="img_v1_v2_delta/fig06.png" width="100%", alt="Figure 6">
</p>

*Figure 5.*

### 2.4. Edited Surface Reflectance Statistical Prior
### 2.5. Variable atmospheric $CO_2$
### 2.6. Constrained Aerosol Optical Depth Prior Variance
### 2.7. Updated L1B radiometry and wavelength solutions
### 2.8. Removed pressure elevation from solution state

<p align="center">
    <img src="img_v1_v2_delta/fig10.png" width="100%", alt="Figure 10">
</p>

*Figure 10. Comparing spatially interpolated maps of atmospheric variables with (top) pressure elevation turned on and (bot) pressure elevation turned off. The EMIT ID shown here is emit20240626t165035.*

<p align="center">
    <img src="img_v1_v2_delta/fig11.png" width="100%", alt="Figure 11">
</p>

*Figure 11. Histograms of atmospheric solutions.*

<p align="center">
    <img src="img_v1_v2_delta/fig12.png" width="100%", alt="Figure 12">
</p>

*Figure 12. (top) Scene-wide mean and 1.96 * standard deviation reflectance for scenes processed with and without pressure elevation. (bot) Wavelength-specific residual calculated per-pixel as scene processed with pressure elevation - without pressure elevation.*

### 2.9. Edited atmospheric length scales
