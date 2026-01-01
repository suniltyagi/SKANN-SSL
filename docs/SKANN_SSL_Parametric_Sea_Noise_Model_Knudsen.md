# SKANN-SSL — Parametric Sea-Noise Model from Digitised Knudsen Curves

**Role:** Defines the analytic, piecewise parametric sea-noise model used by SKANN, derived from digitised Knudsen curves.

Parametric Construction of Sea–Noise Spectra from Digitised Knudsen Curves
## 1. Introduction
This document describes the procedure followed to construct a continuous, piecewise parametric model of ocean ambient noise suitable for SKANN simulations. The construction begins with digitisation of reference Knudsen curves, continues with segmentation of the spectral domain into physically meaningful sub–bands, and concludes with log–frequency regression and turbulence modelling. The resulting representation is fully analytic, continuous at all band junctions, and suitable for generating synthetic waveforms or coloured noise.
Placeholder for Knudsen Curve Figure:
2. Digitisation of Reference Curves
The Knudsen chart provides spectral noise levels in dB re 1 Pa/Hz for different sea states. Since this information is typically available as a plot, WebPlotDigitizer (WPD) was used to extract numerical points for sea states SS0, SS1, SS3, and SS6 in the frequency range 10 Hz to 100 kHz.
2.1 WebPlotDigitizer Procedure
Axes were configured as: X–axis logarithmic, Y–axis linear.
Two calibration points were selected on the X–axis (for example 10 Hz and 100 kHz).
Two calibration points were selected on the Y–axis (for example 20 dB and 100 dB).
Each sea–state curve was isolated using colour masking.
Automatic curve extraction was used to obtain a set of (index, frequency, noise level) values.
The resulting CSV files were exported without headers.
Each file contained three columns of the form

3. Loading and Plotting the Digitised Curves
The CSV files are loaded and plotted using Python. The example below shows how the four sea states SS0, SS1, SS3, and SS6 are read and displayed.
import pandas as pd
import matplotlib.pyplot as plt

# Load digitised Knudsen curves for four sea states.
# Each file is: index, frequency_Hz, noise_level_dB (no header row).
SS0 = pd.read_csv("SS0CSV.txt", header=None, names=["idx","f","NL"])
SS1 = pd.read_csv("SS1CSV.txt", header=None, names=["idx","f","NL"])
SS3 = pd.read_csv("SS3CSV.txt", header=None, names=["idx","f","NL"])
SS6 = pd.read_csv("SS6CSV.txt", header=None, names=["idx","f","NL"])

# Ensure increasing frequency order for safety and interpolation.
for df in [SS0, SS1, SS3, SS6]:
    df.sort_values("f", inplace=True)

# Plot all four digitised curves on a semilog frequency axis.
plt.figure(figsize=(12,8))
plt.semilogx(SS0["f"], SS0["NL"], label="SS0")
plt.semilogx(SS1["f"], SS1["NL"], label="SS1")
plt.semilogx(SS3["f"], SS3["NL"], label="SS3")
plt.semilogx(SS6["f"], SS6["NL"], label="SS6")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Noise Level (dB re 1 µPa²/Hz)")
plt.title("Digitised Knudsen Curves for Four Sea States")
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.legend()
plt.show()
Placeholder for Digitised Curves Plot:
Superposition of the four curves confirms the correctness and smoothness of the digitised data across the entire 10 Hz to 100 kHz range.
4. Motivation for Spectral Segmentation
Inspection of the digitised Knudsen sea–noise spectra reveals that no single functional expression accurately models the entire 10 Hz–100 kHz range. Instead, distinct physical mechanisms dominate separate frequency regions, and the spectrum therefore exhibits piecewise power–law behaviour.
The principal regions observed in the Knudsen, Wenz, and Kießling curves are:
Hydrodynamic turbulence below approximately 40–60 Hz, characterised by a steep downward slope in  versus .
Low–frequency (LF) shipping and wind noise, beginning at the turbulence–LF intersection  and extending to about 200 Hz, with a comparatively shallow spectral slope.
Mid–frequency (MF) shoulder, typically spanning 200–800 Hz in classical presentations. For SKANN, the MF plateau used for parametrisation is restricted to 200–500 Hz, where the digitised SS3 data show the flattest spectral region.
High–frequency (HF) wind–sea band, beginning near 500 Hz and extending to 100 kHz, following an approximately straight line with slope  to  dB per decade.
These regions correspond to standard physical interpretations of the Knudsen curve and also yield models that are linear in  space.
Earlier drafts of this document used a 300–800 Hz MF segmentation; however, enforcing an HF join at 300 Hz is physically incorrect and led to an artificial kink. The HF line that best fits the digitised SS3 data intersects the MF region near 500 Hz, not 300 Hz. Therefore the final, physically motivated boundaries adopted for SS3 are

with a true MF plateau between 200–500 Hz and a single straight HF line starting at 500 Hz and continuing to 100 kHz.
This segmentation provides a smooth, stable basis for deriving the parametric SS3 model used in subsequent sections.
5. Log–Frequency Regression and Coefficient Derivation
The Knudsen sea–noise spectrum exhibits approximate power–law behaviour within distinct frequency regions. A power law of the form

becomes linear under the noise–level mapping

Thus each spectral region may be represented by the affine model

where  are obtained either by numerical regression (HF and LF regions) or by enforcing smooth continuity at region boundaries (MF and turbulence regions).
For the corrected SS3 parametrisation, the four regions are:

The goal of this section is to derive the parameters  for the LF and HF regions directly from the digitised WebPlotDigitizer CSV files, and to use continuity conditions to determine the MF plateau level and the turbulence intersection frequency .
5.1 Least–Squares Determination of HF and LF Coefficients
To determine  for the HF and LF regions, we fit the affine model

to the digitised samples .
## Let

Then the regression model becomes

This leads to the linear system

The least–squares solution is

computed numerically using the QR–based routine numpy.linalg.lstsq. This formulation solves ; it does not use .
## Software Dependencies.
Only the numpy library is required for regression; pandas is used solely for reading and sorting CSV files. No SciPy or scikit–learn components are required.
Construction of the Design Matrix via np.vstack.
The design matrix  is assembled in Python using
A_hf = np.vstack([x_hf, np.ones_like(x_hf)]).T
which performs:
construction of row vectors  and ,
vertical stacking into a  matrix,
transposition to obtain the  design matrix .
Least–Squares Solution via np.linalg.lstsq.
The HF regression is carried out as
a_HF, b_HF_raw = np.linalg.lstsq(A_hf, hf["NL"], rcond=None)[0]
which returns the unconstrained HF slope and intercept. A similar call on the LF subset yields .
## Python Implementation Used.
Consistent with the corrected 200–500 Hz plateau and the single HF line starting at 500 Hz, the revised regression bands are:
# HF regression (3 kHz – 20 kHz)
hf = df[(df["f"] >= 3000.0) & (df["f"] <= 20000.0)]
x_hf = np.log10(hf["f"])
A_hf = np.vstack([x_hf, np.ones_like(x_hf)]).T
a_HF, b_HF_raw = np.linalg.lstsq(A_hf, hf["NL"], rcond=None)[0]

# LF regression (100 – 200 Hz)
lf = df[(df["f"] >= 100.0) & (df["f"] <= 200.0)]
x_lf = np.log10(lf["f"])
A_lf = np.vstack([x_lf, np.ones_like(x_lf)]).T
a_LF, b_LF = np.linalg.lstsq(A_lf, lf["NL"], rcond=None)[0]
The HF intercept  is further adjusted in Section 5.2 so that the HF line passes exactly through the MF plateau at 500 Hz.
5.2 Derivation of MF and Turbulence Coefficients
The MF band uses algebraic continuity. Let

Let  be interpolated from the SS3 digitised data in the neighbourhood of 200 Hz, and let  be obtained from the HF regression (Section 5.1). The MF line is then defined by

where the coefficients follow from the two–point construction

Turbulence line.
The turbulence segment is anchored using two physically motivated points on the Knudsen turbulence branch, for example

In  versus  coordinates, the turbulence line takes the form

with coefficients

Intersection frequency.
The turbulence line meets the low–frequency (LF) line at the frequency  defined by

so that

The numerical values of ,  and the resulting  are presented in Section 8, while Sections 9 and 10 describe the mid–frequency plateau and high–frequency segments constructed using these coefficients.
5.3 Summary of Numerical Coefficients (SS3)
The parametric SS3 model consists of four affine components in  versus  space:

with coefficients derived either from least–squares regression (HF and LF), algebraic continuity (MF), or physically selected anchor points (turbulence). The final expressions are:
Turbulence region ().


Low–frequency region ().

where the coefficients  are obtained by least–squares regression of the digitised SS3 points in the 100–200 Hz interval. Their numerical values appear in Section 8 together with the intersection frequency  defined in Section 5.2.
Mid–frequency region ().

with

where  is interpolated from the digitised SS3 data and  is evaluated using the HF regression line. The resulting MF coefficients are reported in Section 9.
High–frequency region ().

where  is obtained via least–squares regression of the SS3 points from 500 Hz to 100 kHz. The intercept  is subsequently adjusted to ensure continuity with the MF segment at 500 Hz. The numerical HF values are presented in Section 10.
The next four sections (7–10) provide the explicit numerical coefficients for the turbulence, LF, MF, and HF regions, together with the corresponding frequency bounds defining the complete SS3 piecewise model.
## 6. Turbulence Region (10 Hz–)
The turbulence band models the steep decay in ambient sea noise at the lowest frequencies. Following the classical Knudsen formulation, this region is defined using two physically motivated anchor points,

which lie on the turbulence-dominated portion of the spectrum.
In  versus  coordinates, the turbulence line is

Numeric coefficients.
Using the anchor points,

The turbulence branch is taken to extend from 10 Hz up to the frequency  at which it meets the low–frequency (LF) regression line. This intersection satisfies

and its numerical value is obtained in Section 7 once the LF coefficients are known.
## 7. Low–Frequency Region (–200 Hz)
The low–frequency (LF) region represents the transition from the steep turbulence slope into the mid–frequency shoulder. It is modelled as an affine function in ,

Numeric coefficients from SS3 regression.
A least–squares regression of the digitised SS3 points between 100 Hz and 200 Hz yields

Intersection with the turbulence branch.
The lower boundary  is obtained by solving

hence

Substituting ,  and ,  gives

Thus the LF region spans approximately 35 Hz to 200 Hz in the SS3 model.
8. High–Frequency Region (500 Hz–10 kHz)
The high–frequency (HF) region captures the wind–driven and breaking–wave noise components of the Knudsen spectrum. Over the range

the digitised SS3 curve is well approximated by a straight line in  versus  space.
Accordingly, the HF band is fitted using a least–squares regression over all SS3 points in this interval:

Numeric coefficients (SS3 regression).
Regression over 500–10 kHz gives

Evaluating the HF fit at 500 Hz gives

This value forms the right–hand anchor for the mid–frequency (MF) segment constructed in Section 9.
## 9. Mid–Frequency Region (200–500 Hz)
The digitised SS3 spectrum exhibits a pronounced flattening between 200 Hz and 500 Hz, forming the characteristic mid–frequency (MF) shoulder. The MF segment is represented by an affine function in ,

constructed so as to meet the LF and HF lines exactly at 200 Hz and 500 Hz, respectively.
Boundary conditions.
## Let

and

Thus the MF line interpolates between

Numeric coefficients.
## With

the MF slope and intercept are


This produces a nearly flat MF shoulder, consistent with the SS3 curve, and guarantees continuity at both 200 Hz and 500 Hz.
10. Final Piecewise SS3 Model
Combining the turbulence, low–frequency, mid–frequency, and high–frequency components derived in Sections 6–9 yields the complete SS3 parametric noise model. In  versus  coordinates, the spectrum is represented by the four affine segments:

The parameters , , , and  are summarised in Section 5.3, with their numerical values given in Sections 7–10.
This four–segment representation provides a continuous, physically interpretable approximation to the digitised SS3 Knudsen curve, suitable for synthetic waveform generation and downstream processing in the SKANN pipeline.
11. Verification by Superposition
To verify the accuracy of the parametric SS3 model, the four affine segments defined in Section 11 are superimposed onto the digitised SS3 data. The comparison plot serves two purposes:
it demonstrates that each spectral region (turbulence, LF, MF, HF) provides a faithful local approximation to the corresponding portion of the SS3 curve, and
it confirms that the imposed continuity conditions at , 200 Hz, and 500 Hz produce a smooth, visually seamless reconstruction of the overall spectrum.
A Python script for producing the superposition plot is provided in Section 13. The plotted result should show the SS3 points (scatter) and the four parametric branches (solid lines) with excellent alignment across all frequency ranges.
12. Python Code for Complete Parametric Fitting
The following Python script reproduces the SS3 parametric coefficients, constructs the four affine spectral branches, and generates a verification plot comparing the model with the digitised SS3 data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load digitised SS3 CSV: columns = Index, Freq, NL
df = pd.read_csv("SS3CSV.txt", header=None, names=["Index","Freq","NL"])

# --- Turbulence coefficients (from anchor points)
a_turb = (55 - 108) / (np.log10(40) - np.log10(10))
b_turb = 108 - a_turb * np.log10(10)

# --- LF regression (100--200 Hz)
lf = df[(df["Freq"]>=100) & (df["Freq"]<=200)]
x_lf = np.log10(lf["Freq"].values)
A_lf = np.vstack([x_lf, np.ones_like(x_lf)]).T
a_LF, b_LF = np.linalg.lstsq(A_lf, lf["NL"].values, rcond=None)[0]

# --- Intersection f_t
log_f_t = (b_turb - b_LF) / (a_LF - a_turb)
f_t = 10**log_f_t

# --- MF coefficients
x200 = np.log10(200)
x500 = np.log10(500)
NL_200 = a_LF * x200 + b_LF

# Raw HF regression (500--100000 Hz)
hf = df[(df["Freq"]>=500) & (df["Freq"]<=100000)]
x_hf = np.log10(hf["Freq"].values)
A_hf = np.vstack([x_hf, np.ones_like(x_hf)]).T
a_HF_raw, b_HF_raw = np.linalg.lstsq(A_hf, hf["NL"].values, rcond=None)[0]

# MF right-end value from HF regression
NL_HF_500 = a_HF_raw * x500 + b_HF_raw

# MF slope and intercept
a_MF = (NL_HF_500 - NL_200) / (x500 - x200)
b_MF = NL_200 - a_MF * x200

# Final HF intercept ensuring continuity at 500 Hz
b_HF = NL_HF_500 - a_HF_raw * x500
a_HF = a_HF_raw

# --- Plot
plt.figure()
plt.scatter(np.log10(df["Freq"]), df["NL"], s=10, label="SS3 data")

f = np.logspace(1, 5, 500)

def NL_turb(f): return a_turb*np.log10(f) + b_turb
def NL_LF(f):   return a_LF*np.log10(f) + b_LF
def NL_MF(f):   return a_MF*np.log10(f) + b_MF
def NL_HF(f):   return a_HF*np.log10(f) + b_HF

plt.plot(np.log10(f[f< f_t]), NL_turb(f[f < f_t]), label="Turbulence")
plt.plot(np.log10(f[(f>=f_t)&(f<200)]), NL_LF(f[(f>=f_t)&(f<200)]), label="LF")
plt.plot(np.log10(f[(f>=200)&(f<=500)]), NL_MF(f[(f>=200)&(f<=500)]), label="MF")
plt.plot(np.log10(f[f>=500]), NL_HF(f[f>=500]), label="HF")

plt.xlabel("log10(frequency [Hz])")
plt.ylabel("Noise Level [dB]")
plt.legend()
plt.title("SS3 Parametric Model vs Digitised Data")
plt.show()
This script reproduces all coefficients in Sections 7–10 and produces the verification figure described in Section 12.