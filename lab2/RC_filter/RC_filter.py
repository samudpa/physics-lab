import numpy as np
import json
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# load data about the RC high-pass filter
with open("data/filter_data.json", "r") as f:
    filter_data = json.loads(f.read())
R = filter_data["R"] # [ohm]
R_err = filter_data["R_err"] # [ohm]
C = filter_data["C"] # [F]
C_err = filter_data["C_err"] # [F]

# find expected cutoff frequency from R, C values
fcut_exp = 1 / (2*np.pi*R*C)
fcut_exp_err = fcut_exp * np.sqrt((R_err/R)**2 + (C_err/C)**2)
print("Cutoff frequency [Hz]", fcut_exp, fcut_exp_err)

# load frequency measurements
f, f_err, Vin, Vin_err, Vout, Vout_err, deltaT, deltaT_err = np.loadtxt("data/frequency_data.csv", unpack=True, skiprows = 1)

# output gain
G = Vout/Vin
G_err = G * np.sqrt((Vout_err/Vout)**2 + (Vin_err/Vin)**2)
GdB = 20 * np.log10(G) # output gain in decibels
GdB_err = 20/np.log(10) * G_err/G

def gain(f, fcut, A):
    """Gain model for a RC high-pass filter"""
    return A / np.sqrt(1 + (fcut/f)**2)

# find best-fit parameters for the RC high-pass filter model
init = (fcut_exp, 1) # initial parameters
popt, pcov = curve_fit(gain, f, G, sigma=G_err, p0=init, absolute_sigma=False)
perr = np.sqrt(np.diag(pcov))
k2 = (((G - gain(f, *popt))/G_err)**2).sum()
ndof = len(G) - 2
corr = pcov[1,0]/(perr[0]*perr[1]) # correlation between parameters
print("Cutoff frequency (best fit) [Hz]", popt[0], perr[0])
print("A (best fit)", popt[1], perr[1])
print("Chi2/ndof", k2, ndof, k2/ndof)
print("Correlation", corr)

# plot
plt.style.use(["science"])
fig, ax = plt.subplots()
fig.set_size_inches(3.5, 2.5)

ax.errorbar(f, G, xerr=f_err, yerr=G_err, fmt='.', markersize=3, color="#004488", label="Dati", zorder=2)

ax.axhline(1, color="black", linewidth=0.5) # Gain = 1

xx = np.logspace(np.log10(20), np.log10(15000), 100)
ax.plot(xx, gain(xx, *popt), color="red", label="Best-fit", alpha=0.7)

ax.set_yscale("log")
ax.grid(which="minor", color="#dddddd", linestyle="--")
ax.grid(which="major", color="#aaaaaa")
ax.legend(loc="lower right")
ax.set_title("Filtro RC passa-alto")
ax.set_xlabel("$f$ [Hz]")
ax.set_ylabel("Guadagno")

ax.set_xlim(25,15000)
ax.set_ylim(1e-2,1.2)

ax_dB = ax.twinx()
ax_dB.set_xscale("log")
ax_dB.set_ylim(20*np.log10(1e-2), 20*np.log10(1.2))
ax_dB.set_ylabel("Guadagno [dB]")

plt.savefig("Gain.png", dpi=400)