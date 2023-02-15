import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# load frequency response data and circuit parameters
f, f_err, vin, vin_err, vout, vout_err = np.loadtxt(
    "./data/frequency_response.csv", delimiter=",", unpack=True)
RE, CE, RC, RB, rb = np.loadtxt(
    "./data/circuit_parameters.csv", delimiter=",", unpack=True)

# find gain
G = vout/vin
G_err = G * np.sqrt((vout_err/vout)**2 + (vin_err/vin)**2)

def transfer_fun(f, betaf, RE, CE, RBb, RC):
    """Transfer function (complex valued)"""
    w = 2*np.pi*f
    return - RC * betaf / (RBb + (betaf-1)*RE/(1 + 1j*w*RE*CE))

def gain_fun(f, betaf, RE, CE):
    """Gain function used as model"""
    return np.abs(transfer_fun(f, betaf, RE, CE, RB+rb, RC))

# find best fit parameters
init = (200, RE, CE) # initial guess
popt, pcov = curve_fit(gain_fun, xdata=f, ydata=G,
    p0=init, sigma=G_err, absolute_sigma=False)
perr = np.sqrt(np.diag(pcov))

print(f"betaf = {popt[0]} ({perr[0]:.2g})")
print(f"RE = {popt[1]*1e-3} ({perr[1]*1e-3:.2g}) [kΩ]")
print(f"CE = {popt[2]*1e6} ({perr[2]*1e6:.2g}) [µF]")

# plot results
plt.style.use(["science"])
plt.figure(figsize=(4,3), dpi=320)

ff = np.logspace(0, 5, 100)
plt.plot(ff, gain_fun(ff, *popt),
    color="red", label="$G_V(f)$", zorder=2)
plt.errorbar(f, G, xerr=f_err, yerr=G_err,
    fmt=".", capsize=1, ms=2, color="blue", label="Dati", zorder=3)

plt.grid(zorder=0)
plt.legend()
plt.xlabel("$f$ [Hz]")
plt.ylabel("Guadagno (v$_\\text{out}/$v$_\\text{in}$)")

plt.xlim(8, 5e4)
plt.xscale("log")
plt.yscale("log")

plt.savefig("./frequency_response.png")