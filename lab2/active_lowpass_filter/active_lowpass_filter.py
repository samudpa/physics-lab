import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

R_C = 993
betaf = 200
r_b = 2800
C = 0.47e-6

f, f_err, vin, vin_err, vout, vout_err = np.loadtxt(
    "./data/active_lowpass_filter_frequency_response.csv",
    delimiter=",", unpack=True)

gain = vout/vin
gain_err = gain * np.sqrt((vout_err/vout)**2 + (vin_err/vin)**2)

def G(f, beta_over_rb, C):
    """Gain model obtained from the absolute value of the transfer function"""

    omega = 2 * np.pi * f
    coeff = -R_C * beta_over_rb

    return np.absolute(
        coeff * (1 - 1j*omega*C/beta_over_rb)/(1 + 1j*omega*R_C*C)
    )

init = (betaf/r_b, C)
popt, pcov = curve_fit(G, f, gain, sigma=gain_err, p0=init, absolute_sigma=False)
perr = np.sqrt(np.diag(pcov))
print(popt, perr)

# initialize plot
plt.style.use(["science"])
plt.figure(figsize=(5,4), dpi=320)
ax = plt.axes()

# plot model
ff = np.logspace(np.log10(f.min()),np.log10(f.max()), 50)
ax.plot(ff, G(ff, *popt), color="red", label="Best fit")

# plot data
ax.errorbar(f, gain, xerr=f_err, yerr=gain_err, fmt=".", capsize=3, color="blue", label="Dati")

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("$f$ [Hz]")
ax.set_ylabel("Guadagno")

ax.legend(loc="lower left")
ax.grid()

plt.savefig("active_lowpass_filter.png")