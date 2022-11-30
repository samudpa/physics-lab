import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

V2, V1 = np.loadtxt("data/diode.txt", unpack=True) # [digit]
V1_err = np.full(V1.shape, 1) # 1 digit
V2_err = np.full(V2.shape, 1)

# load circuit data
with open("data/circuit_data.json", "r") as f:
    circuit_data = json.loads(f.read())

R = circuit_data["R"] # [ohm]
R_err = circuit_data["R_err"] # [ohm]
Vmax = circuit_data["Vmax"] # [V]
Vmax_err = circuit_data["Vmax_err"] # [V]

# find conversion factor
conv_factor = Vmax/1023 # [V/digit]
conv_factor_err = Vmax_err/1023 # [V/digit]
print(f"conv_factor = {1e3 * conv_factor:.5f} {1e3 * conv_factor_err:.2g} mV/digit")

# convert V1, V2 to Volts
V1_err = np.sqrt((V1_err*conv_factor)**2 + (conv_factor_err*V1)**2) # using error of 1 digit
V2_err = np.sqrt((V2_err*conv_factor)**2 + (conv_factor_err*V2)**2)
V1 *= conv_factor # [V]
V2 *= conv_factor # [V]

# find current flowing through diode
I = (V1 - V2)/R * 1e3 # [mA]
I_err = np.sqrt(
    (V1_err)**2 * (1/R)**2 + \
    (V2_err)**2 * (1/R)**2 + \
    (R_err)**2 * ((V1 - V2)/(R**2))**2
) * 1e3

def char_curve(V, A, B):
    """I-V model for diode"""
    return A * (np.exp(V/B) - 1)

# find best fit
init = (1e-5, 50e-3) # initial parameters ([mA], [V])
popt = init
for i in range(0, 5): # iterate curve_fit
    err = np.sqrt(
        (I_err**2) +
        (V2_err**2) * ((popt[0]/popt[1]) * np.exp(V2/popt[1]))**2
    )
    popt, pcov = curve_fit(char_curve, V2, I, p0=popt, sigma=err, absolute_sigma=False)

perr = np.sqrt(np.diag(pcov))
k2 = (((I - char_curve(V2, *popt))/err)**2).sum()
ndof = len(I) - 2
corr = pcov[1,0] / (perr[0] * perr[1])
print(f"A = {popt[0] * 1e6:.5f} {perr[0] * 1e6:.2g} nA")
print(f"B = {popt[1] * 1e3:.5f} {perr[1] * 1e3:.2g} mV")
print(f"chi2/ndof = {k2:.1f}/{ndof} = {k2/ndof:.1f}")
print(f"corr = {corr:.3f}")

# plot
plt.style.use(["science"])
plt.figure(figsize=(4,3.5))

xx = np.linspace(V2.min(), V2.max(), 100)
plt.errorbar(V2, I, xerr=V2_err, yerr=I_err, fmt=",", capsize=0.5, lw=0.5, capthick=0.5, alpha=0.8, color="black", label="Dati")
plt.plot(xx, char_curve(xx, *popt), lw=0.8, alpha=0.8, color="red", label="Best-fit")

plt.title("Curva caratteristica del diodo")
plt.xlabel("$\Delta V$ [V]")
plt.ylabel("$I$ [mA]")
plt.legend(loc="upper left")

plt.grid()
plt.savefig("diode_IV_curve.png", dpi=320)
plt.show()