import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import fmt_measure as fmt

INDUCTANCE = 0.5 # H; inductance of the RLC circuit
RESISTANCE = 40 # ohm; estimated resistance

def V_C(t, A, tau, omega, phi, offset):
    """Model for the potential difference at the end of the capacitor"""
    return A*np.exp(-t/tau)*np.cos(omega*t+phi)+offset

measures = np.genfromtxt("./measures.csv", delimiter=",", names=True,
    dtype=None, encoding="utf-8")

for folder, basename, C in measures:

    filename = folder+basename
    print("\n=== {} ===".format(filename))

    # load data
    t, t_err, V, V_err = np.loadtxt(filename, unpack=True)
    t *= 1e-6 # convert us to s
    t_err *= 1e-6 # see above
    C_err = C*0.2 # 20%

    # initial parameters
    A = (V.max() - V.min())/2 # digit
    tau = 2*INDUCTANCE/RESISTANCE # s
    omega = 1/np.sqrt(INDUCTANCE*C) # rad/s
    phi = np.pi/2 # rad
    offset = V.mean() # digit
    init = (A, tau, omega, phi, offset)

    # estimate best fit parameters
    popt, pcov = curve_fit(V_C, t, V, p0=init, sigma=V_err, absolute_sigma=False)
    perr = np.sqrt(np.diag(pcov))

    # indirect measurements
    T = 2*np.pi/popt[2] # oscillation period [s]
    T_err = T * perr[2]/popt[2]
    L = 1/(C*popt[2]**2) # inductance [H]
    L_err = L * (C_err/C + 2*perr[2]/popt[2])
    r = 2*L/popt[1] # resistance [ohm]
    r_err = r * (perr[1]/popt[1] + L_err/L)

    # chi2, quality factor
    chi2 = (((V - V_C(t, *popt))/V_err)**2).sum()
    ndof = len(t) - 5
    Qf = popt[2]*popt[1]/2
    Qf_err = Qf * (perr[2]/popt[2] + perr[1]/popt[1])

    # print results
    print("\nBest fit parameters:")
    print("A = {} digit".format(fmt(popt[0], perr[0])))
    print("τ = {} ms".format(fmt(popt[1]*1e3, perr[1]*1e3)))
    print("ω = {} rad/s".format(fmt(popt[2], perr[2])))
    print("φ = {} rad".format(fmt(popt[3], perr[3])))
    print("B = {} digit".format(fmt(popt[4], perr[4])))

    print("\nT = {} ms".format(fmt(T*1e3, T_err*1e3)))
    print("L = {} H".format(fmt(L, L_err)))
    print("r = {} Ω".format(fmt(r, r_err)))

    print("\nχ²/ndof = {:.3f}".format(chi2/ndof))
    print("Qf = {}".format(fmt(Qf, Qf_err)))

    # plot
    plt.figure(dpi=320)

    plt.errorbar(t, V, xerr=t_err, yerr=V_err, fmt=".",
        ms=3, capsize=2, elinewidth=0.75, markeredgewidth=0.75, color="blue", label="Dati", zorder=2)
    plt.plot(t, V_C(t, *popt), linewidth=1.25, color="red", label="Fit", zorder=3)

    plt.xlim(0, t.max())

    plt.title(filename)
    plt.xlabel("$t$ [s]")
    plt.ylabel("$V_C$ [digit]")

    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig("./results/"+basename+".png")
    plt.close()