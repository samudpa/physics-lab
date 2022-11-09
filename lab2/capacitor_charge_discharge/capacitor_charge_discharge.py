from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

filename = "wavedata/charge_discharge.txt"
data = np.loadtxt(filename)

# split charge and discharge data
# first column represents time values [us]
# second column represents electric potential relative
# to the ground, as reported by Arduino [digit] (10bit)
data_charge = data[336:676]
data_discharge = data[676:1014]

# extract t and V columns, making sure to start at t0 = 0
# [us]
t_charge = data_charge[:, 0] - data_charge[0, 0]
t_discharge = data_discharge[:, 0] - data_discharge[0, 0]
# [digit]
V_charge = data_charge[:, 1]
V_discharge = data_discharge[:, 1]

# create error arrays
# +- 1 us
t_charge_err = np.full(t_charge.shape, 1)
t_discharge_err = np.full(t_discharge.shape, 1)
# +- 1 digit
V_charge_err = np.full(V_charge.shape, 1)
V_discharge_err = np.full(V_discharge.shape, 1)

# charge/discharge models
def charge(t, V_0, tau, offset):
    return V_0 * (1 - np.exp(-t / tau)) + offset


def discharge(t, V_0, tau, offset):
    return V_0 * np.exp(-t / tau) + offset


# initial guess
p0 = (907, 1600, 25)  # ([digit], [us], [digit])

# capacitor charge best fit
popt, pcov = curve_fit(charge, t_charge, V_charge, absolute_sigma=False, p0=p0)
perr = np.sqrt(np.diag(pcov))
# k2 and degrees of freedom
k2 = (((V_charge - charge(t_charge, *popt)) / V_charge_err) ** 2).sum()
ndof = len(V_charge) - 3
# print results
print(popt, perr)
print(k2, ndof)

# plot charge data and best fit
tt = np.linspace(0, t_charge[-1], 1000)
plt.figure(figsize=(3, 3), dpi=400)
plt.style.use(["science"])
plt.errorbar(
    t_charge,
    V_charge,
    xerr=t_charge_err,
    yerr=V_charge_err,
    fmt=".",
    markersize=1.5,
    capsize=0,
    color="black",
    zorder=1,
)
plt.plot(tt, charge(tt, *popt), linewidth=1, color="red", alpha=0.7, zorder=2)
plt.title("Carica condensatore")
plt.xlabel("$t$ [$\mu s$]")
plt.ylabel("$V$ [digit]")
plt.grid(zorder=0)
plt.savefig("charge.png")
plt.close()

# capacitor discharge best fit
popt, pcov = curve_fit(discharge, t_discharge, V_discharge, absolute_sigma=False, p0=p0)
perr = np.sqrt(np.diag(pcov))
# k2 and degrees of freedom
k2 = (((V_discharge - discharge(t_discharge, *popt)) / V_discharge_err) ** 2).sum()
ndof = len(V_discharge) - 3
# print results
print(popt, perr)
print(k2, ndof)

# plot charge data and best fit
tt = np.linspace(0, t_discharge[-1], 1000)
plt.figure(figsize=(3, 3), dpi=400)
plt.style.use(["science"])
plt.errorbar(
    t_discharge,
    V_discharge,
    xerr=t_discharge_err,
    yerr=V_discharge_err,
    fmt=".",
    markersize=1.5,
    capsize=0,
    color="black",
    zorder=1,
)
plt.plot(tt, discharge(tt, *popt), linewidth=1, color="red", alpha=0.7, zorder=2)
plt.title("Scarica condensatore")
plt.xlabel("$t$ [$\mu s$]")
plt.ylabel("$V$ [digit]")
plt.grid(zorder=0)
plt.savefig("discharge.png")
plt.close()
