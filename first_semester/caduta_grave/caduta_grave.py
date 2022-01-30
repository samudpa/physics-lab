import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.optimize import curve_fit

# load data
position_data = pd.read_csv('data/positions.csv') # position data
measures_data = pd.read_json('data/measures.json') # tape and ball data
t = position_data['t']
h = position_data['h']
h_err = position_data['h_err']

# fit model
def parabola(t, a, v0, h0):
    '''Quadratic fit model'''
    return a * (t**2 / 2) + v0 * t + h0

# find the best-fit parameters
popt, pcov = curve_fit(parabola, t, h, sigma=h_err)
a_hat, v0_hat, h0_hat = popt
a_err, v0_err, h0_err = np.sqrt(np.diag(pcov))
print(f'Fit results: {a_hat:.4f} {a_err:.2g}, {v0_hat:.4f} {v0_err:.2g}, {h0_hat:.4f} {h0_err:.2g}')

# calculate residuals and chi2
res = h - parabola(t, *popt)
chi2 = ((res / h_err)**2).sum()
dof = len(h) - 3
chi2_err = np.sqrt(2 * dof) # std deviation
chi2_sigma_dist = (chi2 - dof) / chi2_err # distance from dof in sigma
print(f'chi2: {chi2:.1f}/{dof}, {chi2_sigma_dist:.2f}')

# initialize plot
plt.style.use(['science', 'grid'])
fig = plt.figure(figsize=(5,4))
gs = gridspec.GridSpec(2,1, height_ratios=[3, 1])
plt.subplots_adjust(hspace=0)
plt.tight_layout()

color = 'black'
alpha = 0.3
capsize = 1.5

# best-fit axis
ax_fit = fig.add_subplot(gs[0])

ax_fit.errorbar(t, h, yerr=h_err, color=color, fmt='.', capsize=capsize, label='Dati', zorder=2)

t_fit = np.linspace(0,0.6,100)
ax_fit.plot(t_fit, parabola(t_fit, *popt), color=color, alpha=alpha, label='Modello', zorder=1)
ax_fit.axhline(0, color='black', ls='--', alpha=alpha, zorder=1) # ground line (y=0)
#ax_fit.vlines(t, h, parabola(t, *popt), color=color, alpha=alpha) # model to data lines

# residuals axis
ax_res = fig.add_subplot(gs[1], sharex=ax_fit)

ax_res.errorbar(t, res, yerr=h_err,
        color=color, fmt='.', capsize=capsize, zorder=2) # residuals
ax_res.axhline(0, color='black', ls='--', alpha=alpha, zorder=1) # model (horizontal line)
ax_res.vlines(t, 0, res, color=color, alpha=alpha) # model to data lines

# limits
ax_fit.set_ylim(-0.15, 1.7)
ax_res.set_xlim(-0.035, 0.6)
ax_res.set_ylim(-0.12,0.12)

# setup grids
ax_fit.grid(which='both', ls='dashed', color='lightgray', zorder=0)
ax_res.grid(which='both', ls='dashed', color='lightgray', zorder=0)

# labels and legend
ax_fit.legend()
ax_fit.set_title('Altezza misurata VS modello quadratico')
ax_fit.set_ylabel('Altezza [m]')
ax_res.set_ylabel('Residui [m]')
ax_res.set_xlabel('Tempo [s]')
plt.setp(ax_fit.get_xticklabels(), visible=False)

plt.savefig('graphs/quadratic_model.pdf')
plt.show()