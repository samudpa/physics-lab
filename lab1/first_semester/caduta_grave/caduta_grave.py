import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from draw_plot import plot_time_height

# load raw data
height_data = pd.read_csv('data/height_data.csv') # height data
measures_data = pd.read_json('data/measures.json', typ='series') # tape and ball data

# convert tape measure to actual height and calculate error
err_factor = 10 # Maybe too high?
h = measures_data['tape']['height'] / 100 - height_data['h'] / 100 # [cm] to [m]
h_err = (height_data['h_delta'] / 100) / err_factor

# convert frame numbers to seconds
t = (height_data['frame'] - height_data['frame'][0]) / 30 # [s]

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

# plot quadratic model
plot_time_height(
    title = 'Altezza misurata VS modello quadratico',
    filename = 'graphs/quadratic_model.pdf',
    data = (t, h, h_err, res, popt),
    model = parabola,
    xlim = (-0.035, 0.6),
    ylim_res = (-0.017, 0.017),
    ylim_fit = (-0.15, 1.7)
)