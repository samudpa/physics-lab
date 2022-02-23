import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# load data
data = pd.read_csv('data/refraction.csv')

R = 8.1e-2 # radius [m]
R_err = 0.1e-2 # [m]
sini = data['Rsini'] / (100 * R) # angle of incidence
sinr = data['Rsinr'] / (100 * R) # angle of refraction

# calculate errors
err = 0.15e-2 # [m]
cosi = np.sqrt(1 - sini**2)
cosr = np.sqrt(1 - sinr**2)
sini_err = err/R * cosi
sinr_err = err/R * cosr

# calculate best-fit parameters
def line(x, m, q):
    '''Linear fit model'''
    return m*x + q

popt, pcov = curve_fit(line, sini, sinr, sigma=sinr_err)
perr = np.sqrt(np.diag(pcov))
n = 1 / popt[0] # refractive index
n_err = perr[0] / popt[0]**2

# print fit results
print('Fit results for sinr = 1/n * sini')
print(f'\tn \t= {n:.5f}\t: {n_err:.2g}')
print(f'\tq \t= {popt[1]:.5f}\t: {perr[1]:.2g}')

# temporary plot
plt.errorbar(sini, sinr, xerr=sini_err, yerr=sinr_err, fmt='.')
plt.show()