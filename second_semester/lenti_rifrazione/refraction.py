import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# load data
data = pd.read_csv('data/refraction.csv')

R = 8.1e-2 # radius [m]
R_err = 0.1e-2 # [m]
sini = data['Rsini'] / (100 * R) # sine of the angle of incidence
sinr = data['Rsinr'] / (100 * R) # sine of the angle of refraction

# calculate errors:
#   These are estimated using the arc length error (err).
#   Since what we need is the error on the *sine* of the angle,
#   we need to scale the arc length error by the cosine of the angle (and by 1/R).
err = 0.15e-2 # arc length error [m] (too high?)
cosi = np.sqrt(1 - sini**2)
cosr = np.sqrt(1 - sinr**2)
sini_err = err/R * cosi
sinr_err = err/R * cosr

# fit model:
#   Here we assume that the refractive index of air is n~1.
#   Therefore, Snell's law can be rewritten as sinr = 1/n * sini.
def line(x, m, q):
    '''Linear fit model'''
    return m*x + q

# calculate fit parameters:
#   Since the error on the x (sini) is NOT negligible,
#   we need to iterate curve_fit multiple times
#   while changing the y error each time
#   using the newly estimated refractive index.

n = 1.5 # initial guess
for k in range(0, 5):

    # calculate the new sigma using the partial derivative of
    #   y = m * x + q
    # with respect to x:
    #   (sigma_y')^2 = (sigma_y)^2 + (df(x)/dx)^2 * (sigma_x)^2
    sigma = np.sqrt(sinr_err**2 + (1/n)**2 * sini_err**2)
    print(f'Iteration n°{k+1}: avg(sigma) = {np.average(sigma)}')

    # calculate best fit parameters
    popt, pcov = curve_fit(line, sini, sinr, sigma=sigma, p0=(1/n, 0))
    perr = np.sqrt(np.diag(pcov))
    n = 1 / popt[0] # refractive index
    n_err = perr[0] / popt[0]**2 # error on the inverse

# residuals and chi2
res = sinr - line(sini, *popt)
chi2 = ((res/sigma)**2).sum()
dof = len(sini) - 2 # degrees of freedom
chi2_sigdiff = (chi2 - dof) / np.sqrt(2 * dof)
print(f'\nchi2 = {chi2:.2f}/{dof} ({chi2_sigdiff:.2f} sigma)')

# difference between best-fit parameters and expected values (in sigma)
n_sigdiff = (n - 1.5)/n_err
q_sigdiff = popt[1]/perr[1]

# print fit results
print('\nFit results:')
print(f'  n = {n:.5f} : {n_err:.2g}\t({n_sigdiff:.2f} sigma)')
print(f'  q = {popt[1]:.5f} : {perr[1]:.2g}\t({q_sigdiff:.2f} sigma)')

# temporary plots
plt.errorbar(sini, sinr, yerr=sigma, fmt='.')
plt.plot(sini, line(sini,*popt))
plt.show()

plt.errorbar(sini, res, yerr=sigma,fmt='.')
plt.axhline(0)
plt.show()