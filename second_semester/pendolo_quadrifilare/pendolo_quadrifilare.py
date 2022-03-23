import numpy as np
import calc
import draw_plot
from scipy.optimize import curve_fit

# 1. Load data:

# data contained in data/data.txt:
#   t is the time passed since the beginning of the recording [s]
#   T is the period of each swing [s]
#   t_T is the transition time of the flag (time that it takes for the flag to go past the photocell) [s]
t, T, t_T = np.loadtxt('data/data.txt', delimiter='\t', unpack=True) # [s]

measures = calc.get_measures() # all measures are in [cm]
l, l_err = calc.get_cm_distance() # distance of center of mass from rotation axis [m]
M, M_err = calc.get_total_mass() # total mass of the pendulum [kg]
I, I_err = calc.get_moment_of_inertia() # [kg m2]

print('Measurements:')
print(f'  l [m] = {l:.6f} \pm {l_err:.2g}')
print(f'  M [kg] = {M:.6f} \pm {M_err:.2g}')

# 2. Find the velocities v0 of the pendulum at the lowest point in the trajectory

w = measures['alum'][0]['x'] / 100 # flag width [m]
w_err = measures['alum'][0]['x_err'] / 100 # [m]
d = measures['d'] / 100 # photocell distance from the rotation axis [m]
d_err = measures['d_err'] / 100 # [m]

def v0_model(t, v0_in, tau):
    '''Model for the exponential decay of the velocity of the pendulum'''
    return v0_in * np.exp(- t / tau)

v0 = w * l / (t_T * d) # velocity at the lowest point of the trajectory [m/s]

popt, pcov = curve_fit(v0_model, t, v0)
perr = np.sqrt(np.diag(pcov))
print('\nResults for v0_model:')
print(f'  v0_in [m/s] = {popt[0]:.6f} \pm {perr[0]:.2g}')
print(f'  tau [s] = {popt[1]:.6f} \pm {perr[1]:.2g}')

# draw plots (see draw_plot.py)
draw_plot.plot_v0_time(t, v0, popt, v0_model, 'graphs/v0_vs_time.pdf')
draw_plot.plot_period_time(t, T, 'graphs/period_vs_time.pdf')

# 3. Find the amplitude theta0 in relation to the velocity v0

g = 9.81 # [m/s2]
theta0 = np.arccos(1 - v0**2 / (2 * g * l))

# sort period T by theta0 (ascending)
theta0_indexes = theta0.argsort()
theta0 = theta0[theta0_indexes]
T_theta0 = T[theta0_indexes]

def T_model(theta0, l):
    '''Model for the period of the pendulum as a function of theta0'''
    return 2 * np.pi * np.sqrt(l/g) * (1 + 1/16 * theta0**2 + 11/3072 * theta0**4)

popt, pcov = curve_fit(T_model, theta0, T_theta0)
perr = np.sqrt(np.diag(pcov))
print('\nResults for T_model:')
print(f'  l [m] = {popt[0]:.6f} \pm {perr[0]:.2g}')

draw_plot.plot_period_theta0(theta0, T_theta0, l, popt, T_model, 'graphs/period_vs_theta0.pdf')