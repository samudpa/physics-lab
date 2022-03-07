import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import load_data, find_roots, find_period

# Part 1: find the angular frequency of a free pendulum and a damped pendulum

# free pendulum
data_0 = load_data('data/single_pendulum.txt')['red']
pos_0 = data_0['pos'] - np.mean(data_0['pos'])
T_0, T_0_err, omega_0, omega_0_err = find_period(data_0['t'], pos_0)

# damped pendulum
data_d = load_data('data/single_pendulum_damped.txt')['red']
pos_d = data_d['pos'] - np.mean(data_d['pos'])
T_d, T_d_err, omega_d, omega_d_err  = find_period(data_d['t'], pos_d)

print('Angular frequency estimate for free and damped pendulum:')
print(f'  T_0 [s]\t= {T_0} ± {T_0_err:.2g}')
print(f'  T_d [s]\t= {T_d} ± {T_d_err:.2g}')
print(f'  omega_0 [rad/s]\t= {omega_0} ± {omega_0_err:.2g}')
print(f'  omega_d [rad/s]\t= {omega_d} ± {omega_d_err:.2g}')

# Part 2: find the decay time (tau) of the damped pendulum

def model(t, A, omega, phi, tau):
    '''Damped pendulum's model'''
    return A * np.cos(omega * t + phi) * np.exp(- t / tau)

pos_d_err = np.full(len(pos_d), 1) # 1 mm is the minimum step
popt, pcov = curve_fit(model, data_d['t'], pos_d, sigma = pos_d_err, p0=(200, 4, -np.pi/4, 25))
perr = np.sqrt(np.diag(pcov))
A_hat, omega_hat, phi_hat, tau_hat = popt
A_err, omega_err, phi_err, tau_err = perr

print('\nBest fit on the damped pendulum:')
print(f'  tau [s]\t= {tau_hat} ± {tau_err:.2g}')
print(f'  A [mm]\t= {A_hat} ± {A_err:.2g}')
print(f'  omega [rad/s]\t= {omega_hat} ± {omega_err:.2g}')
print(f'  phi [rad]\t= {phi_hat} ± {phi_err:.2g}')