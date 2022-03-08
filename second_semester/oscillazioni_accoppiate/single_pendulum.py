import numpy as np
from scipy.optimize import curve_fit
from utils import load_data, find_period, model

# Part 1: compare the angular frequency of a free pendulum and a damped pendulum

data_dicts = []
for filename in ['data/single_pendulum.txt', 'data/single_pendulum_damped.txt']:
    data_dict = load_data(filename)
    data_dicts.append(data_dict['red'])

T_0, T_0_err, omega_0, omega_0_err = find_period(**data_dicts[0])
T_d, T_d_err, omega_d, omega_d_err = find_period(**data_dicts[1])

print('Angular frequency estimate for free and damped pendulum:')
print(f'  T_0 [s]\t\t= {T_0} ± {T_0_err:.2g}')
print(f'  T_d [s]\t\t= {T_d} ± {T_d_err:.2g}')
print(f'  omega_0 [rad/s]\t= {omega_0} ± {omega_0_err:.2g}')
print(f'  omega_d [rad/s]\t= {omega_d} ± {omega_d_err:.2g}')

# Part 2: find the decay time (tau) of the damped pendulum

popt, pcov = curve_fit(
    model,
    data_dicts[1]['t'],
    data_dicts[1]['pos'],
    sigma=data_dicts[1]['pos_err'],
    p0=(200, 4, -np.pi / 4, 25),
)
perr = np.sqrt(np.diag(pcov))
A_hat, omega_hat, phi_hat, tau_hat = popt
A_err, omega_err, phi_err, tau_err = perr

print('\nBEST FIT parameters of the damped pendulum:')
print(f'  tau [s]\t= {tau_hat} ± {tau_err:.2g}')
print(f'  A [au]\t= {A_hat} ± {A_err:.2g}')
print(f'  omega [rad/s]\t= {omega_hat} ± {omega_err:.2g}')
print(f'  phi [rad]\t= {phi_hat} ± {phi_err:.2g}')