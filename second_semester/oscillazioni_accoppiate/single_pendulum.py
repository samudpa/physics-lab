import numpy as np
from scipy.optimize import curve_fit
from utils import load_data, find_period, model
from expected_omega0 import get_expected_omega0
from draw_plot import draw_plot

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

omega_0_exp, omega_0_exp_err = get_expected_omega0()
print('\nExpected angular frequency:')
print(f'  omega_0_exp [rad/s]\t= {omega_0_exp} ± {omega_0_exp_err:.2g}')

# Part 3: find the decay time (tau) of the damped pendulum

popt, pcov = curve_fit(
    model,
    data_dicts[1]['t'],
    data_dicts[1]['pos'],
    sigma=data_dicts[1]['pos_err'],
    p0=(200, 4, -np.pi / 4, 1/25, 400),
)
perr = np.sqrt(np.diag(pcov))
A_hat, omega_hat, phi_hat, lambda_hat, offset_hat = popt
A_err, omega_err, phi_err, lambda_err, offset_err = perr

print('\nBEST FIT parameters of the damped pendulum:')
print(f'  omega [rad/s]\t= {omega_hat} ± {omega_err:.2g}')
print(f'  phi [rad]\t= {phi_hat} ± {phi_err:.2g}')
print(f'  lambda [s]\t= {lambda_hat} ± {lambda_err:.2g}')
print(f'  A [au]\t= {A_hat} ± {A_err:.2g}')
print(f'  offset [au]\t= {offset_hat} ± {offset_err:.2g}')

# calculate chi2

df_dt = A_hat * np.exp(-data_dicts[1]['t'] * lambda_hat) * (
    omega_hat * np.sin(omega_hat * data_dicts[1]['t'] + phi_hat) + 
    lambda_hat * np.cos(omega_hat * data_dicts[1]['t'] + phi_hat)
)
sigma2 = data_dicts[1]['pos_err']**2 + df_dt**2 * data_dicts[1]['t_err']**2

res = data_dicts[1]['pos'] - model(data_dicts[1]['t'], *popt)
chi2 = (res**2 / sigma2).sum()
ni = len(data_dicts[1]['pos']) - 4
print(f'chi2: {chi2:.2f}/{ni}')

# draw plots

draw_plot(
    [data_dicts[0]],
    limits = {
        'xlim': (-1, 19),
        'ylim_data': (270, 685)
    },
    title = 'Oscillatore singolo non smorzato',
    filename = 'graphs/not_damped.pdf',
    show = False)

draw_plot(
    [data_dicts[1]],
    limits = {
        'xlim': (-1.5, 32),
        'ylim_data': (201, 680),
        'ylim_res': (-22,22)
    },
    models = [model],
    popts = [popt],
    title = 'Oscillatore singolo smorzato',
    filename = 'graphs/damped.pdf',
    show = False)