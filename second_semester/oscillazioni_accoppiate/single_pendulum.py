import numpy as np
from utils import load_data, find_period, fit_data, pendulum_model
from expected_omega0 import get_expected_omega0
from draw_plot import draw_plot

# Part 1: compare the angular frequency of a free pendulum and a damped pendulum

data_dicts = [
    # not damped
    load_data('data/single_pendulum.txt')['red'],

    # damped pendulum
    load_data('data/single_pendulum_damped.txt', start=150, stop=550)['red']
]

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

print('\n(not damped)')
popt_not_damped, _ = fit_data(data_dicts[0], p0 = (200, omega_0, 0, 1/50, 400))

# Part 2: find the decay time (tau) of the damped pendulum

print('\n(damped)')
popt_damped, perr_damped = fit_data(data_dicts[1], p0 = (200, omega_d, -np.pi / 4, 1/25, 400))

# Draw plots

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
    [data_dicts[0]],
    limits = {
        'xlim': (-1, 19),
        'ylim_data': (250, 700),
        'ylim_res': (-3,3)
    },
    models = [pendulum_model],
    popts = [popt_not_damped],
    title = 'Oscillatore singolo non smorzato',
    filename = 'graphs/not_damped_fit.pdf',
    show = False)

draw_plot(
    [data_dicts[1]],
    limits = {
        'xlim': (6.5, 29),
        'ylim_data': (265, 600),
        'ylim_res': (-4,4)
    },
    models = [pendulum_model],
    popts = [popt_damped],
    title = 'Oscillatore singolo smorzato',
    filename = 'graphs/damped.pdf',
    show = False)