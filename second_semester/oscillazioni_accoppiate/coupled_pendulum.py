import numpy as np
from utils import load_data, find_period, beats_model, fit_data, fmt_measure
from draw_plot import draw_plot

# Part 1: compare the angular frequency of two pendula connected by a spring,
#         when in phase or antiphase

# in phase
data_dict_phase = load_data('data/coupled_pendulum_phase.txt', fix_offset=True)
T_red, T_red_err, omega_red, omega_red_err = find_period(**data_dict_phase['red'])
T_blue, T_blue_err, omega_blue, omega_blue_err = find_period(**data_dict_phase['blue'])

popt_red, perr_red, _, _ = fit_data(
    data_dict_phase['red'],
    p0 = (200, omega_red, -2, 0.02, 0),
    print_results=False)
popt_blue, perr_blue, _, _ = fit_data(
    data_dict_phase['blue'],
    p0 = (200, omega_blue, -2, 0.02, 0),
    print_results=False)

print('\nAngular frequency estimates for red and blue pendulum (phase) [rad/s]:')
print(f'omega_red  (sgn): {fmt_measure(omega_red, omega_red_err)}')
print(f'omega_red  (fit): {fmt_measure(popt_red[1], perr_red[1])}')
print(f'omega_blue (sgn): {fmt_measure(omega_blue, omega_blue_err)}')
print(f'omega_blue (fit): {fmt_measure(popt_blue[1], perr_blue[1])}')

omega_f = np.mean([omega_red, omega_blue, popt_red[1], popt_blue[1]])
omega_f_err = np.sqrt(omega_red_err**2 + omega_blue_err**2 + perr_blue[1]**2 + perr_red[1]**2) / 4

# antiphase
data_dict_antiphase = load_data('data/coupled_pendulum_antiphase.txt', fix_offset=True)
T_red, T_red_err, omega_red, omega_red_err = find_period(**data_dict_antiphase['red'])
T_blue, T_blue_err, omega_blue, omega_blue_err = find_period(**data_dict_antiphase['blue'])

popt_red, perr_red, _, _ = fit_data(
    data_dict_antiphase['red'],
    p0 = (200, omega_red, -2, 0.02, 0),
    print_results=False)
popt_blue, perr_blue, _, _ = fit_data(
    data_dict_antiphase['blue'],
    p0 = (200, omega_blue, -2, 0.02, 0),
    print_results=False)

print('\nAngular frequency estimates for red and blue pendulum (antiphase) [rad/s]:')
print(f'omega_red  (sgn): {fmt_measure(omega_red, omega_red_err)}')
print(f'omega_red  (fit): {fmt_measure(popt_red[1], perr_red[1])}')
print(f'omega_blue (sgn): {fmt_measure(omega_blue, omega_blue_err)}')
print(f'omega_blue (fit): {fmt_measure(popt_blue[1], perr_blue[1])}')

omega_c = np.mean([omega_red, omega_blue, popt_red[1], popt_blue[1]])
omega_c_err = np.sqrt(omega_red_err**2 + omega_blue_err**2 + perr_blue[1]**2 + perr_red[1]**2) / 4

percent_diff = (omega_c/omega_f - 1) * 100

# print results
print('\nAngular frequency of in phase / antiphase oscillations (avg):')
print(f'  omega_f [rad/s]\t= {fmt_measure(omega_f, omega_f_err)}')
print(f'  omega_c [rad/s]\t= {fmt_measure(omega_c, omega_c_err)}')
print(f'--> Difference between omega_f and omega_c is around {percent_diff:.2g}%')

# Part 2: beat phenomenom. Compare carrier/modulating frequency to omega_f and omega_c

# expected values
omega_p_exp = (omega_c + omega_f) / 2
omega_b_exp = (omega_c - omega_f) / 2
omega_p_exp_err = np.sqrt(omega_f_err**2 + omega_c_err**2) / 2
omega_b_exp_err = omega_p_exp_err

# load data
data_dict_beats = load_data('data/coupled_pendulum_beats.txt', start=40, stop=810, fix_offset=True)

# carrier wave period, angular frequency
T_p, T_p_err, omega_p, omega_p_err = find_period(**data_dict_beats['blue'])

# A, lambda, omega_p, omega_b, phi_p, phi_b, offset
p0 = (300, 0.02, omega_p_exp, omega_b_exp, -np.pi, -np.pi/2, 0)
r = 0.1

# fit data and print results
print('\n(blue)')
popt, perr, chi2, ni = fit_data(
    data_dict_beats['blue'],
    model = beats_model,
    p0 = p0,
    bounds = [
        (270, 0.005, omega_p_exp-r, omega_b_exp-r, -np.pi-0.5, -np.pi, -10),
        (320, 0.04,  omega_p_exp+r, omega_b_exp+r,  -np.pi+0.5,  0,  10)],
    print_results=True)

omega_p = popt[2]
omega_b = popt[3]
omega_p_err = perr[2]
omega_b_err = perr[3]

print('\nCarrier/modulating frequency VS expected:')
print(f'  omega_p [rad/s]\t= {fmt_measure(omega_p, omega_p_err)}')
print(f'  omega_p_exp [rad/s]\t= {fmt_measure(omega_p_exp, omega_p_exp_err)}')
print(f'  omega_b [rad/s]\t= {fmt_measure(omega_b, omega_b_err)}')
print(f'  omega_b_exp [rad/s]\t= {fmt_measure(omega_b_exp, omega_b_exp_err)}')

# Plots

# phase
draw_plot(
    [data_dict_phase['red'], data_dict_phase['blue']],
    title = 'Oscillatori in fase',
    colors = ['firebrick','royalblue'],
    limits = {
        'xlim': (5,15),
        'ylim_data': None,
    },
    filename='graphs/phase.pdf',
    show=False
)

# antiphase
draw_plot(
    [data_dict_antiphase['red'], data_dict_antiphase['blue']],
    title = 'Oscillatori in controfase',
    limits = {
        'xlim': (5,15),
        'ylim_data': None,
    },
    colors = ['firebrick','royalblue'],
    filename='graphs/antiphase.pdf',
    show=False
)

# beats
draw_plot(
    [data_dict_beats['blue']],
    title = 'Fenomeno dei battimenti',
    popts = [popt],
    models = [beats_model],
    chi2 = chi2,
    ni = ni,
    limits = {
        'xlim': (1, 42),
        'ylim_data': (-220, 220),
        'ylim_res': (-7,7),
    },
    figsize = (8,3.5),
    filename='graphs/beats.pdf',
    show=True
)