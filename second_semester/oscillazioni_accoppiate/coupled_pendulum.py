import numpy as np
from scipy.optimize import curve_fit
from utils import find_roots, load_data, find_period, abs_model, model
from draw_plot import draw_plot

# Part 1: compare the angular frequency of two pendula connected by a spring,
#         when in phase or antiphase

# in phase
data_dict_phase = load_data('data/coupled_pendulum_phase.txt', fix_offset=True)
T_red, T_red_err, omega_red, omega_red_err = find_period(**data_dict_phase['red'])
T_blue, T_blue_err, omega_blue, omega_blue_err = find_period(**data_dict_phase['blue'])

omega_f = np.mean([omega_red, omega_blue])
omega_f_err = np.sqrt(omega_red_err**2 + omega_blue_err**2) / 2

# antiphase
data_dict_antiphase = load_data('data/coupled_pendulum_antiphase.txt', fix_offset=True)
T_red, T_red_err, omega_red, omega_red_err = find_period(**data_dict_antiphase['red'])
T_blue, T_blue_err, omega_blue, omega_blue_err = find_period(**data_dict_antiphase['blue'])

omega_c = np.mean([omega_red, omega_blue])
omega_c_err = np.sqrt(omega_red_err**2 + omega_blue_err**2) / 2

# print results
print('Angular frequency of in phase / antiphase oscillations:')
print(f'  omega_f [rad/s]\t= {omega_f} ± {omega_f_err:.2g}')
print(f'  omega_c [rad/s]\t= {omega_c} ± {omega_c_err:.2g}')

# Part 2: beat phenomena. Compare carrier/modulating frequency to omega_f and omega_c

# expected values
omega_p_exp = (omega_c + omega_f) / 2
omega_b_exp = (omega_c - omega_f) / 2
omega_p_exp_err = np.sqrt(omega_f_err**2 + omega_c_err**2) / 2
omega_b_exp_err = omega_p_exp_err

# load data
data_dict_beats = load_data('data/coupled_pendulum_beats.txt', stop=1200, fix_offset=True)

# carrier wave period, angular frequency
T_p, T_p_err, omega_p, omega_p_err = find_period(**data_dict_beats['red'])

# modulating oscillator period, angular frequency:
#   to find the modulating period we first find all the maxima and minima of the wave,
#   then compute the best fit parameters on the damped pendulum model
pos_diff = np.gradient(data_dict_beats['red']['pos'])
t_mod, indexes = find_roots(data_dict_beats['red']['t'], pos_diff)
pos_mod = np.abs(data_dict_beats['red']['pos'][indexes])
pos_mod_err = data_dict_beats['red']['pos_err'][indexes]

popt, pcov = curve_fit(
    abs_model,
    t_mod,
    pos_mod,
    sigma=pos_mod_err,
    p0=(270, omega_b_exp, np.pi/4, 1/40, -100)
)
perr = np.sqrt(np.diag(pcov))
A_b, omega_b, phi_b, lambda_b, offset_b = popt
A_b_err, omega_b_err, phi_b_err, lambda_b_err, offset_b_err = perr

print('\nBEST FIT parameters of the modulating oscillator:')
print(f'  lambda [1/s]\t= {lambda_b} ± {lambda_b_err:.2g}')
print(f'  A [au]\t= {A_b} ± {A_b_err:.2g}')
print(f'  omega [rad/s]\t= {omega_b} ± {omega_b_err:.2g}')
print(f'  phi [rad]\t= {phi_b} ± {phi_b_err:.2g}')
print(f'  offset [au]\t= {offset_b} ± {offset_b_err:.2g}')

print('\nCarrier/modulating frequency VS expected:')
print(f'  omega_p [rad/s]\t= {omega_p} ± {omega_p_err:.2g}')
print(f'  omega_p_exp [rad/s]\t= {omega_p_exp} ± {omega_p_exp_err:.2g}')
print(f'  omega_b [rad/s]\t= {omega_b} ± {omega_b_err:.2g}')
print(f'  omega_b_exp [rad/s]\t= {omega_b_exp} ± {omega_b_exp_err:.2g}')

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