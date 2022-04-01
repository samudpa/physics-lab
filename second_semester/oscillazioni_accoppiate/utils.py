import numpy as np
from scipy.optimize import curve_fit

t_err_factor = 1/np.sqrt(12)
pos_err_factor = 1/np.sqrt(12)

def load_raw_data(filename, start=0, stop=600):
    '''Load time and position data from .txt'''

    # load and format data inside a dict
    data = np.loadtxt(filename, unpack=True)
    data_dict = {
        'red': {'t': data[0], 'pos': data[1]},
        'blue': {'t': data[2], 'pos': data[3]}
    }

    # discards the data after the cutoff
    for color in data_dict.keys():
        for category in data_dict[color].keys():
            data_dict[color][category] = data_dict[color][category][start:stop]

    return data_dict

def load_data(filename, start=0, stop=600, fix_offset=False):
    '''Load time, position data from .txt and calculate the errors'''

    # load data of both pendulums
    data_raw = load_raw_data(filename, start, stop)

    # iterate over each pendulum
    result = {}
    for color in data_raw.keys():

        data = data_raw[color]
        n = len(data['t'])
        if fix_offset:
            data['pos'] -= np.mean(data['pos'])

        # calculate error on time
        # the error on time is the time between each sample divided by 2
        t_err = t_err_factor * np.diff(data['t'])[0]
        data['t_err'] = np.full(n, t_err)

        # calculate error on position
        possible_values = np.sort(np.unique(data['pos']))
        pos_err = pos_err_factor * np.min(np.abs(np.diff(possible_values)))
        data['pos_err'] = np.full(n, pos_err)

        result[color] = data

    return result

def find_roots(x, y, offset=0):
    '''Find the roots of a set of points'''
    y_ = y - offset
    sign = np.sign(y_)
    np.place(sign, sign==0, [1])
    sign_diff = np.diff(sign)
    indexes = np.concatenate([abs(sign_diff) == 2, [False]])
    return x[indexes], indexes

def find_period(t, pos, t_err, **kwargs):
    '''Find the period and angular frequency of a wave'''
    roots, _ = find_roots(t, pos, np.mean(pos))
    t_diff = np.diff(roots)

    # period
    T = np.mean(t_diff) * 2
    T_err = t_err[0] / np.sqrt(len(t_diff))

    # angular frequency
    omega = 2 * np.pi / T
    omega_err = omega * T_err / T

    return T, T_err, omega, omega_err

def model(t, A, omega, phi, lambda_, offset):
    '''Damped pendulum model'''
    return A * np.cos(omega * t + phi) * np.exp(-t * lambda_) + offset

def abs_model(t, A, omega, phi, lambda_, offset):
    '''Damped pendulum model'''
    return np.abs(model(t, A, omega, phi, lambda_, offset))

def fit_data(data_dict, p0=None, model=model, print_results=True):
    '''Fit pendulum position data on a model, and estimate the decay time (tau)'''

    t = data_dict['t']
    pos = data_dict['pos']
    pos_err = data_dict['pos_err']

    popt, pcov = curve_fit(model, t, pos, sigma = pos_err, p0 = p0)
    perr = np.sqrt(np.diag(pcov))
    A_hat, omega_hat, phi_hat, lambda_hat, offset_hat = popt
    A_err, omega_err, phi_err, lambda_err, offset_err = perr

    if print_results:
        print('BEST FIT parameters:')
        print(f'  omega [rad/s]\t= {omega_hat} ± {omega_err:.2g}')
        print(f'  phi [rad]\t= {phi_hat} ± {phi_err:.2g}')
        print(f'  lambda [s]\t= {lambda_hat} ± {lambda_err:.2g}')
        print(f'  A [au]\t= {A_hat} ± {A_err:.2g}')
        print(f'  offset [au]\t= {offset_hat} ± {offset_err:.2g}')

        tau = 1/popt[3]
        tau_err =  perr[3] * tau**2

        print(f'Decay time tau [s] = {tau} ± {tau_err:.2g}')

    return popt, perr