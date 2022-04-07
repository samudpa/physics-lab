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

def pendulum_model(t, A, omega, phi, lambda_, offset):
    '''Damped pendulum model'''
    return A * np.cos(omega * t + phi) * np.exp(-t * lambda_) + offset

def pendulum_model_dfdt(t, A, omega, phi, lambda_, offset):
    return A * np.exp(-lambda_ * t) * (- omega * np.sin(omega * t + phi) - lambda_ * np.cos(omega * t + phi))

def beats_model(t, A, lambda_, omega_p, omega_b, phi_p, phi_b, offset):
    '''Model describing the beats phenomenon'''
    return offset + A * np.exp(- lambda_ * t) * np.cos(omega_p * t + phi_p) * np.cos(omega_b * t + phi_b)

def beats_model_dfdt(t, A, lambda_, omega_p, omega_b, phi_p, phi_b, offset):
    return A * np.exp(- lambda_ * t) * (
        (- lambda_) * np.cos(omega_p * t + phi_p) * np.cos(omega_b * t + phi_b) + \
        (- omega_p) * np.sin(omega_p * t + phi_p) * np.cos(omega_b * t + phi_b) + \
        (- omega_b) * np.cos(omega_p * t + phi_p) * np.sin(omega_b * t + phi_b)
    )


def fit_data(data_dict, p0=None, model=pendulum_model, print_results=True, bounds=(-np.inf, np.inf), N=10):
    '''Fit pendulum position data on a model, and estimate the decay time (tau)'''

    # choose df/dt model to use in sigma calculation
    if model == pendulum_model:
        dfdt = pendulum_model_dfdt
        ddf = 5
    elif model == beats_model:
        dfdt = beats_model_dfdt
        ddf = 7
    else:
        dfdt = model
        ddf = 1

    t = data_dict['t']
    pos = data_dict['pos']
    t_err = data_dict['t_err']
    pos_err = data_dict['pos_err']

    # iterate curve_fit to get the best results,
    # propagating the error on t over sigma using
    #   sigma^2 = y_err^2 + (df/dx)^2 * x_err^2

    sigma = pos_err
    for _ in range(N):
        popt, pcov = curve_fit(model, t, pos, sigma = sigma, p0 = p0, bounds = bounds)
        perr = np.sqrt(np.diag(pcov))
        sigma = np.sqrt(pos_err**2 + (dfdt(t, *popt))**2 * t_err**2)

    # this part could be improved
    chi2 = 0
    ni = 0
    if print_results:

        if model == pendulum_model:

            A_hat, omega_hat, phi_hat, lambda_hat, offset_hat = popt
            A_err, omega_err, phi_err, lambda_err, offset_err = perr

            print('BEST FIT parameters:')
            print(f'  omega [rad/s]\t= {omega_hat} ± {omega_err:.2g}')
            print(f'  phi [rad]\t= {phi_hat} ± {phi_err:.2g}')
            print(f'  lambda [s-1]\t= {lambda_hat} ± {lambda_err:.2g}')
            print(f'  A [au]\t= {A_hat} ± {A_err:.2g}')
            print(f'  offset [au]\t= {offset_hat} ± {offset_err:.2g}')

        elif model == beats_model:

            A_hat, lambda_hat, omega_p_hat, omega_b_hat, phi_p_hat, phi_b_hat, offset_hat = popt
            A_err, lambda_err, omega_p_err, omega_b_err, phi_p_err, phi_b_err, offset_err = perr

            print('BEST FIT parameters:')
            print(f'  A [au]\t= {A_hat} ± {A_err:.2g}')
            print(f'  offset [au]\t= {offset_hat} ± {offset_err:.2g}')
            print(f'  lambda [s-1]\t= {lambda_hat} ± {lambda_err:.2g}')
            print(f'  omega_p [rad/s]\t= {omega_p_hat} ± {omega_p_err:.2g}')
            print(f'  omega_b [rad/s]\t= {omega_b_hat} ± {omega_b_err:.2g}')
            print(f'  phi_p [rad]\t= {phi_p_hat} ± {phi_p_err:.2g}')
            print(f'  phi_b [rad]\t= {phi_b_hat} ± {phi_b_err:.2g}')

        else:

            print('BEST FIT parameters:')
            print(popt)
            print(perr)

        # find decay time
        tau = 1/lambda_hat
        tau_err =  lambda_err * tau**2
        print(f'Decay time tau [s] = {tau} ± {tau_err:.2g}')
        
        # find chi2 value
        res = pos - model(t, *popt)
        chi2 = ((res/sigma)**2).sum()
        ni = len(pos) - ddf
        ni_err = np.sqrt(2*ni)
        chi2_sigma_diff = (chi2 - ni)/ni_err
        print(f'chi2 = {chi2:.1f}/{ni}, ({chi2_sigma_diff:.2f} sig)')

    return popt, perr, chi2, ni