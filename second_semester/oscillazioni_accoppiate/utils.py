import numpy as np

def load_raw_data(filename, cutoff=600):
    '''Load time and position data from .txt'''

    # load and format data inside a dict
    data = np.loadtxt(filename, unpack=True)
    data_dict = {
        'red': {'t': data[0], 'pos': data[1]},
        'blue': {'t': data[2], 'pos': data[3]}
    }

    # discards the data after the cutoff
    if cutoff != None:
        for color in data_dict.keys():
            for category in data_dict[color].keys():
                data_dict[color][category] = data_dict[color][category][0:cutoff]

    return data_dict

def load_data(filename, cutoff=600, fix_offset=False, t_err_factor=1, pos_err_factor=1):
    '''Load time, position data from .txt and calculate the errors'''

    # load data of both pendulums
    data_raw = load_raw_data(filename, cutoff)

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

def find_roots(x, y):
    '''Find the roots of a set of points'''
    sign = np.sign(y)
    np.place(sign, sign==0, [1])
    sign_diff = np.diff(sign)
    indexes = np.concatenate([abs(sign_diff) == 2, [False]])
    return x[indexes]

def find_period(t, pos, t_err, **kwargs):
    '''Find the period and angular frequency of a wave'''
    roots = find_roots(t, pos)
    t_diff = np.diff(roots)

    # period
    T = np.mean(t_diff) * 2
    T_err = t_err[0] / np.sqrt(len(t_diff))

    # angular frequency
    omega = 2 * np.pi / T
    omega_err = omega * T_err / T

    return T, T_err, omega, omega_err