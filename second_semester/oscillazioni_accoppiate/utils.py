import numpy as np

def load_data(filename, cutoff=600):
    '''Load time and position data from .txt'''

    # load and format data inside a dict
    data = np.loadtxt(filename, unpack=True)
    data_dict = {
        'red': {
            't': data[0], 'pos': data[1]
        },
        'blue': {
            't': data[2], 'pos': data[3]
        }
    }

    # discards the data after the cutoff
    if cutoff != None:
        for color in data_dict.keys():
            for category in data_dict[color].keys():
                data_dict[color][category] = data_dict[color][category][0:cutoff]

    return data_dict

def find_roots(x,y):
    '''Find the roots of a set of points'''
    sign = np.sign(y)
    np.place(sign, sign==0, [1])
    sign_diff = np.diff(sign)
    indexes = np.concatenate([abs(sign_diff) == 2, [False]])
    return x[indexes]

def find_period(t,pos):
    '''Find the period of a wave'''
    roots = find_roots(t, pos)
    t_diff = np.diff(roots)

    # period
    T = np.mean(t_diff) * 2
    T_err = 0.25 * T / np.sqrt(len(t_diff))

    # angular frequency
    omega = 2 * np.pi / T
    omega_err = omega * T_err / T

    return T, T_err, omega, omega_err