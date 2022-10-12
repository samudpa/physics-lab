import numpy as np


def fmt_measure(value, err, sig=2, sep=" \pm "):
    """Returns a formatted string of (value, err) with the error rounded to sig significant figures"""

    decimals = int(sig - np.floor(np.log10(np.abs(err))) - 1)
    if decimals >= 0:
        formatter = "{:." + str(decimals) + "f}"
        return formatter.format(value) + sep + formatter.format(err)
    if decimals < 0:
        return f"{int(np.round(value, decimals))}{sep}{int(np.round(err, decimals))}"
