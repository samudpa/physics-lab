import math
import sys
import json


def fmt_resistance(
    figures, multiplier, tolerance=None, tcr=None, ohm="Ω", pm="±", **kwargs
):
    """
    Returns a string containing correctly formatted resistance value.
    Examples:
        fmt(33, 0, 5) -> 33Ω ±5%
        fmt(68, 5, 10) -> 6.8MΩ ±10%
    """

    # list of S.I. prefixes (1e-24 ~ 1e24)
    si_prefixes = {
        24: "Y",
        21: "Z",
        18: "E",
        15: "P",
        12: "T",
        9: "G",
        6: "M",
        3: "k",
        0: "",
        -3: "m",
        -6: "μ",
        -9: "n",
        -12: "p",
        -15: "f",
        -18: "a",
        -21: "z",
        -24: "y",
    }

    sig = len(str(figures))  # significant figures
    # magnitude = multiplier + sig  # magnitude of resistance value
    prefix_magnitude = 3 * math.ceil(
        multiplier / 3
    )  # choose the correct prefix's magnitude (24 ~ -24)

    if prefix_magnitude in si_prefixes:
        # if chosen prefix is in the list of S.I. prefixes,
        # correctly shift the digits to the correct magnitude
        shift = multiplier - prefix_magnitude
        figures = figures * 10**shift
    else:
        # fallback
        prefix_magnitude = 0
        figures = figures * 10**multiplier

    unit = si_prefixes[prefix_magnitude] + ohm  # Ω, kΩ, MΩ etc.
    result = f"{figures:.{sig}g}{unit}"

    # add tolerance and TCR values if needed
    if tolerance:
        result += f" {pm}{tolerance}%"
    if tcr:
        result += f" {tcr}ppm/K"

    return result

def get_dictionary_key(key, dic):
    """Returns dic[key] if it exists, else raises an Exception"""
    if key in dic:
        return dic[key]
    else:
        raise Exception(f"Can't find color '{key}'")

def get_resistance(bands):
    """
    Find resistance value from a list of color bands, as per IEC 60062:2016.
    (see https://en.wikipedia.org/wiki/Electronic_color_code#Resistors)
    """


    # load json file containing color to value data
    with open("color_code.json", "r") as json_file:
        data = json.load(json_file)

    # convert all to lowercase, and replace synonims
    colors = []
    for band in bands:
        band = band.lower()  # lowercase
        if band in data["synonims"]:
            band = data["synonims"][band]  # synonims
        colors.append(band)

    # split bands into categories: digits, multiplier, tolerance and TCR
    n = len(colors)
    if n > 6 or n < 3:
        raise Exception(f"Wrong number of arguments ({n}). Please input 3 to 6 colors")
    elif n == 3:
        # 2 digits + multiplier
        figures_bands = colors[0:2]
        multiplier_band = colors[2]
        tolerance_band = None
        tcr_band = None
    elif n == 4:
        # 2 digits + multiplier + tolerance
        figures_bands = colors[0:2]
        multiplier_band = colors[2]
        tolerance_band = colors[3]
        tcr_band = None
    elif n == 5:
        # 3 digits + multiplier + tolerance
        figures_bands = colors[0:3]
        multiplier_band = colors[3]
        tolerance_band = colors[4]
        tcr_band = None
    elif n == 6:
        # 3 digits + multiplier + tolerance + TCR
        figures_bands = colors[0:3]
        multiplier_band = colors[3]
        tolerance_band = colors[4]
        tcr_band = colors[5]
    else:
        raise Exception(
            "Something terribly wrong happened. You shouldn't be getting this."
        )

    # find resistance significant figures (2 or 3, depending on the number of bands)
    figures = []
    for digit_band in figures_bands:
        figures.append(str(get_dictionary_key(digit_band, data["digit"])))
    figures = int("".join(figures))

    # find multiplier, tolerance and TCR values
    multiplier = get_dictionary_key(multiplier_band, data["multiplier"])
    tolerance, tcr = None, None
    if tolerance_band:
        tolerance = get_dictionary_key(tolerance_band, data["tolerance"])
    if tcr_band:
        tcr = get_dictionary_key(tcr_band, data["tcr"])

    # calculate resistance
    resistance = figures * 10**multiplier

    # result dictionary:
    # contains significant figures, multiplier, tolerance and TCR, plus
    # actual resistance (in ohms)
    result = {
        "resistance": resistance,  # [ohm]
        "figures": figures,  # significant figures
        "multiplier": multiplier,
        "tolerance": tolerance,  # %
        "tcr": tcr,  # ppm/K
    }
    return result


if __name__ == "__main__":
    # script has been called from the terminal

    args = sys.argv[1:]  # get command-line arguments, while skipping the script name
    result = get_resistance(args)
    print(fmt_resistance(**result))
