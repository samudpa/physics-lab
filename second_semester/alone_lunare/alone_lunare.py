import numpy as np
import json
import itertools


def ang_dist(alpha_A, delta_A, alpha_B, delta_B):
    """Returns the angular distance [rad] between two points (A, B),
    given their right ascensions (alpha_A, alpha_B)
    and declinations (delta_A, delta_B)"""

    return np.arccos(
        np.sin(delta_A) * np.sin(delta_B)
        + np.cos(delta_A) * np.cos(delta_B) * np.cos(alpha_A - alpha_B)
    )


def dist(vec_A, vec_B):
    """Returns the distance between two points (A, B)"""

    diff = vec_A - vec_B
    return np.sqrt(diff.dot(diff))


# read lunar halo data (image coords)
halo = {}
halo["x"], halo["y"] = np.loadtxt("data/halo.csv", delimiter=", ", unpack=True)

# read calibration data (image coords, galactic coords)
with open("data/calibration.json", "r") as f:
    calibration = json.loads(f.read())

# format calibration data
stars = {}
for star in calibration:

    star_dict = {}
    star_data = calibration[star]

    star_dict["img_coords"] = np.array(
        [star_data["x_px"], star_data["y_px"]]
    )  # image coordinates [px]
    ra = star_data["ra_sec"] * (2 * np.pi / (24 * 3600))  # right ascension (RA)
    dec = star_data["dec_deg"] * (np.pi / 180)  # declination (dec)
    star_dict["astro_coords"] = np.array([ra, dec])  # astronomical coordinates [rad]

    stars[star] = star_dict
print(f"star data:\n\t{stars}")

# find all possible pairs of stars
pairs = list(itertools.combinations(calibration, 2))
pair_names = list(pair[0][0] + pair[1][0] for pair in pairs)

# find angular and image distances between stars
ang_dists = {}
img_dists = {}
for pair, name in zip(pairs, pair_names):

    ang_dists[name] = ang_dist(
        *stars[pair[0]]["astro_coords"], *stars[pair[1]]["astro_coords"]
    )
    img_dists[name] = dist(stars[pair[0]]["img_coords"], stars[pair[1]]["img_coords"])
print(f"angular distances between stars [rad]:\n\t{ang_dists}")
print(f"pixel distances between stars [px]:\n\t{img_dists}")

# find conversion factor
px_tot = sum(img_dists.values())
rad_tot = sum(ang_dists.values())
px_to_rad = rad_tot / px_tot
print(f"pixel-to-radian conversion factor:\n\t{px_to_rad:4f} rad/px")
# TODO: uncertainty on conversion factor
