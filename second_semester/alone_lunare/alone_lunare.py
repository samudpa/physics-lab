import numpy as np
import json
import itertools
from scipy.optimize import minimize, curve_fit
from utils import fmt_measure
from draw_plot import draw_halo, draw_halo_residuals


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


# read lunar halo data (image coordinates)
err = 1.5  # error on pixel measuremenet [px]
x, y = np.loadtxt("data/halo.csv", delimiter=", ", unpack=True)

# read calibration data (star astronomical coordinates and image coordinates)
#   https://en.wikipedia.org/wiki/Spica
#   https://en.wikipedia.org/wiki/Regulus
#   https://en.wikipedia.org/wiki/Arcturus
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

# find angular and image distances between stars
ang_dists = []
img_dists = []
for pair in pairs:

    ang_dists.append(
        ang_dist(*stars[pair[0]]["astro_coords"], *stars[pair[1]]["astro_coords"])
    )
    img_dists.append(dist(stars[pair[0]]["img_coords"], stars[pair[1]]["img_coords"]))

img_dist_err = err * np.sqrt(2)  # error on pixel distances

print(f"angular distances between stars [rad]:\n\t{ang_dists}")
print(f"pixel distances between stars [px]:\n\t{img_dists}")
print(f"pixel distance error [px]:\n\t{img_dist_err}")

# is a linear relationship between angular and pixel distances good enough?


def line(x, m, q):
    """Linear fit model"""
    return x * m + q


# convert list of distances to numpy arrays
ang_dists = np.array(ang_dists)
img_dists = np.array(img_dists)

# linear fit (x = ang_dists, y = img_dists)
popt, pcov = curve_fit(line, ang_dists, img_dists)
perr = np.sqrt(np.diag(pcov))
print(f"\nlinear fit on angular/pixel relationship:")
print(f"\tpopt: {popt}\n\tperr: {perr}")

# chi2
res = img_dists - line(ang_dists, *popt)
print(f"linear fit residuals:\n\t{res}")
chi2 = (res**2 / img_dist_err**2).sum()
dof = len(ang_dists) - 2
chi2_sigma_dist = (chi2 - dof) / np.sqrt(2 * dof)

print(f"linear fit chi2:\n\t{chi2:.1f}/{dof} ({chi2_sigma_dist:.1f} sig)")


# find conversion factor
px_tot = img_dists.sum()
rad_tot = ang_dists.sum()
px_to_rad = rad_tot / px_tot
# error
px_tot_err = img_dist_err * np.sqrt(len(img_dists))
px_to_rad_err = px_tot_err * px_to_rad / px_tot
print(f"\npixel-to-radian factor:\n\t{fmt_measure(px_to_rad, px_to_rad_err)} rad/px")


def cost_fun(x, data_x, data_y, err=2):
    """Returns chi2 value of circular fit"""

    x0, y0, R = x  # unpack x
    r = (data_x - x0, data_y - y0)  # relative positions from center
    dist = np.sqrt(r[0] ** 2 + r[1] ** 2)  # distances from center
    chi2 = ((dist - R) ** 2 / err**2).sum()

    return chi2


# circular fit
x0 = (620, 380, 190)
res = minimize(cost_fun, x0=x0, args=(x, y, err))
print(f"\ncircular fit results:\n{res}")

# chi2
chi2 = res.fun
dof = len(x) - 3
chi2_sigma_dist = (chi2 - dof) / np.sqrt(2 * dof)
print(f"\nchi2 result:\n\t{chi2:.1f}/{dof} ({chi2_sigma_dist:.1f} sig)")

# calculate angular radius of halo
x0, y0, R = res.x
angular_radius = R * px_to_rad  # [rad]
angular_radius_deg = angular_radius * (180 / np.pi)  # [degrees]
print(f"\nlunar halo angular radius:\n\t{angular_radius:.4g} rad, or")
print(f"\t{angular_radius_deg:.4g} degrees")

# plot halo datapoints and circular best-fit
# draw_halo(x, y, err, px_to_rad, x0, y0, R, stars, pairs, ang_dists)

r = (x - x0, y - y0)  # datapoints relative positions
res = np.sqrt(r[0] ** 2 + r[1] ** 2) - R  # residuals
theta = np.arctan2(*r[::-1])  # find angle of each datapoint

indexes = np.argsort(theta)
theta = theta[indexes]
res = res[indexes]

draw_halo_residuals(theta, res, err, R, chi2, dof)
