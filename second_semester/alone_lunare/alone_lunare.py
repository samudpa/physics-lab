import numpy as np
import json
import itertools
from scipy.optimize import curve_fit, least_squares
from utils import fmt_measure
from draw_plot import (
    draw_fit,
    draw_halo,
    draw_halo_residuals,
    draw_refractive_index,
)


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
err = 1.5  # error on pixel measurement [px]
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

# draw fit results
text = f"$m = {fmt_measure(popt[0], perr[0])}$ px/rad\n$q = {fmt_measure(popt[1], perr[1])}$\n$\chi^2/\\nu = {chi2:.1f}/{dof}$"
draw_fit(ang_dists, img_dists, 0, err, res, line, popt, text)

# find conversion factor
px_to_rad = 1 / popt[0]
px_to_rad_err = perr[0] / (popt[0] ** 2)
# standard error
print(f"\npixel-to-radian factor:\n\t{fmt_measure(px_to_rad, px_to_rad_err)} rad/px")


def circular_fit_res(x, data_x, data_y):
    """Returns residuals of circular fit"""

    x0, y0, R = x  # unpack x
    r = (data_x - x0, data_y - y0)  # relative positions from center
    dist = np.sqrt(r[0] ** 2 + r[1] ** 2)  # distances from center
    res = dist - R

    return res


# circular fit
x0 = (620, 380, 190)
res_lsq = least_squares(circular_fit_res, x0=x0, args=(x, y))
x0, y0, R = res_lsq.x
# find standard error
J = res_lsq.jac  # jacobian matrix
pcov = np.linalg.inv(J.T @ J)
x0_err, y0_err, R_err = np.sqrt(np.diag(pcov))
print(f"\ncircular fit results:")
print(f"\tx0 [px] = {fmt_measure(x0, x0_err)}")
print(f"\ty0 [px] = {fmt_measure(y0, y0_err)}")
print(f"\tR  [px] = {fmt_measure(R, R_err)}")

# chi2
chi2 = res_lsq.cost / (err**2)
dof = len(x) - 3
chi2_sigma_dist = (chi2 - dof) / np.sqrt(2 * dof)
print(f"\nchi2 result:\n\t{chi2:.1f}/{dof} ({chi2_sigma_dist:.1f} sig)")

# calculate angular radius of halo
# radians and degrees
angular_radius = R * px_to_rad  # [rad]
angular_radius_deg = angular_radius * (180 / np.pi)  # [degrees]
# errors
angular_radius_err = angular_radius * np.sqrt(
    (R_err / R) ** 2 + (px_to_rad_err / px_to_rad) ** 2
)
angular_radius_deg_err = angular_radius_err * (180 / np.pi)
# print results
print(
    f"\nlunar halo angular radius:\n\t{fmt_measure(angular_radius, angular_radius_err)} rad, or"
)
print(f"\t{fmt_measure(angular_radius_deg, angular_radius_deg_err)} degrees")

# plot halo datapoints and circular best-fit
draw_halo(x, y, err, px_to_rad, x0, y0, R, stars, pairs, ang_dists)

r = (x - x0, y - y0)  # datapoints relative positions
res = res_lsq.fun  # residuals
theta = np.arctan2(*r[::-1])  # find angle of each datapoint

indexes = np.argsort(theta)
theta = theta[indexes]
res = res[indexes]

# plot circular fit residuals
draw_halo_residuals(theta, res, err, R, chi2, dof)


def refractive_index(theta_min, apex_angle, theta_min_err=0):
    """Returns the refractive index (and its standard error) knowing
    minimum angle of deviation and apex angle"""
    n = np.sin((theta_min + apex_angle) / 2) / np.sin(apex_angle / 2)
    n_err = theta_min_err * np.abs(
        1 / 2 * np.cos((theta_min + apex_angle) / 2) / np.sin(apex_angle / 2)
    )
    return n, n_err


def vertex_angle(n):
    """Returns the vertex angle of a regular polygon with n sides"""
    return (n - 2) * np.pi / n


def to_radians(degrees):
    """degree -> radians conversion"""
    return degrees / 180 * np.pi


# apex angles of regular polygons (3 to 8 sides)
apex_angle_poly = np.array(
    [
        vertex_angle(3),  # triangle
        vertex_angle(4),  # square
        to_radians(120) - vertex_angle(5),  # pentagon
        vertex_angle(6) / 2,  # hexagon
        to_radians(540) - 4 * vertex_angle(7),  # heptagon
        to_radians(180) - vertex_angle(8),  # octagon
    ]
)
n_poly, n_poly_err = refractive_index(
    angular_radius, apex_angle_poly, angular_radius_err
)

# refractive_index() graph
apex_angle_graph = np.linspace(0.01, 2 * np.pi - 0.01, 100)
n_graph, _ = refractive_index(angular_radius, apex_angle_graph)

# special polygons labels
poly_labels = ["$3/6$\nlati", "$4$\nlati", "$5$\nlati", "", "$7$\nlati", "$8$\nlati"]
# draw refractive_index() graph
draw_refractive_index(apex_angle_graph, n_graph, apex_angle_poly, n_poly, poly_labels)

# print results for regular polygons
print("\nrefractive indexes for regular polygons from 3 to 8 sides:")
print("\t(sides, vertex angle, apex angle, n)")
for i, (alpha, n, n_err) in enumerate(zip(apex_angle_poly, n_poly, n_poly_err)):
    print(
        f"\t{i+3}   {vertex_angle(i+3)/np.pi*180:.2f}°   {alpha/np.pi*180:.2f}°   {fmt_measure(n,n_err)}"
    )

# print hexagon refractive index calculation results
n_exp = 1.31
n = n_poly[3]
n_err = n_poly_err[3]
n_sigma_dist = (n - n_exp) / n_err
print("\nrefractive index for hexagon:")
print(f"\tn = {fmt_measure(n, n_err)}\n\tn_exp = {n_exp}")
print(f"\tn_sigma_dist = {n_sigma_dist:.1f}")
