import numpy as np
import json
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import minimize


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
err = 2  # error on pixel measuremenet [px]
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
print(f"\npixel-to-radian conversion factor:\n\t{px_to_rad:.4g} rad/px")
# TODO: uncertainty on conversion factor


def cost_fun(x, data_x, data_y, err=2):
    """Returns chi2 value of circular fit"""

    x0, y0, R = x  # unpack x
    C = (x0, y0)  # center of the circumference
    r = (data_x - C[0], data_y - C[1])  # relative positions from center
    dist = np.sqrt(r[0] ** 2 + r[1] ** 2)  # distances from center
    chi2 = ((dist - R) ** 2 / err**2).sum()

    return chi2


# circular fit
x0 = (620, 380, 190)
res = minimize(cost_fun, x0=x0, args=(x, y, err))
print(f"\ncircular fit results:\n\t{res}")

# chi2
chi2 = res.fun
dof = len(x) - 3
chi2_sigma_dist = (chi2 - dof) / np.sqrt(2 * dof)
print(f"\nchi2 result:\n\t{chi2:.1f}/{dof} ({chi2_sigma_dist:.1f} sig)")

# calculate angular radius of halo
x0, y0, R = res.x
angular_radius = R * px_to_rad  # [rad]
angular_radius_deg = angular_radius * (180 / np.pi)  # [degrees]
print(
    f"\nangular radius of halo:\n\t{angular_radius:.4g} rad, or\n\t{angular_radius_deg:.4g} degrees"
)

# setup plot
plt.style.use(["science", "grid"])
fig = plt.figure(figsize=(4, 4), dpi=600)

# plot style configuration
color_data = "#004488"
color_fit = "#bb5566"
color_star = "#367e65"
alpha_fit = 0.7
alpha_star_lines = 0.5
scatter_size = 5
text_offset = (8, -8)
text_bbox = {"facecolor": "white", "edgecolor": "none", "pad": 1}

# plot datapoints
ax = fig.add_subplot(111)
ax.errorbar(x, y, fmt=".", xerr=err, yerr=err, color=color_data, label="Dati", zorder=3)

# draw best-fit circumference
theta = np.linspace(0, 2 * np.pi, 100)
x_circ = x0 + R * np.cos(theta)
y_circ = y0 + R * np.sin(theta)
ax.scatter(x0, y0, s=scatter_size, color=color_fit, zorder=2)
ax.text(x0 - text_offset[0], y0 + text_offset[1], "C", color=color_fit)
ax.plot(
    x_circ, y_circ, color=color_fit, alpha=alpha_fit, label="Fit circolare", zorder=2
)

# plot position and names of stars
for star in stars:

    ax.scatter(*stars[star]["img_coords"], s=scatter_size, color=color_star, zorder=1)
    ax.text(
        *(stars[star]["img_coords"] + text_offset),
        star,
        fontsize=8,
        color=color_star,
        bbox=text_bbox,
    )

# draw lines between stars
for pair, name in zip(pairs, pair_names):

    A = stars[pair[0]]["img_coords"]
    B = stars[pair[1]]["img_coords"]

    c = (A + B) / 2  # text box position
    xx = (A[0], B[0])
    yy = (A[1], B[1])

    ax.plot(xx, yy, ls="--", color=color_star, alpha=alpha_star_lines, zorder=1)
    ax.text(
        *c,
        f"{ang_dists[name] * (180/np.pi):.3g}°",
        fontsize=8,
        bbox=text_bbox,
        color=color_star,
    )

# pixel/radian comparison
ax.errorbar(370, 180, xerr=0.05 / px_to_rad, yerr=0, capsize=2, color="#000000")
ax.text(370 - 38, 180 - 15, "$0.1$ rad")

# x, y limits
ax.set_xlim((300, 850))
ax.set_ylim((20, 600))

# grid, labels, legend
ax.invert_yaxis()  # invert the vertical axis
ax.set_title("Alone lunare: fit circolare")
ax.set_xlabel("x [px]")
ax.set_ylabel("y [px]")
ax.set_aspect(1)
ax.legend()
ax.grid(which="both", axis="both", color="lightgray", zorder=0)

plt.savefig("graphs/halo.png")
