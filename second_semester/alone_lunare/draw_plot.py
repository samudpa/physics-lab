import matplotlib.pyplot as plt
import numpy as np

# plot style configuration
color_data = "#004488"
color_fit = "#bb5566"
color_star = "#367e65"
alpha_fit = 0.7
alpha_star_lines = 0.5
scatter_size = 5
text_offset = (8, -8)
text_bbox = {"facecolor": "white", "edgecolor": "none", "pad": 1}


def draw_halo(
    x,
    y,
    err,  # x, y coordinates and error
    px_to_rad,  # pixel to radian conversion factor
    x0,
    y0,
    R,  # center coordinates
    stars,
    pairs,
    ang_dists,  # star data, pairs of stars and angular distances
    filename="graphs/circular_fit.png",  # save figure as
):

    # setup plot
    plt.style.use(["science", "grid"])
    fig = plt.figure(figsize=(4, 4), dpi=600)

    # plot datapoints
    ax = fig.add_subplot(111)
    ax.errorbar(
        x, y, fmt=".", xerr=err, yerr=err, color=color_data, label="Dati", zorder=3
    )

    # draw best-fit circumference
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circ = x0 + R * np.cos(theta)
    y_circ = y0 + R * np.sin(theta)
    ax.scatter(x0, y0, s=scatter_size, color=color_fit, zorder=2)
    ax.text(x0 - text_offset[0], y0 + text_offset[1], "C", color=color_fit)
    ax.plot(
        x_circ,
        y_circ,
        color=color_fit,
        alpha=alpha_fit,
        label="Fit circolare",
        zorder=2,
    )

    # plot position and names of stars
    for star in stars:

        ax.scatter(
            *stars[star]["img_coords"], s=scatter_size, color=color_star, zorder=1
        )
        ax.text(
            *(stars[star]["img_coords"] + text_offset),
            star,
            fontsize=8,
            color=color_star,
            bbox=text_bbox,
        )

    # draw lines between stars
    for pair, ang_dist in zip(pairs, ang_dists):

        A = stars[pair[0]]["img_coords"]
        B = stars[pair[1]]["img_coords"]

        c = (A + B) / 2  # text box position
        xx = (A[0], B[0])
        yy = (A[1], B[1])

        ax.plot(xx, yy, ls="--", color=color_star, alpha=alpha_star_lines, zorder=1)
        ax.text(
            *c,
            f"{ang_dist * (180/np.pi):.3g}°",
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

    plt.savefig(filename)


def draw_halo_residuals(
    theta, res, err, R, chi2, dof, filename="graphs/circular_fit_residuals.png"
):

    # setup plot
    plt.style.use(["science", "grid"])
    fig = plt.figure(figsize=(4, 2.6), dpi=600)

    # datapoints
    ax = fig.add_subplot(111)
    ax.errorbar(
        theta,
        res,
        fmt=".",
        xerr=err / R,
        yerr=err,
        color=color_data,
        label="Dati",
        zorder=3,
    )

    # model horizontal line
    ax.axhline(0, color=color_fit, label="Modello")

    # xticks in multiples of pi
    # https://stackoverflow.com/a/61737382
    ax.set_xticks(np.arange(-np.pi, np.pi + 0.01, np.pi / 4))
    labels = ["$-\pi$", "", "$-\pi/2$", "", "0", "", "$\pi/2$", "", "$\pi$"]
    ax.set_xticklabels(labels)

    # x, y limits
    ax.set_xlim((-np.pi - 0.2, np.pi + 0.2))
    ax.set_ylim((-5.5, 5.5))

    # chi2 result
    ax.text(0, +4.3, f"$\chi^2/\\nu = {chi2:.1f}/{dof}$", ha="center")

    # grid, labels, legend
    ax.set_title("Alone lunare: residui fit circolare")
    ax.set_xlabel("$\\theta$ [rad]")
    ax.set_ylabel("Residui [px]")
    ax.legend(loc="lower left")
    ax.grid(which="both", axis="both", color="lightgray", zorder=0)

    plt.savefig(filename)


def draw_refractive_index(
    x, y, x_poly, y_poly, labels_poly, filename="graphs/refractive_index.png"
):

    # setup plot
    plt.style.use(["science", "grid"])
    fig = plt.figure(figsize=(4, 2.6), dpi=600)

    # graph
    ax = fig.add_subplot(111)
    ax.plot(x, y, color=color_fit, label="$n(\\alpha)$", zorder=2)

    # horizontal line at n=1
    # ax.axhline(1, color="#000000", linestyle="--", zorder=1)

    # horizontal line connecting to triangle/hexagon
    ax.axhline(1.3, color=color_data, ls="--", label="$n = 1.3$", zorder=4)

    # vertical lines (regular polygons)
    for i, (xx, yy, label) in enumerate(zip(x_poly, y_poly, labels_poly)):

        if i != 3:  # skip hexagon since it's equal to triangle

            # vertical line
            if i == 0:

                # add label
                ax.axvline(
                    xx,
                    ls="--",
                    color="gray",
                    label="prismi\nregolari",
                    alpha=alpha_fit,
                    zorder=3,
                )

            else:

                # don't add label
                ax.axvline(xx, ls="--", color="gray", alpha=alpha_fit, zorder=3)

            ax.scatter(xx, yy, s=6, zorder=5)
            ax.text(
                xx, 3.6, label, ha="center", va="center", color="gray", bbox=text_bbox
            )

    # xticks in multiples of pi
    # https://stackoverflow.com/a/61737382
    ax.set_xticks(np.arange(0, np.pi + np.pi / 8 + 0.01, np.pi / 4))
    labels = ["0", "$\pi/4$", "$\pi/2$", "$3\pi/4$", "$\pi$"]
    ax.set_xticklabels(labels)

    # x, y limits
    ax.set_xlim((0, np.pi - np.pi / 8))
    ax.set_ylim((0.65, 4))

    # grid, labels, legend
    ax.set_title("Indice di rifrazione")
    ax.set_xlabel("$\\alpha$ [rad]")
    ax.set_ylabel("$n$")
    ax.legend(loc="upper right")
    ax.grid(which="both", axis="both", color="lightgray", zorder=0)

    plt.savefig(filename)
