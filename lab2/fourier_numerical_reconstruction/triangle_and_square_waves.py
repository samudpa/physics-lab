import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# parameters
SAMPLES = 2048 # number of samples in [-PERIOD, PERIOD]
PERIOD = 1

omega = 2 * np.pi / PERIOD # base frequency [rad/s]
t = np.linspace(-PERIOD, PERIOD, SAMPLES)

# initialize square and triangle wave arrays
square_wave = np.zeros(SAMPLES)
triangle_wave = np.zeros(SAMPLES)

# waves to compare the results with
model_square_wave = 0.5 * signal.square(omega * t)
model_triangle_wave = - 0.5 * signal.sawtooth(omega * t, 0.5)

# plot the intermediate results of the sum at these values of N
Ns = np.array([1, 3, 5, 11, 51, 101, 1001, 5001, 10001])

# initialize plot
plt.style.use(["science"])
fig, axes = plt.subplots(nrows=Ns.size, ncols=2, sharex=True, sharey=True, figsize=(4, 7), dpi=400)
row_counter = 0

# iterate over ODD numbers since EVEN coefficients are = 0
for k in range(1, Ns.max() + 1, 2):

    # k-th coefficients of the fourier series
    ck_square = 2 / (k * np.pi) # SINE coefficient (square wave)
    bk_triangle = 4 / (k * np.pi) ** 2 # COSINE coefficient (triangle wave)

    # add the k-th term to the sum
    square_wave += ck_square * np.sin(k * omega * t)
    triangle_wave += bk_triangle * np.cos(k * omega * t)

    # plot the intermediate result
    if k in Ns:

        # sum of squared residuals. Used to determine "quality" of results
        sumres_square = ((square_wave - model_square_wave)**2).sum()
        sumres_triangle = ((triangle_wave - model_triangle_wave)**2).sum()
        print(f"N = {k}; sumres_square   = {sumres_square-0.5:.2g} ({sumres_square:.2g})")
        print(f"N = {k}; sumres_triangle = {sumres_triangle:.2g}")
        # NOTE: if sumres_square converges to a value >0, that is because of
        #       aliasing problems at the "jumping" points between y=-0.5 and y=0.5

        # plot results
        ax_square = axes[row_counter, 0]
        ax_triangle = axes[row_counter, 1]
        ax_square.plot(t, square_wave, "blue")
        ax_triangle.plot(t, triangle_wave, "red")

        # grid and bounds
        ax_square.grid()
        ax_triangle.grid()
        ax_square.set_xlim(-1, 1)
        ax_triangle.set_xlim(-1, 1)
        ax_square.set_ylim(-0.7, 0.7)
        ax_triangle.set_ylim(-0.7, 0.7)

        # labels
        ax_triangle.set_ylabel(f"$N = {k}$")
        ax_triangle.yaxis.set_label_position("right")

        row_counter += 1 # increase counter

# labels
fig.tight_layout(rect=[0.03,0.03,1,0.97]) # compact layout
fig.suptitle(f"Onda quadra e triangolare (${SAMPLES}$ campioni)") # requires matplotlib >=3.4
fig.supylabel("Ampiezza [unità arbitraria]")
fig.supxlabel("Tempo [n° di periodi]")

plt.savefig(f"graphs/TRI_and_SQ_waves_{SAMPLES}.png")