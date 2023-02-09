import numpy as np
import matplotlib.pyplot as plt

SAMPLES = 2048
ITERATIONS = SAMPLES * 2

# oscilloscope parameters
horizontal_divs = 10 # number of (horizontal) divisions on the display
vertical_divs = 8 # vertical divisions
cutoff = 10 # [Hz] high-pass cutoff frequency
sweeptime = 10e-3 # [s/DIV]
vertscale = 0.2 # [V/DIV]

# input wave parameters
f = 30 # [Hz] wave frequency
amplitude = 1 # [V]

t = np.linspace(0, horizontal_divs*sweeptime, SAMPLES)
wave = np.zeros(SAMPLES)

for k in range(1, ITERATIONS+1, 2):

    fk = k*f
    ck = 2 / (k * np.pi)
    gaink = 1/np.sqrt(1 + (cutoff/fk)**2)
    phasek = np.arctan(cutoff/fk)

    wave += gaink * ck * np.sin(2*np.pi*fk*t + phasek)

wave *= amplitude

plt.style.use(["science"])
plt.figure(figsize=(5,4))
ax = plt.axes()
ax.set_facecolor('#1e1e1e')

# grid
plt.locator_params(axis='y', nbins=vertical_divs)
plt.locator_params(axis='x', nbins=horizontal_divs)
plt.grid(color="#707070")

plt.xlabel("Tempo [$s$]")
plt.ylabel("Ampiezza [$V$]")

props = dict(boxstyle='round', facecolor='white', alpha=0.8)
plt.text(1, 1, f"{vertscale}V/DIV\n{sweeptime}s/DIV", fontsize=11, verticalalignment="top", horizontalalignment="right", bbox=props, transform=ax.transAxes)

plt.plot(t, wave, color="lightgreen")
plt.xlim(0, horizontal_divs*sweeptime)
plt.ylim(-vertical_divs*vertscale/2, vertical_divs*vertscale/2)
plt.savefig("ACcoupling/oscilloscope_ACcoupling.png")