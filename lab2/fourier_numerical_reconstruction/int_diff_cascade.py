import numpy as np
import matplotlib.pyplot as plt

SAMPLES = 2048 # number of samples
N = SAMPLES * 2 # number of iterations

# cutoff frequencies for integrator and differentiator
cutoff_int = 1.03 # Hz
cutoff_diff = 10.7e3 # Hz

# estimated gain for frequencies that are much greater than
# cutoff_int and much smaller than cutoff_diff.
print(f"expected G = {cutoff_int/cutoff_diff:.4g}")

# create an array of different frequencies
freqs_start = np.log10(cutoff_int)
freqs_stop = np.log10(cutoff_diff*1.5)
freqs_rows = 6
freqs = np.logspace(freqs_start,freqs_stop,freqs_rows*2) # generate freqs_N log-spaced numbers

# initialize plot
plt.style.use(["science"])
fig, axes = plt.subplots(nrows=freqs_rows, ncols=2, sharex=True, sharey=True, figsize=(4, 5), dpi=320)
axes = np.array(axes).flatten()

# iterate over every frequency
for f, ax in zip(freqs, axes):
    
    print(f"f = {f:.1f} Hz")
    wave = np.zeros(SAMPLES)
    t = np.linspace(-2/f, 2/f, SAMPLES)
    
    # iterate over odd numbers
    for k in range(1,N,2):

        fk = k*f
        ck = 2 / (k * np.pi) # k-th fourier coefficient
        gain_int = 1 / np.sqrt(1 + (fk/cutoff_int)**2)
        gain_diff = 1 / np.sqrt(1 + (cutoff_diff/fk)**2)
        phaseshift_int = np.arctan(-fk/cutoff_int)
        phaseshift_diff = np.arctan(cutoff_diff/fk)

        wave += ck * gain_int * gain_diff * np.sin(2*np.pi*fk*t + phaseshift_int + phaseshift_diff)

    print(f"G = {wave.max() - wave.min():.4g}") # simple gain estimate

    ax.plot(t*f, 1e4*wave, "blue")
    ax.grid()
    ax.set_ylim(-1, 1)
    ax.set_xlim(-2, 2)

    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(1, 0, f"{f:.1f} Hz", fontsize=8, verticalalignment="center", horizontalalignment="right", bbox=props, transform=ax.transAxes)

fig.supylabel("Ampiezza [$10^{-4}$ u.a.]")
fig.supxlabel("Tempo [periodi]")
plt.savefig("int_diff_cascade/int_diff_cascade.png")