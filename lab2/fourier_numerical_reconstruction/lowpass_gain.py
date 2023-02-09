import numpy as np
import matplotlib.pyplot as plt

SAMPLES = 2048
ITERATIONS = SAMPLES*2

cutoff = 50 # [Hz]
freqs = np.logspace(0, 4, 50)
times = np.linspace(0, 1, SAMPLES)

amps = np.array([])
# iterate over frequencies
for f in freqs:

    t = times * 1/f
    wave = np.zeros(SAMPLES)

    # fourier sum
    for k in range(1,ITERATIONS+1,2):
        fk = f*k
        gaink = 1/np.sqrt(1 + (fk/cutoff)**2)
        phasek = np.arctan(-fk/cutoff)
        wave += gaink * 2/(np.pi*k) * np.sin(2*np.pi*fk*t + phasek)

    # append output amplitude
    amps = np.append(amps, wave.max() - wave.min()) 

def G(f):
    """Gain function"""
    return 1 / np.sqrt(1 + (f/cutoff)**2)

# plot
plt.style.use(["science"])
plt.figure(figsize=(4,3),dpi=320)

plt.plot(freqs, amps, label="Simulazione", color="blue", zorder=2)
plt.plot(freqs, G(freqs), label="$1/\\sqrt{1 + (f/f_T)^2}$", ls="--", color="red", alpha=0.75, zorder=1)

plt.yscale("log")
plt.xscale("log")

plt.xlim(1, 1e4)

plt.title(f"Filtro passa-basso, $f_T = {cutoff}$ Hz")
plt.xlabel("$f$ [Hz]")
plt.ylabel("Guadagno")
plt.grid()
plt.legend()

plt.savefig("lowpass_gain/lowpass_gain.png")