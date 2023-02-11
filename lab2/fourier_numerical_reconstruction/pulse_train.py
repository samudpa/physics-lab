import numpy as np
import matplotlib.pyplot as plt

SAMPLES = 2048 # number of samples
ITERATIONS = SAMPLES * 8 # number of iterations

duties = np.arange(0,1,1/10)[2:]
print(duties)

t = np.linspace(-2, 2, SAMPLES)
f = 1
cutoff = f/50

# initialize plot
plt.style.use(["science"])
props = dict(boxstyle='round', facecolor='white', alpha=0.9)

# init figures
fig_pulsetrain = plt.figure(1, figsize=(4, 4), dpi=320) # pulse train figure
fig_integrator = plt.figure(2, figsize=(4, 4), dpi=320) # pulse train through differentiator

# init subplots
axes_p = fig_pulsetrain.subplots(nrows=4, ncols=2, sharex=True, sharey=True)
axes_i = fig_integrator.subplots(nrows=4, ncols=2, sharex=True, sharey=True)
axes_p = np.array(axes_p).flatten()
axes_i = np.array(axes_i).flatten()

for duty, ax_p, ax_i in zip(duties, axes_p, axes_i):

    # initialize wave
    wave_p = np.zeros(SAMPLES) + duty
    wave_i = np.zeros(SAMPLES) + duty

    for k in range(1, ITERATIONS+1):
        
        fk = f*k
        ck = 2/(k*np.pi) * np.sin(k*np.pi*duty)

        gaink = 1 / np.sqrt(1 + (fk/cutoff)**2)
        phasek = np.arctan(-fk/cutoff)

        wave_p += ck * np.cos(2*np.pi*fk*t)
        wave_i += gaink * ck * np.cos(2*np.pi*fk*t + phasek)

    wave_i_noDC = wave_i - wave_i.mean() #
    print(f"Vripple = {wave_i.max()-wave_i.min()}")

    ax_p.plot(t,wave_p,"blue") # pulse train
    ax_p.plot(t,wave_i,"red") # integrated pulse train
    ax_p.set_xlim(-2,2)
    ax_p.set_ylim(-0.1,1.1)
    ax_p.grid()

    # integrated pulse train, without DC component
    ax_i.plot(t,1e2*wave_i_noDC,"red")
    ax_i.set_xlim(-2,2)
    ax_i.grid()

    # add textboxes
    ax_p.text(1, 0, f"$\delta = {duty:.1f}$", fontsize=8,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=props, transform=ax_p.transAxes)
    ax_i.text(1, 0, f"$\delta = {duty:.1f}$", fontsize=8,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=props, transform=ax_i.transAxes)

fig_pulsetrain.supylabel("Ampiezza [u.a.]")
fig_pulsetrain.supxlabel("Tempo [n° di periodi]")
fig_integrator.supylabel("Ampiezza [$10^{-2}$ u.a.]")
fig_integrator.supxlabel("Tempo [n° di periodi]")

fig_pulsetrain.savefig("pulse_train/pulse.png")
fig_integrator.savefig("pulse_train/pulse_integrated_withoutDC.png")