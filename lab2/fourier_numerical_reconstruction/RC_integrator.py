import numpy as np
import json
import matplotlib.pyplot as plt

# parameters
SUM_ITERATIONS = 10001
DATA_PHASE_SHIFT = np.pi

# load data about the circuit (Resistance, Capacity)
with open("RC_integrator/data/RC_integrator_circuit_data.json") as f:
    circuit_data = json.loads(f.read())
R, R_err = circuit_data["R"], circuit_data["R_err"] # [ohm]
C, C_err = circuit_data["C"], circuit_data["C_err"] # [Fahrad]

# cutoff frequency for the RC filter
cutoff_freq = 1 / (2 * np.pi * (R + 50) * C) # r_G = 50 ohm is the resistor inside the generator
cutoff_freq_err = cutoff_freq * (C_err/C + R_err/R)
print(f"f_T = {cutoff_freq}, {cutoff_freq_err:.2g} Hz")

# load csv file containing information about each measurement
waves_data = np.genfromtxt("RC_integrator/data/RC_integrator_waves.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")

# compute scale multiplier
# the amplitude is taken at a frequency where the filter is irrelevant
# the 1023/3.2 refers to Arduino's digitalizing factor
scale = 1023/3.2 * waves_data[0]["peaktopeak_out"]

# iterate over each entry
for data_entry in waves_data.transpose():
    
    # get filename
    filename = data_entry["filename"]
    print(filename)

    wave_data = np.loadtxt("RC_integrator/data/" + filename + ".txt") # load wave measurements (time in [ms], voltage in [digit]s)
    f = data_entry["frequency"] # base frequency [Hz]
    f_err = data_entry["frequency_err"] # [Hz]
    T = 1/f # period [s]
    
    # prepare simulated wave array
    samples = len(wave_data) # total number of samples
    t_min = wave_data[:,0][0]
    t_max = wave_data[:,0][samples-1]
    t = np.linspace(t_min, t_max, samples) # create an array of times [us]
    simulated_wave = np.zeros(samples)

    # fourier series
    for k in range(1, SUM_ITERATIONS + 1, 2):

        freq_k = f * k # [Hz]
        gain_k = 1 / np.sqrt(1 + (freq_k/cutoff_freq)**2) # gain
        phaseshift_k = np.arctan(-freq_k/cutoff_freq) + DATA_PHASE_SHIFT # phase shift [rad]
        coeff_k = 2 / (k * np.pi) # sine coefficient

        simulated_wave += coeff_k * gain_k * np.sin(2 * np.pi * freq_k * t * 1e-6 + phaseshift_k) # add k-th term to the sum

    # find DC offset in [digit]s, discarding points that may skew results
    # here we compute the number of COMPLETE wave periods inside the data,
    # and compute the mean inside a window large exactly N periods, centered at the half point
    # between t_min and t_max
    N_complete_periods = np.floor(1e-6 * t_max * f) # total number N of complete periods
    T_complete_periods = 1e6 * N_complete_periods * T # total time needed to complete said periods [us]
    dt = np.diff(wave_data[:,0]).mean() # real time difference between measurements [us]
    total_indexes = T_complete_periods / dt # number of array elements that correspond to N periods

    mean_min_index = np.floor((samples - total_indexes)/2).astype(int)
    mean_max_index = np.floor((samples + total_indexes)/2).astype(int)
    mean_min_t = mean_min_index * dt + t_min # [us]
    mean_max_t = mean_max_index * dt + t_min # [us]

    # compute mean, discarding points outside a COMPLETE wave period
    DC_offset = wave_data[:,1][mean_min_index:mean_max_index].mean() 

    # apply DC offset and scale multiplier to the simulated wave
    simulated_wave = scale * simulated_wave + DC_offset

    # test quality of results: chi2 assuming error of 1 digit
    chi2 = ((simulated_wave - wave_data[:,1])**2).sum()
    print(f"chi2 = {chi2:.1f}; chi2/N = {chi2/samples:.1f}")

    # plot results
    plt.style.use(["science"])
    plt.figure(figsize=(3.2, 2), dpi=320)
    plt.errorbar(wave_data[:,0], wave_data[:,1], xerr=dt, yerr=1, color="red", fmt='.', markersize=2, zorder=1) # plot data
    plt.plot(t, simulated_wave, color="blue", alpha=0.8, zorder=2) # plot simulated wave

    plt.title(f"$f$ = ${f}\pm{f_err}$ Hz, {samples} pts, $\Delta t = {dt:.1f} \mu s$")
    plt.xlabel("Tempo [$\mu s$]")
    plt.ylabel("$V$ [digit]")

    # color area that was discarded in the DC offset calculation
    plt.axvspan(0, mean_min_t, alpha=0.2, color="black", lw=0)
    plt.axvspan(mean_max_t, t_max, alpha=0.2, color="black", lw=0)
    plt.axhline(DC_offset, color="blue", lw=0.5, zorder=0)

    if N_complete_periods >= 10:
        # restrict graph window if record is too long
        plt.xlim(50000, 60000)
    else:
        plt.xlim(t_min, t_max)
    
    plt.savefig("RC_integrator/" + filename + ".png")
    plt.close()