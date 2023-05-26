# version 20230509
# changelog:
#   20230509 - generate plots for waves averaged over Ncycles cycles

import os
import numpy as np
import matplotlib.pyplot as plt

# get list of directories
# https://stackoverflow.com/a/973488
dirlist = [x[0] for x in os.walk(".")]
dirlist = [f.path for f in os.scandir("./") if f.is_dir()]
dirlist.append("./")

# get list of files in directory
# https://stackoverflow.com/a/3964691
filelist = []
for dir in dirlist:
    for file in os.listdir(dir):
        if file.endswith(".txt"):
            filelist.append(f"{dir}/{file}")
print(filelist)

for file in filelist:

    print(file)
    data = np.loadtxt(file, unpack=True)
    if len(data) == 2:
        t, V = data
        t_std = np.zeros(t.shape)
        V_std = np.zeros(V.shape)
    elif len(data) == 4:
        t, t_std, V, V_std = data
    else:
        exit()

    N = len(t)
    difft = np.diff(t) 
    dt = np.mean(difft)
    dtnom = int(np.floor(dt - 6)) 
    stdevt = np.std(difft)
    # guess the number of avg cycles
    cycles = (2,4,8,16)
    diff = np.abs(0.65 - stdevt*np.sqrt(cycles))
    Ncycles = cycles[np.argmin(diff)]
    if len(data) == 2: Ncycles = 1

    print(N, dt, stdevt, Ncycles)

    plt.figure(figsize=(12,3), dpi=400)
    plt.errorbar(t, V, xerr=t_std, yerr=V_std, color="black", ecolor="red", linewidth=0.7)
    plt.xlim(0, t[-1])
    title = f"{os.path.basename(file)}, $N_{{pts}}$ = {N}, $N_{{cycles}}$ = {Ncycles}, ${{\Delta t}}_{{nom}}$ = {dtnom} $\mu s$, $\Delta t$ = {dt:.2f} $\mu s$, $\sigma_t$ = {stdevt:.2f} $\mu s$"
    plt.title(title)
    plt.xlabel("$t$ [$\mu s$]")
    plt.ylabel("$V$ [digit]")
    plt.savefig(file + ".png",bbox_inches='tight')
    plt.close()