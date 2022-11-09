import os
import numpy as np
import matplotlib.pyplot as plt

# get list of files in current directory
# https://stackoverflow.com/a/3964691
filelist = []
for file in os.listdir("."):
    if file.endswith(".txt"):
        filelist.append(file)
print(filelist)

for file in filelist:

    print(file)
    t, V = np.loadtxt(file, unpack=True)
    N = len(t)
    difft = np.diff(t) 
    dt = np.mean(difft)
    dtnom = int(np.floor(dt - 6)) 
    stdevt = np.std(difft)
    print(N, dt, stdevt)

    plt.figure(figsize=(12,3), dpi=400)
    plt.plot(t, V, color="black", linewidth=0.7)
    plt.xlim(0, t[-1])
    title = f"{file}, $N$ = {N}, ${{\Delta t}}_{{nom}}$ = {dtnom} $\mu s$, $\Delta t$ = {dt:.2f} $\mu s$, stdev = {stdevt:.2f} $\mu s$"
    plt.title(title)
    plt.xlabel("$t$ [$\mu s$]")
    plt.ylabel("$V$ [digit]")
    plt.savefig(file + ".png",bbox_inches='tight')