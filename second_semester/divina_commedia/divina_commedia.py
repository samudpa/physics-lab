import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm, binom

with open("data/Divina_Commedia.json", "r", encoding="utf-8") as f:
    data = json.loads(f.read())  # see parse_txt.py

# create an array containing the length of every verse
l = []
for cantica in data:
    for canto in data[cantica]:
        for verse in data[cantica][canto]:
            l.append(len(verse))

l = np.array(l)
N = len(l)
mu = l.mean()
std = l.std(ddof=1)

print(f"Numero di versi: {N}")
print(f"Lunghezza media: {mu:.2f}")
print(f"Deviazione standard: {std:.2f}")

# histogram bins
bins = np.arange(l.min() - 0.5, l.max() + 1.5)

# plot histogram
plt.style.use(["science", "ieee"])
plt.figure(figsize=(3.3, 2.5), dpi=600)
o, bins, _ = plt.hist(
    l, bins=bins, rwidth=0.25, color="#004488", label="Conteggi", zorder=2
)

# grid, label, legend
plt.grid(which="major", axis="y", color="lightgray", zorder=0)
# plt.title("Lunghezza dei versi")
plt.xlabel("Numero di caratteri per verso")
plt.ylabel("Occorrenze")
plt.xlim((23, 50))

# expected values from Poissonian and Gaussian models
k = np.arange(l.min(), l.max() + 1)
e_poisson = N * poisson.pmf(k, mu)
e_gauss = N * norm.pdf(k, mu, std)

# plot poisson and gauss distributions
plt.bar(k - 0.3, e_poisson, width=0.25, color="#DDAA33", label="Poisson", zorder=2)
x = np.linspace(l.min() - 3, l.max() + 3, 200)
e_gauss_smooth = N * norm.pdf(x, mu, std)
plt.plot(x, e_gauss_smooth, color="#BB5566", label="Gauss", zorder=3)

# find chi2 values for both distributions
chi2_poisson = ((o - e_poisson) ** 2 / e_poisson).sum()
chi2_gauss = ((o - e_gauss) ** 2 / e_gauss).sum()
ni_poisson = len(k) - 2
ni_gauss = len(k) - 3
print(f"chi2 (Poisson) = {chi2_poisson:.1f}/{ni_poisson}")
print(f"chi2 (Gauss) = {chi2_gauss:.1f}/{ni_gauss}")

plt.legend()
plt.savefig("graphs/verse_length_hist.png")
