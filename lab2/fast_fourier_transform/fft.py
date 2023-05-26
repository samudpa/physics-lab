import os
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib as mpl
import matplotlib.ticker
import scienceplots

# https://stackoverflow.com/a/44079725
locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)

# load csv containing info about data
data = np.genfromtxt(
    "./data/data.csv", delimiter=",", names=True, dtype=None, encoding="utf-8"
)

def fmt_measure(value, err, sig=2, sep=" \pm "):
    """Returns a formatted string of (value, err)
    with the error rounded to `sig` significant figures"""

    decimals = int(sig - np.floor(np.log10(np.abs(err))) - 1)
    if decimals >= 0:
        formatter = "{:." + str(decimals) + "f}"
        return formatter.format(value) + sep + formatter.format(err)
    if decimals < 0:
        return f"{int(np.round(value, decimals))}{sep}{int(np.round(err, decimals))}"

class WaveEntry:
    _BASE_TIME_ERR = 10  # [us]
    _BASE_VALUE_ERR = 10  # [digit]

    path: str
    basename: str
    filename: str
    data: np.ndarray
    dataentry: np.ndarray
    N: int
    cycles: int

    # time variables
    dt: float
    t: np.ndarray[float]
    terr: np.ndarray[float]

    # voltage variables
    values: np.ndarray[float]  # [digit]s
    errs: np.ndarray[float]  # [digit]s

    # frequency variables
    f: np.ndarray[float]
    samplerate: float
    spectrum: np.ndarray[float]  # complex valued
    spectrum_errs: np.ndarray[float]  # complex valued
    basefreq: float
    basefreq_amplitude: float

    # plot variables
    title: str
    fig: Figure
    axes: list[Axes]
    t_min: float
    t_max: float
    f_min: float
    f_max: float
    fft_min: float | None
    show_closeup: bool

    # RLC oscillator variables
    Qf: float
    Qf_err: float

    def __init__(self, entry_data):
        folder = entry_data["folder"]
        filename = entry_data["filename"]

        self.filename = os.path.join(folder, filename)
        self.basename = os.path.splitext(os.path.basename(filename))[0]
        self.path = os.path.join("./data", self.filename)

        print(self.filename)

        if not os.path.isfile(self.path):
            raise Exception(f"{self.path} doesn't exist or is not a file.")

        self.data = np.loadtxt(self.path)
        N, columns = self.data.shape
        self.N = N

        if columns == 4:
            t, t_err, V, V_err = self.data.T
        else:
            assert columns == 2
            t, V = self.data.T
            t_err = np.full(t.shape, self._BASE_TIME_ERR)
            V_err = np.full(V.shape, self._BASE_VALUE_ERR)

        self.cycles = entry_data["cycles"]

        # time
        self.t = t * 1e-6
        self.terr = t_err * 1e-6
        self.dt = np.average(np.diff(self.t))
        self.samplerate = 1 / self.dt
        # values
        self.values = V
        self.errs = V_err
        # plot
        self.dataentry = entry_data
        self.title = self._loadkey("custom_title", self.basename)
        self.t_min = self._loadkey("t_min", 0)
        self.t_max = self._loadkey("t_max", self.t.max())
        self.f_min = self._loadkey("f_min", 0)
        self.f_max = self._loadkey("f_max", self.samplerate / 2)
        self.fft_min = self._loadkey("fft_min", None)
        self.show_closeup = self._loadkey("show_closeup", True)

    def _loadkey(self, key, default):
        data = self.dataentry
        if np.isnan(data[key]) or data[key] == None:
            return default
        else:
            return data[key]

    def do_fft(self):
        """Compute Fast Fourier Transform on data"""

        # compute fast fourier transform
        self.f = fftfreq(self.N, self.dt)[: self.N // 2]
        self.spectrum = fft(self.values)[: self.N // 2]
        self.spectrum_errs = fft(self.errs)[: self.N // 2]

        basefreq_index = np.argmax(np.abs(self.spectrum)[1:]) + 1
        self.basefreq = self.f[basefreq_index]
        self.basefreq_amplitude = 1 / self.N * np.abs(self.spectrum)[basefreq_index]
        print(f"\tf1 = {self.basefreq:.3g} Hz")

    def do_graph(self):
        """Graph the output"""

        fig = Figure(figsize=(3.6, 3), dpi=320)
        fig.set_tight_layout(True)
        fig.set_tight_layout(dict(w_pad=0.6, h_pad=0.3))
        # fig.suptitle(self.title)
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 3], width_ratios=[3, 1])

        # FULL time plot
        if self.show_closeup:
            t_ax = fig.add_subplot(gs[0, 0])
        else:
            t_ax = fig.add_subplot(gs[0, :])
        t_ax.minorticks_on()
        t_ax.errorbar(
            x=self.t * 1e3,
            y=self.values,
            xerr=self.terr,
            yerr=self.errs,
            fmt=".",
            ms=0.4,
            color="black",
        )
        t_ax.plot(
            self.t * 1e3,
            self.values,
            linewidth=0.7,
            alpha=0.5,
            color="black",
        )
        # limits
        t_ax.set_xlim(self.t_min * 1e3, self.t_max * 1e3)
        # labels
        t_ax.set_xlabel("$t$ [ms]")
        t_ax.set_ylabel("$V(t)$ [digit]")
        # grid
        t_ax.grid(which="major", axis="both", color="gray", zorder=0)

        # ZOOM time plot
        if self.show_closeup:
            tzoom_ax = fig.add_subplot(gs[0, 1], sharey=t_ax)
            tzoom_ax.minorticks_on()
            tzoom_ax.errorbar(
                x=self.t * 1e3,
                y=self.values,
                xerr=self.terr,
                yerr=self.errs,
                fmt=".",
                ms=0.4,
                color="black",
            )
            tzoom_ax.plot(
                self.t * 1e3,
                self.values,
                linewidth=0.7,
                alpha=0.5,
                color="black",
            )
            # limits
            halft = self.t_max / 2
            period = 1 / self.basefreq
            tzoom_ax.set_xlim((halft - period) * 1e3, (halft + period) * 1e3)
            # labels
            tzoom_ax.tick_params(labelleft=False)
            tzoom_ax.set_xlabel("$t$ [ms]")
            # grid
            tzoom_ax.grid(which="major", axis="both", color="gray", zorder=0)

        # frequency spectrum
        basefreq_err = self.samplerate/self.N
        f_ax = fig.add_subplot(gs[1, :])
        f_ax.plot(
            self.f * 1e-3,
            np.abs(self.spectrum) * 1 / self.N,
            linewidth=0.7,
            color="black",
            zorder=2,
        )
        f_ax.axvline(
            self.basefreq * 1e-3,
            color="blue",
            label=f"$f_1 = {fmt_measure(self.basefreq, basefreq_err)}$ Hz",
            linewidth=0.7,
            zorder=1,
        )

        # limits
        f_ax.set_yscale("log")
        f_ax.set_xlim(self.f_min * 1e-3, self.f_max * 1e-3)
        f_ax.set_ylim(bottom=self.fft_min)
        # labels
        f_ax.set_xlabel("$f$ [kHz]")
        f_ax.set_ylabel("$\\tilde{V}(f)$ [u.a.]")
        # grid
        f_ax.yaxis.set_major_locator(locmaj)
        f_ax.yaxis.set_minor_locator(locmin)
        f_ax.grid(which="major", axis="both", color="gray", zorder=0)

        self.fig = fig
        self.axes = [t_ax, f_ax]
        if self.show_closeup:
            self.axes.append(tzoom_ax)

        if self._loadkey("graph_harmonics", False):
            self.graph_harmonics(30)

        power = self._loadkey("power", False)
        if power:
            self.graph_one_over(power)  # graph 1 over f to the power
        rcfilter_cutoff = self._loadkey("RC_int_cutoff", False)
        if rcfilter_cutoff:
            self.graph_RC_filter(rcfilter_cutoff, power)

        if self._loadkey("RLC_oscillator", False):
            self.rlc_oscillator()

    def graph_one_over(self, power):
        scale = self.basefreq_amplitude
        f_ax = self.axes[1]
        if power != 1:
            label = f"^{power:.0f}"
        else:
            label = ""

        def model(f):
            return scale * (self.basefreq / f) ** power

        f_ax.plot(
            self.f[1:] * 1e-3,
            model(self.f[1:]),
            color="#2c9e2c",
            label="$\sim 1/f" + label + "$",
        )

    def graph_RC_filter(self, cutoff, power):
        scale = self.basefreq_amplitude
        f_ax = self.axes[1]

        def model(f):
            return (
                scale
                * (self.basefreq / f) ** power
                * 1
                / np.sqrt(1 + (f / cutoff) ** 2)
            )

        f_ax.plot(
            self.f[1:] * 1e-3, model(self.f[1:]), color="#fc9732", label="filtro RC"
        )

    def graph_harmonics(self, N):
        color_odd = "blue"  # "#004488"
        color_even = "red"  # "#BB5566"

        for n in range(2, N + 1):
            alpha = 0.4 * (1 - n / (N + 1))
            f_ax = self.axes[1]

            if n % 2 == 0:
                color = color_even
            else:
                color = color_odd

            f_ax.axvline(
                n * self.basefreq * 1e-3,
                color=color,
                linewidth=0.7,
                zorder=1,
                alpha=alpha,
            )

    def rlc_oscillator(self):
        """Compute RLC oscillator parameters"""

        omega0 = self.dataentry["RLC_omega"]
        omega0_err = self.dataentry["RLC_omega_err"]
        tau = self.dataentry["RLC_tau"] *1e-3
        tau_err = self.dataentry["RLC_tau_err"] *1e-3

        Qf = omega0 * tau/2
        Qf_err = Qf * (omega0_err/omega0 + tau_err/tau)

        self.Qf = Qf
        self.Qf_err = Qf_err
        print(f"\tQf = {fmt_measure(Qf, Qf_err)}")

    def save_graph(self, folder):
        """Save the graph into folder"""

        # legend
        # https://stackoverflow.com/a/25540279
        legend = self.axes[1].legend(
            fontsize=8, loc="upper right", framealpha=0.7, labelspacing=0.2
        )
        legend.get_frame().set_linewidth(0)

        graphfolder = os.path.join(folder, os.path.dirname(self.filename))
        if not os.path.isdir(graphfolder):
            os.makedirs(graphfolder)

        self.fig.savefig(os.path.join(graphfolder, self.basename + ".png"))
        plt.close(self.fig)

Qf_plot_wavs = []

# main loop starts here
for data_entry in data:
    wav = WaveEntry(data_entry)
    wav.do_fft()

    plt.style.use(["science", "grid"])
    wav.do_graph()
    wav.save_graph("./graphs")
    
    if wav._loadkey("include_in_Qf_plot", False):
        Qf_plot_wavs.append(wav)

# quality factor comparison plot
Qf_plot_wavs.sort(key=lambda x: x.Qf)
Qf_max = Qf_plot_wavs[-1].Qf
Qf_min = Qf_plot_wavs[0].Qf

cmap = mpl.colormaps["plasma"] # https://matplotlib.org/stable/tutorials/colors/colormaps.html
norm = mpl.colors.Normalize(vmin=Qf_min, vmax=Qf_max)
def Qf_cmap(Qf):
    x = (Qf - Qf_min)/(Qf_max - Qf_min)
    return cmap(x)

plt.close()
fig = plt.figure(figsize=(3.6, 3), dpi=320)
ax = fig.add_subplot(111)

ax.set_yscale("log")
ax.set_ylim(1e-2, 3)
ax.set_xlim(0.25,1.75)

for wav in Qf_plot_wavs:
    
    fs = wav.f[1:]
    spectrum = np.abs(wav.spectrum)[1:]/wav.N
    xscale = 1/wav.basefreq
    yscale = 1/wav.basefreq_amplitude
    fs *= xscale
    spectrum *= yscale

    Qf = wav.Qf
    Qf_err = wav.Qf_err

    ax.plot(fs, spectrum, color=Qf_cmap(Qf), label=f"Qf = ${fmt_measure(Qf, Qf_err)}$", linewidth=0.7, alpha=0.8)

ax.set_ylabel("Ampiezza [u.a.]")
ax.set_xlabel("Frequenza [u.a.]")

cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
cbar.set_label("Quality factor", rotation=270, labelpad=15)
plt.savefig("./graphs/qf.png")