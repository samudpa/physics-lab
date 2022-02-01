import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec

def plot_time_height(
    title, filename,
    data, model,
    xlim, ylim_fit, ylim_res):

    t, h, h_err, res, popt = data # unpack data

    # initialize plot
    plt.style.use(['science', 'grid'])
    fig = plt.figure(figsize=(5,4))
    gs = gridspec.GridSpec(2,1, height_ratios=[3, 1])
    plt.subplots_adjust(hspace=0)
    plt.tight_layout()

    color = 'black'
    alpha = 0.3
    capsize = 1.5

    # best-fit axis
    ax_fit = fig.add_subplot(gs[0])

    ax_fit.errorbar(t, h, yerr=h_err, color=color, fmt='.', capsize=capsize, label='Dati', zorder=2)

    t_fit = np.linspace(0,0.6,100)
    ax_fit.plot(t_fit, model(t_fit, *popt), color=color, alpha=alpha, label='Modello', zorder=1)
    ax_fit.axhline(0, color='black', ls='--', alpha=alpha, zorder=1) # ground line (y=0)

    # residuals axis
    ax_res = fig.add_subplot(gs[1], sharex=ax_fit)

    ax_res.errorbar(t, res, yerr=h_err,
            color=color, fmt='.', capsize=capsize, zorder=2) # residuals
    ax_res.axhline(0, color='black', ls='--', alpha=alpha, zorder=1) # model (horizontal line)
    ax_res.vlines(t, 0, res, color=color, alpha=alpha) # model to data lines

    # limits
    ax_res.set_xlim(xlim)
    ax_res.set_ylim(ylim_res)
    ax_fit.set_ylim(ylim_fit)

    # setup grids
    ax_fit.grid(which='both', ls='dashed', color='lightgray', zorder=0)
    ax_res.grid(which='both', ls='dashed', color='lightgray', zorder=0)

    # labels and legend
    ax_fit.legend()
    ax_fit.set_title(title)
    ax_fit.set_ylabel('Altezza [m]')
    ax_res.set_ylabel('Residui [m]')
    ax_res.set_xlabel('Tempo [s]')
    plt.setp(ax_fit.get_xticklabels(), visible=False)

    if filename is not None: plt.savefig(filename)
    plt.show()