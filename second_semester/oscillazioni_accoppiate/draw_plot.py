import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec

GAP = 5
ALPHA = 0.4
MARKERSIZE = 4

def draw_plot(
    data_dicts,     # list of dictionaries containing time and position data
    title,          # plot title
    models=None,    # best-fit models ...
    popts=None,     # ... and parameters
    limits=None,    # graph limits (xlim, ylim ...)
    labels=None,    # data and model labels
    colors=None,    # list of plot colors
    filename=None,  # save graph with this filename
    show=True,      # show interactive plot after saving
):

    # setup plot
    plt.style.use(['science', 'grid'])
    if models:
        fig = plt.figure(figsize=(4, 3.5))
    else:
        fig = plt.figure(figsize=(4, 2.3))

    # setup default parameters
    n = len(data_dicts) # number of data_dicts
    
    if limits == None:
        limits = {'xlim': None, 'ylim_data': None, 'ylim_res': None}

    if labels == None:
        labels = [{'data': 'Dati', 'model': 'Modello'}] * n

    if colors == None:
        colors = ['black'] * n

    if models == None:

        # if there are no models, create a gridspec with only 1 cell
        models = [None] * n
        popts = [None] * n
        ax_res = None
        gs = gridspec.GridSpec(1, 1)
        ax_data = fig.add_subplot(gs[0])

    else:

        # if there is a model, create a gridspec with two axes (data and residuals)
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax_data = fig.add_subplot(gs[0])
        ax_res = fig.add_subplot(gs[1], sharex=ax_data)

    # plot data, best-fit models and residuals
    # iterate over waves
    for data, label, color, model, popt in zip(
        data_dicts, labels, colors, models, popts
    ):

        # plot data
        ax_data.errorbar(
            data['t'], data['pos'],
            xerr=data['t_err'],
            yerr=data['pos_err'],
            fmt='.',
            ms=MARKERSIZE,
            color=color,
            label=label['data'],
            zorder=2,
        )

        if model:

            # plot model
            t_lin = np.linspace(min(data['t']) - GAP, max(data['t']) + GAP, 1000)
            ax_data.plot(
                t_lin,
                model(t_lin, *popt),
                alpha=ALPHA,
                color=color,
                label=label['model'],
                zorder=1,
            )

            # plot residuals
            res = data['pos'] - model(data['t'], *popt)
            ax_res.errorbar(
                data['t'],
                res,
                xerr=data['t_err'],
                yerr=data['pos_err'],
                fmt='.',
                ms=MARKERSIZE,
                color=color,
                zorder=2,
            )

        else:

            # connect data points
            ax_data.plot(
                data['t'],
                data['pos'],
                ls='-',
                color=color,
                alpha=ALPHA,
            )

            ax_data.axhline(
                np.mean(data['pos']),
                linewidth=1,
                ls='--',
                color=color,
                alpha=ALPHA,
                zorder=1
            )

    # add residuals horizontal lines
    if ax_res:
        ax_res.axhline(
            0, ls='--', color='black', alpha=ALPHA, zorder=1
        )

    # limits
    ax_data.set_ylim(limits['ylim_data'])
    if ax_res:
        ax_res.set_xlim(limits['xlim'])
        ax_res.set_ylim(limits['ylim_res'])
    else:
        ax_data.set_xlim(limits['xlim'])

    # grids
    ax_data.grid(which='both', ls='dashed', color='lightgray', zorder=0)
    if ax_res:
        ax_res.grid(which='both', ls='dashed', color='lightgray', zorder=0)

    # labels and legend
    if model:
        ax_data.legend()
    ax_data.set_title(title)
    ax_data.set_ylabel('Posizione [au]')
    if ax_res:
        ax_res.set_ylabel('Residui [au]')
        ax_res.set_xlabel('Tempo [s]')
        plt.setp(ax_data.get_xticklabels(), visible=False)
    else:
        ax_data.set_xlabel('Tempo [s]')

    # remove vertical space between residuals and plot
    fig.tight_layout()
    plt.subplots_adjust(hspace=0)

    # show plot and save graph
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()

    plt.close()