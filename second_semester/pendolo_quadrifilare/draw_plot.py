import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_period_time(t, T, filename):
    '''Plot pendulum period (T) versus time (t)'''

    MARKERSIZE = 0.5

    # setup plot
    plt.style.use(['science'])
    plt.figure(figsize=(4,3))

    # plot
    plt.errorbar(t, T, fmt='.', ms=MARKERSIZE, color='black')

    # title and labels
    plt.title('Periodo $T$ di un pendolo smorzato')
    plt.xlabel('$t$ [s]')
    plt.ylabel('Periodo [s]')
    plt.grid(which='both', ls='dashed', color='lightgray')

    plt.savefig(filename)

def plot_v0_time(t, v0, popt, model, filename):
    '''Plot velocity at the lowest point of the trajectory (v0) versus time (t)'''

    MARKERSIZE = 0.1

    # setup plot
    plt.style.use(['science'])
    fig = plt.figure(figsize=(4.5,3.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    # axis and grids
    ax_data = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_data)
    ax_data.grid(which='both', ls='dashed', color='lightgray')
    ax_res.grid(which='both', ls='dashed', color='lightgray')
    ax_data.set_axisbelow(True)
    ax_res.set_axisbelow(True)

    # data axis
    ax_data.errorbar(t, v0, fmt='.', ms=MARKERSIZE, color='black', alpha=0.5, label='Dati', zorder=2)
    ax_data.plot(t, model(t, *popt), color='tab:blue', alpha=0.8, label='Modello', zorder=1)
    
    # residuals axis
    res = v0 - model(t, *popt)
    ax_res.errorbar(t, res, fmt='.', ms=MARKERSIZE, color='black', alpha=0.5, zorder=2)
    ax_res.axhline(0, ls='--', color='tab:blue', alpha=0.8, zorder=1)

    # limits
    ax_res.set_ylim((-0.013, 0.013))

    # title and labels
    ax_data.set_title('Velocità $v_0$ nel punto più basso')
    ax_data.set_ylabel('$v_0$ [m/s]')
    ax_res.set_ylabel('Residui [m/s]')
    ax_res.set_xlabel('$t$ [s]')
    ax_data.legend()

    # remove vertical space between plots
    plt.setp(ax_data.get_xticklabels(), visible=False)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0)

    plt.savefig(filename)

def plot_period_theta0(theta0, T_theta0, l, popt, model, filename):
    '''Plot pendulum period (T) versus angular amplitude (theta0) '''

    MARKERSIZE = 0.1

    # setup plot
    plt.style.use(['science'])
    fig = plt.figure(figsize=(5,4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 2])

    # axis and grids
    ax_data = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_data)
    ax_data.grid(which='both', ls='dashed', color='lightgray')
    ax_res.grid(which='both', ls='dashed', color='lightgray')
    ax_data.set_axisbelow(True)
    ax_res.set_axisbelow(True)

    # data axis
    theta0_lin = np.linspace(0,0.26,1000)
    ax_data.errorbar(theta0, T_theta0, fmt='.', ms=MARKERSIZE, color='black', alpha=0.5, label='Dati', zorder=1)
    ax_data.plot(theta0_lin, model(theta0_lin, *popt), color='tab:blue', alpha=0.8, label='Modello con valore \emph{best-fit} di $l$', zorder=2)
    ax_data.plot(theta0_lin, model(theta0_lin, l), color='tab:orange', alpha=0.8, label='Modello con valore misurato di $l$', zorder=2)

    # residuals axis
    res_model = T_theta0 - model(theta0, *popt)
    res_expected = T_theta0 - model(theta0, l)
    ax_res.errorbar(theta0, res_model, fmt='.', ms=MARKERSIZE, color='tab:blue', alpha=0.5, zorder=2)
    ax_res.errorbar(theta0, res_expected, fmt='.', ms=MARKERSIZE, color='tab:orange', alpha=0.5, zorder=2)
    ax_res.axhline(0, color='black', alpha=0.8, zorder=1)

    # limits
    ax_data.set_ylim((2.1210, 2.1360))
    ax_res.set_ylim((-0.004, 0.0027))
    ax_res.set_xlim((0,0.26))

    # title and labels
    ax_data.set_title(r"Periodo $T$ in funzione dell'ampiezza $\theta_0$")
    ax_data.set_ylabel('Periodo [s]')
    ax_res.set_ylabel('Residui [s]')
    ax_res.set_xlabel(r'$\theta_0$ [rad]')
    ax_data.legend()

    # remove vertical space between plots
    plt.setp(ax_data.get_xticklabels(), visible=False)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0)

    plt.savefig(filename)