import mne
import numpy as np

# To Do: make this function more modular

def visualize_CSP(W, chs, vmin, vmax, fig, ax, width, cbar=False, montage_type='standard_1020', ch_type='eeg'):
    info = mne.create_info(ch_names=chs, sfreq=1., ch_types=ch_type)
    montage = mne.channels.make_standard_montage(montage_type)
    info.set_montage(montage)
    
    im, cm = mne.viz.plot_topomap(W[:,0]-W[:,-1], info, vlim=(vmin,vmax), show=False, axes=ax, cmap='RdBu')
    
    if cbar:
        # manually fiddle the position of colorbar
        ax_x_start = 0.95
        ax_x_width = 0.1 / width
        ax_y_start = 0.28
        ax_y_height = 0.5
        cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
        clb = fig.colorbar(im, cax=cbar_ax)
        clb.ax.set_title('AU',fontsize=18) # title on top of colorbar
        clb.ax.tick_params(labelsize=18)

    fig.canvas.draw()
    fig.canvas.flush_events()


def plot_pmfs(Ps, ax, legend_str='Distribution', x_label='State', y_label='Probability'):
    """
    Plot discete probability mass function (PMF) as a step function.

    Parameters
    ----------
    Ps : array_like (Nd, Ns,)
        The probability mass function(s). The first dimension represents
        the number of distributions and the second dimension represents the
        number of states.
    
    ax : matplotlib.axes.Axes
        The axis object to plot the PMF.

    legend_str : str
        The string to pre-append to the legend labels for each distribution.

    x_label : str
        String to apply to the x-axis label.

    y_label : str
        String to apply to the y-axis label.

    Returns
    -------
    None
    """
    Nd, Ns = Ps.shape
    x = np.arange(Ns+1) # add one to the number of states for the step plot
    for i in range(Nd):
        pmf_pts = np.concatenate((Ps[i], [0])) # add zero to the end for the step plot
        ax.step(x, np.concatenate((Ps[i], [0])), where='post') # line
        ax.fill_between(
            x, 
            pmf_pts, 
            step='post', 
            alpha=0.3, 
            label=f'{legend_str} {i+1}'
        ) # fill

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim([0, Ns])
    ax.set_ylim([0, 1])
    ax.set_xticks(np.arange(Ns)+0.5)
    ax.set_xticklabels(np.arange(Ns)+1)
    ax.legend()


def show_transition_matrices(A, ax, title=None, x_label='Origin', y_label='Destination'):
    """
    Plot transition matrices as a heatmap.

    Parameters
    ----------
    A : array_like (Ns, Ns)
        The transition matrix(s) to plot. The first and second dimensions
        represent the number of states.

    ax : matplotlib.axes.Axes
        The axis object to plot the transition matrix.

    title : str
        The title to apply to the plot.

    x_label : str
        String to apply to the x-axis label.

    y_label : str
        String to apply to the y-axis label.

    Returns
    -------
    None
    """
    Ns = A.shape[0]

    ax.imshow(A, cmap='Greens', vmin=0, vmax=1)

    # add the probabilities to the heatmap
    for i in range(Ns):
        for j in range(Ns):
            ax.text(j, i, f'{A[i, j]:.2f}', ha='center', va='center', color='black')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)

    

