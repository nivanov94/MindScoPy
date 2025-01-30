import mne

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