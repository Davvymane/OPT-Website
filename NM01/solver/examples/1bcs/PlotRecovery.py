import numpy as np
import matplotlib.pyplot as plt

def plot_recovery(xo, x, pos=(100, 100, 800, 400), ind=True):

    xo = np.asarray(xo).flatten()
    x = np.asarray(x).flatten()
    n = len(x)

    fig = plt.figure(figsize=(pos[2]/100, pos[3]/100))  
    fig.canvas.manager.set_window_title('Plot Recovery')
    ax = fig.add_axes([0.05, 0.1, 0.9, 0.8])

    ax.stem(np.where(xo != 0)[0], xo[xo != 0],
            linefmt='o-', markerfmt='o', basefmt=' ', label='Ground-Truth',
            use_line_collection=True)
    
    ax.stem(np.where(x != 0)[0], x[x != 0],
            linefmt='o:', markerfmt='o', basefmt=' ', label='Recovered',
            use_line_collection=True)

    for stemline in ax.stem(np.where(xo != 0)[0], xo[xo != 0])[1]:
        stemline.set_color('#f26419')
        stemline.set_linewidth(1)
    for marker in ax.stem(np.where(xo != 0)[0], xo[xo != 0])[0]:
        marker.set_markersize(7)

    for stemline in ax.stem(np.where(x != 0)[0], x[x != 0])[1]:
        stemline.set_color('#1c8ddb')
        stemline.set_linewidth(1)
    for marker in ax.stem(np.where(x != 0)[0], x[x != 0])[0]:
        marker.set_markersize(4)

    xx = np.concatenate((xo, x))
    ymin = min(np.min(xx[xx < 0]) - 0.1 if np.any(xx < 0) else -0.1, -0.1)
    ymax = max(np.max(xx[xx > 0]) + 0.1 if np.any(xx > 0) else 0.2, 0.2)
    ax.set_xlim([1, n])
    ax.set_ylim([ymin, ymax])
    ax.grid(True)

    if ind:
        snr = -10 * np.log10(np.linalg.norm(x - xo)**2)
        support_mismatch = np.count_nonzero((x != 0) & (xo == 0)) + \
                           np.count_nonzero((x == 0) & (xo != 0))

        title = f"SNR={snr:.4f}, Number of mis-supports = {support_mismatch}"
        ax.set_title(title, fontweight='normal')
        ax.legend()

    plt.show()