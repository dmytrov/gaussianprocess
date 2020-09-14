import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as plp
import pylab


def save_plot_matrix(directory, prefix, A=np.identity(5), xticks=None, yticks=None, xlabel=None, ylabel=None):
    if xticks is None:
        xticks = [0, A.shape[1]]
    if yticks is None:
        yticks = [0, A.shape[0]]
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(A, origin='lower', interpolation='nearest',
               extent=(xticks[0]+0.5, xticks[-1]+0.5, yticks[-1]+0.5, yticks[0]+0.5),
               cmap=plt.jet())
    #plt.gca().xaxis.tick_top()
    plt.colorbar()
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.savefig("{}/{}.pdf".format(directory, prefix))
    plt.close(fig)

def plot_matrix(A=np.identity(5), xticks=None, yticks=None):
    if xticks is None:
        xticks = [0, A.shape[1]]
    if yticks is None:
        yticks = [0, A.shape[0]]
    plt.gcf()
    plt.imshow(A, interpolation='nearest',
               extent=(xticks[0]+0.5, xticks[-1]+0.5, yticks[-1]+0.5, yticks[0]+0.5),
               cmap=plt.jet())
    plt.gca().xaxis.tick_top()
    plt.colorbar()
    plt.show()

def plot_sequences(x, title=None):
    plt.plot(x)
    if title is not None:
        plt.title(title)
    plt.show()

def plot_sequence2d(X=np.identity(2), title=None):
    plt.plot(X[:, 0], X[:, 1], '-o')
    if title is not None:
        plt.title(title)
    plt.show()

def plot_variance2d(S=np.identity(5), extent=(0, 1, 0, 1)):
    plt.imshow(S, extent=extent, cmap="summer", origin="lower")

def show():
    plt.show()



def plot_sequence_variance_2d(X=np.identity(2), predictor=None):
    xMin = X.min(axis=0)
    xMax = X.max(axis=0)
    xRange = xMax[0] - xMin[0]
    yRange = xMax[1] - xMin[1]
    xExtentMin, xExtentMax = xMin[0] - 0.5*xRange, xMax[0] + 0.5*xRange
    yExtentMin, yExtentMax = xMin[1] - 0.5*yRange, xMax[1] + 0.5*yRange
    xticks = np.arange(xExtentMin, xExtentMax, 0.1)
    yticks = np.arange(yExtentMin, yExtentMax, 0.1)
    xpoints, ypoints = pylab.meshgrid(xticks, yticks)
    xypoints = np.vstack([xpoints.ravel(), ypoints.ravel()]).T

    yStar, covStar = predictor(xypoints)
    covStar = np.reshape(covStar, xpoints.shape)

    plot_variance2d(covStar, (xExtentMin, xExtentMax, yExtentMin, yExtentMax))
    plot_sequence2d(X)
    show()

def plot_gaussians_mean_diag_covar_2d(x_means, x_covars_diags, color="b"):
    """
    x_means : [N, 2]
    x_covars_diags : [N, 2]
    """
    ax = plt.gca()
    N = x_means.shape[0]
    std_width = 2.0 * np.sqrt(x_covars_diags[:, 0])
    std_height = 2.0 * np.sqrt(x_covars_diags[:, 1])
    for i in range(N):
        ellipse = plp.Ellipse(xy=x_means[i], width=std_width[i], height=std_height[i])
        ellipse.set_alpha(0.2)
        ellipse.set_color(color)
        ax.add_artist(ellipse)


def plot_arrows_1(tail, head=None, width=0.005, color="b", alpha=0.5):
    """
    Plot a path with arrows
    tail : [N, 2] from
    head : [N, 2] to
    """
    if head is None:
        assert tail.shape[-1] == 4
        tail, head = tail[:, 0:2], tail[:, 2:4]
    assert tail.shape == head.shape
    assert tail.shape[-1] == 2
    N = tail.shape[0]
    xdiff = head-tail
    for i in range(N):
        plt.arrow(tail[i, 0], tail[i, 1], xdiff[i, 0], xdiff[i, 1], color=color, alpha=alpha, width=width)


def plot_arrows_2(tail, head=None, width=0.005, color="b", alpha=0.5):
    """Plot arrows from head to tail.
        tail : [N, 2] or [N, 4] 
        head : [N, 2] or None 
    """
    if head is None:
        assert tail.shape[-1] == 4
        tail, head = tail[:, 0:2], tail[:, 2:4]
    assert tail.shape == head.shape
    assert tail.shape[-1] == 2
    xdiff = head-tail
    plt.quiver(tail[:, 0], tail[:, 1], xdiff[:, 0], xdiff[:, 1],
            scale_units='xy', angles='xy', scale=1, 
            width=width, 
            color=color, 
            alpha=alpha)


def plot_arrows_2d(tail, head=None, width=0.005, color="b", alpha=0.5):
    plot_arrows_1(tail, head, width, color, alpha)


def plot_path_2d(x, y=None, width=0.005, color="b", alpha=0.5):
    """
    Plot a path with arrows
    x : [N] or [N, 2]
    y : [N] or None
    """
    if y is not None:
        x = np.vstack([x, y]).T
    plot_arrows_2d(x[:-1, :], x[1:, :])
    

def plot_pathes_2d(m, width=0.005, color="b", alpha=0.5):
    """
    Plot a path with arrows
    m : [..., 2]
    """
    assert m.shape[-1] == 2
    m = np.reshape(m, [-1, 2])
    plot_path_2d(m[:, 0], m[:, 1], 
            width=width, 
            color=color, 
            alpha=alpha)

    

def plot_2nd_order_points_2d(x, color="g"):
    """
    x : [N, 4]
    """
    plot_arrows_2d(x, color=color)
    

def plot_2nd_order_mapping_2d(x, y=None, width=0.005, alpha=0.5):
    """
    x : [N, 4] or [N, 3, 2]
    y : [N, 2] or None
    """
    if y is None:
        plot_arrows_2d(x[:, 0, :], x[:, 1, :], color="b", width=width, alpha=alpha)
        plot_arrows_2d(x[:, 1, :], x[:, 2, :], color="r", width=width, alpha=alpha)
    else:
        plot_arrows_2d(x, color="b", width=width, alpha=alpha)
        plot_arrows_2d(x[:, 2:4], y, color="r", width=width, alpha=alpha)
    

if __name__ == "__main__":
    N = 20
    plot_pathes_2d(np.reshape(np.random.uniform(size=2*N), [-1, 2]))
    plt.show()
    #exit()

    N = 10
    x = np.reshape(np.random.uniform(size=N*4), [N, 4])
    y = np.reshape(np.random.uniform(size=N*2), [N, 2])
    plot_2nd_order_mapping_2d(x, y, width=0.01, alpha=0.1)
    plt.show()
    exit()



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    w = 3
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    U = -1 - X**2 + Y
    V = 1 + X - Y**2
    speed = np.sqrt(U*U + V*V)

    fig = plt.figure(figsize=(7, 9))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.streamplot(X, Y, U, V, density=[0.5, 1])
    ax0.set_title('Varying Density')

    # Varying color along a streamline
    ax1 = fig.add_subplot(gs[0, 1])
    strm = ax1.streamplot(X, Y, U, V, color=U, linewidth=2, cmap='autumn')
    fig.colorbar(strm.lines)
    ax1.set_title('Varying Color')

    #  Varying line width along a streamline
    ax2 = fig.add_subplot(gs[1, 0])
    lw = 5*speed / speed.max()
    ax2.streamplot(X, Y, U, V, density=0.6, color='k', linewidth=lw)
    ax2.set_title('Varying Line Width')

    # Controlling the starting points of the streamlines
    seed_points = np.array([[-2, -1, 0, 1, 2, -1], [-2, -1,  0, 1, 2, 2]])

    ax3 = fig.add_subplot(gs[1, 1])
    strm = ax3.streamplot(X, Y, U, V, color=U, linewidth=2,
                        cmap='autumn', start_points=seed_points.T)
    fig.colorbar(strm.lines)
    ax3.set_title('Controlling Starting Points')

    # Displaying the starting points with blue symbols.
    ax3.plot(seed_points[0], seed_points[1], 'bo')
    ax3.axis((-w, w, -w, w))

    # Create a mask
    mask = np.zeros(U.shape, dtype=bool)
    mask[40:60, 40:60] = True
    U[:20, :20] = np.nan
    U = np.ma.array(U, mask=mask)

    ax4 = fig.add_subplot(gs[2:, :])
    ax4.streamplot(X, Y, U, V, color='r')
    ax4.set_title('Streamplot with Masking')

    ax4.imshow(~mask, extent=(-w, w, -w, w), alpha=0.5,
            interpolation='nearest', cmap='gray', aspect='auto')
    ax4.set_aspect('equal')

    plt.tight_layout()
    plt.show()