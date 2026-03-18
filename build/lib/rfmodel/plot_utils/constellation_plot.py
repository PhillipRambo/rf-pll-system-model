import matplotlib.pyplot as plt
import numpy as np

def plot_constellation(x, title="Constellation Diagram", max_points=None):
    """
    Plot complex constellation points.

    Parameters
    ----------
    x : array-like
        Complex symbols.
    title : str
        Plot title.
    max_points : int | None
        If set, plot only the first max_points symbols.
    """
    x = np.asarray(x)

    if max_points is not None:
        x = x[:max_points]

    plt.figure(figsize=(6, 6))
    plt.scatter(x.real, x.imag, s=20)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.title(title)
    plt.grid(True)
    plt.axis("equal")
    plt.show()

"""
usage: plot_constellation(sig_out.x, title="64-QAM Constellation - All Symbols")

"""