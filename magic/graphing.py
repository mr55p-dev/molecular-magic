from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from magic.config import aggregation


cfg = aggregation["plotting"]


def plot_histogram(
    data: np.ndarray, density: tuple[np.ndarray], bins: np.ndarray, name: str
) -> None:
    # Create a new figure
    fig = plt.figure()

    # Plot the data as a histogram
    ax = sns.histplot(
        data,
    )

    # Plot vertical lines for every bin boundary
    # Do not try to plot lines at positive or negative infinity
    [ax.axvline(i) for i in bins if i not in [-np.inf, np.inf]]

    # Plot the KDE
    sns.lineplot(x=density[0], y=density[1], ax=ax)

    # Set the figure title
    fig.suptitle(name)

    # Save the figure
    outdir = Path(cfg["save-dir"])
    outdir.mkdir(exist_ok=True)

    plt.savefig(outdir / (name + '.png'))
