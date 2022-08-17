from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from molmagic.config import plotting as cfg


sns.set_style(cfg["plot-style"])


def get_plot_name(feature_name: str, atom_sequence: tuple[int]) -> tuple[str]:
    """Calculate the name of a plot based on the feature and atom sequence"""

    # TODO: #45 Use the feature name to get a more accurate plot name
    element_map = {1: "H", 6: "C", 7: "N", 8: "O"}

    if feature_name == "hbonds":
        # Hbonds are a special case
        return (
            feature_name,
            f"{element_map[atom_sequence[0]]}-H...{element_map[atom_sequence[1]]}",
        )
    elif feature_name == "angles":
        # Create the usual sequence (excluding the coordination measure)
        sequence = list(map(lambda x: element_map[x], atom_sequence[:3]))

        # Extract the carbon coordination if applicable
        # Coordination no will be 0 if the central atom
        # is not carbon
        sequence[1] += str(atom_sequence[3] or "")
    elif feature_name in ["bonds", "dihedrals"]:
        # For bonds and dihedrals the same code works
        sequence = map(lambda x: element_map[x], atom_sequence)
    else:
        raise NotImplementedError(f"{feature_name} not implemented for plotting.")

    return feature_name, "-".join(sequence)


def draw_and_save_hist(
    data: np.ndarray, density: tuple[np.ndarray], bins: np.ndarray, name: tuple[str]
) -> None:
    """
    ::args::
        name : tuple[feature name, feature type]
    """
    # Create a new figure
    fig = plt.figure()

    # Plot the data as a histogram
    ax = sns.histplot(data, stat="density")  # Normalize the area under the histogram

    # Plot vertical lines for every bin boundary
    # Do not try to plot lines at positive or negative infinity
    [ax.axvline(i) for i in bins if i not in [-np.inf, np.inf]]

    # Plot the KDE
    sns.lineplot(x=density[0], y=density[1], ax=ax)

    # Set the figure title
    axis_title = {
        "bonds": "Bond length",
        "angles": "Angle",
        "dihedrals": "Dihedral angle",
        "hbonds": "Hydrogen bond length",
    }
    feature_name, file_name = name
    fig.suptitle(f"{feature_name.capitalize()}: {file_name}")

    # Set axis title
    ax.set_xlabel(axis_title[feature_name])

    # Save the figure
    outdir = Path(cfg["save-dir"]) / feature_name
    outdir.mkdir(exist_ok=True, parents=True)

    plt.savefig(outdir / (file_name + ".png"))

    # Close the figure
    plt.close()
    plt.close(fig)
