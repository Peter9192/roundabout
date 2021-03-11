import matplotlib.pyplot as plt
import numpy as np


def get_scaled_colors(values, cmap="RdYlGn"):
    """Map a range of values onto a colormap."""
    values = np.array(values)
    translated = values - values.min()
    scaled = translated / translated.max()
    cm = plt.cm.get_cmap(cmap)
    return cm(scaled, alpha=0.5)
