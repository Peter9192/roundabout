import matplotlib.pyplot as plt
import numpy as np

from roundabout import draw_chord_diagram

if __name__ == "__main__":
    labels = "ABCDE"
    matrix = np.random.randint(low=5, high=25, size=(5, 5))
    colors = plt.get_cmap("Set1")(np.linspace(0, 1, 5))
    threshold = 10

    draw_chord_diagram(
        matrix,
        labels=labels,
        colors=colors,
        spacing="proportional",
        padding=5,
        sort_rows="mincross",
        threshold=threshold,
    )
    plt.savefig("example.png", transparent=False, bbox_inches="tight", dpi=200)
    plt.close()
