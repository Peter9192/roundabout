"""Visualize a matrix in the form of a radial network graph."""
# gradients: https://stackoverflow.com/a/29331211
# https://github.com/Silmathoron/mpl_chord_diagram/blob/master/gradient.py

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np


@np.vectorize
def bezier(p0, p2, pref, n=50):
    t = np.linspace(0, 1, n)
    return (1 - t) ** 2 * p0 + 2 * t * (1 - t) * pref + t ** 2 * p2


@np.vectorize
def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


@np.vectorize
def cartesian_to_polar(x, y):
    r = (x ** 2 + y ** 2) ** 0.5
    theta = np.arctan2(y, x)
    return r, theta


def polar_bezier_curve(theta1, theta2, r=1):
    """Return a quadratic bezier curve in polar coordinates.

    theta1 and theta2 are two angles on the unit circle
    The origin is used as third reference point for the cubic bezier.
    """
    x, y = polar_to_cartesian([r, r], [theta1, theta2])
    xcurve = bezier(*x, pref=0)
    ycurve = bezier(*y, pref=0)
    r, theta = cartesian_to_polar(xcurve, ycurve)
    return r, theta


def constuct_chord(p1, p2, p3, p4):
    """Connect 4 points on a unit circle to draw a polygon."""

    r1, theta1 = polar_bezier_curve(p1, p2)
    r2, theta2 = np.ones(10), np.linspace(p2, p3, 10)
    r3, theta3 = polar_bezier_curve(p3, p4)
    r4, theta4 = np.ones(10), np.linspace(p4, p1, 10)

    r = np.concatenate([r1, r2, r3, r4])
    theta = np.concatenate([theta1, theta2, theta3, theta4])

    return r, theta


def sort(matrix, strategy="original"):
    """Sort rows of a matrix with a given strategy."""

    n = len(matrix)

    if strategy == "original":
        sorter = np.tile(np.arange(n), (n, 1))

    elif strategy == "increasing":
        sorter = np.argsort(matrix, axis=1)

    elif strategy == "mincross":
        sorter = np.array([-(np.arange(n) - i) % n for i in range(n)])

    else:
        raise ValueError("Unknown strategy %s", strategy)

    colidx = np.arange(len(matrix))[:, None]
    msorted = matrix[colidx, sorter]

    unsorter = np.argsort(sorter, axis=1)
    return msorted, unsorter


def calculate_segments(matrix, spacing="even", padding=5):
    """Calculate arc segments representing the cells of a matrix."""

    n = len(matrix)
    pad = np.pi / 180 * padding  # degrees to radians

    if spacing == "even":
        edges = np.linspace(0, 1, n * n + 1)[1:]
    elif spacing == "proportional":
        edges = np.cumsum(matrix) / matrix.sum()
    else:
        raise NotImplementedError(spacing=spacing)

    edges = edges.reshape(n, n) * (2 * np.pi - n * pad)

    # Duplicate outer edges so we can insert padding between the main segments
    edges = np.roll(np.append(edges, edges[:, -1][:, None], axis=1), 1)
    edges[0, 0] = 0
    for i, row in enumerate(edges):
        row += i * pad

    # Segments are the intervals [start, stop] between all pairs of edges
    segments = np.array([edges[:, :-1], edges[:, 1:]]).transpose(1, 2, 0)
    return segments


def draw_chord_diagram(
    matrix,
    labels,
    colors,
    spacing="proportional",
    padding=5,
    sort_rows="mincross",
    threshold=None,
):
    """Draw a chord diagram.

    More info
    """
    n = len(matrix)

    sorted_matrix, unsort = sort(matrix, strategy=sort_rows)
    segments = calculate_segments(sorted_matrix, spacing=spacing, padding=padding)

    # Plot arcs
    ax = plt.subplot(111, projection="polar")
    ax.set_axis_off()
    for arc, color in zip(segments, colors):
        r = np.linspace(arc.min(), arc.max(), 25)
        ax.fill_between(r, 1, 1.1, color=color)

    # Plot chords
    for i, j in combinations(range(n), 2):

        p1, p4 = segments[i, unsort[i, j]]
        # p1, p4 = segments[i, j] if unsort is applied to segments instead
        p3, p2 = segments[j, unsort[j, i]]
        # p3, p2 = segments[j, i] if unsort is applied to segments instead
        r, theta = constuct_chord(p1, p2, p3, p4)

        width = matrix[i, j]
        if threshold and width > threshold:
            ax.fill(theta, r, color=colors[i], alpha=0.75, lw=0)
            ax.plot(theta, r, color="k", lw=0.5)
        else:
            ax.fill(theta, r, color=colors[i], alpha=0.25, lw=0)

    # Add labels
    for segment, label in zip(segments, labels):
        midpoint = (segment.min() + segment.max()) / 2
        ax.text(midpoint, 1.2, label, ha="center", va="center")
