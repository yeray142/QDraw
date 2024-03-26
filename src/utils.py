import matplotlib.pyplot as plt
from typing import List


def plot_doodle(img_lines: List[List[int]]) -> None:
    """
    Plot doodles using matplotlib
    Args:
        img_lines: doodle img lines.

    Returns: None

    """
    # Setup the plot
    plt.figure(figsize=(10, 10))
    plt.axis('equal')

    # Plot each line
    for line in img_lines:
        x, y = line
        plt.plot(x, y, marker='o')  # 'o' is optional, it adds markers to the line's points

    plt.gca().invert_yaxis()  # Invert y-axis to match the drawing's coordinate system
    plt.show()
