import matplotlib.pyplot as plt


def plot_doodle(img_lines):
    # Setup the plot
    plt.figure(figsize=(10, 10))
    plt.axis('equal')

    # Plot each line
    for line in img_lines:
        x, y = line
        plt.plot(x, y, marker='o')  # 'o' is optional, it adds markers to the line's points

    plt.gca().invert_yaxis()  # Invert y-axis to match the drawing's coordinate system
    plt.show()
