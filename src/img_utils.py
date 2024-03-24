def scale_down_coordinate(coordinate, old_size, new_size):
    """
    Scale down a single coordinate from old_size to new_size.

    Parameters:
    coordinate (int): The coordinate (x or y) to be scaled down.
    old_size (int): The original size of the image (width/height).
    new_size (int): The target size of the image (width/height).

    Returns:
    int: Scaled down coordinate.
    """
    scale_factor = new_size / old_size
    return int(coordinate * scale_factor)


def draw_line(image, start, end):
    """
    Draw a line on the image using Bresenham's line algorithm.

    Parameters:
    image (list of list of int): The image array to draw the line on.
    start (tuple): The starting coordinate of the line.
    end (tuple): The ending coordinate of the line.
    """
    # Bresenham's line algorithm implementation
    # This is a simplified version without anti-aliasing

    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if 0 <= x0 < len(image) and 0 <= y0 < len(image):
            image[y0][x0] = 255  # Set the pixel value to white
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def convert_sketch_to_image(sketch, old_size=255, new_size=28):
    """
    Convert a sketch to a grayscale image.

    Parameters:
    sketch (list): The sketch represented as a list of line segments with coordinates.
    old_size (int): The original size of the sketch area.
    new_size (int): The desired size of the output image.

    Returns:
    list of list of int: The resulting grayscale image.
    """
    image = [[0 for _ in range(new_size)] for _ in range(new_size)]  # Create a blank image

    for segment in sketch:
        x_coords, y_coords = segment
        scaled_x = [scale_down_coordinate(x, old_size, new_size) for x in x_coords]
        scaled_y = [scale_down_coordinate(y, old_size, new_size) for y in y_coords]
        for i in range(len(scaled_x) - 1):
            draw_line(image, (scaled_x[i], scaled_y[i]), (scaled_x[i + 1], scaled_y[i + 1]))

    return image


def print_image(image):
    """
    Print the image as ASCII art.

    Parameters:
    image (list of list of int): The image to print.
    """
    for row in image:
        print(''.join(['#' if pixel else ' ' for pixel in row]))