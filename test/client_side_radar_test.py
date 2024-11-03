from ctypes.wintypes import POINT
import pygame
import requests
from math import cos, sin, pi, sqrt, atan2, radians
from time import sleep

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 600, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Radar Visualization")
clock = pygame.time.Clock()
CENTER = (WIDTH // 2, HEIGHT // 2)
SWEEP_COLOR = (0, 255, 0)  # Green for radar sweep
POINT_COLOR = (255, 0, 0)  # Red for points
max_distance = 200  # Max distance the sensor can read
i = 0


# Function to fetch data from the Raspberry Pi
def fetch_data():
    global i
    try:
        response = requests.get("http://192.168.8.163:5000/get_data")
        if response.status_code == 200:
            i += 1
            if i == 20:
                i = 0
            return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
    return [0 for _ in range(20)]


# Function to calculate points for the radar
def calculate_point(m, d):
    # Convert 'm' to slope between 0 and 180 degrees
    angle_radians = (pi / 4) * (1 - m)  # This gives us an angle in radians

    # Calculate the x, y coordinates using trigonometry
    x = d * cos(angle_radians)
    y = d * sin(angle_radians)

    # Rotate the point by -135 degrees (in radians: -135 * pi / 180 = -3pi/4)
    theta = -3 * pi / 4
    x_rotated = x * cos(theta) - y * sin(theta)
    y_rotated = x * sin(theta) + y * cos(theta)

    return x_rotated, y_rotated


def calculate_triangle_points(x1, y1, center_x=None, center_y=None, angle_offset=10):
    """Calculates 2 points on either side of the given point (x1, y1) to form a sector."""
    if center_x is None:
        center_x = CENTER[0]
    if center_y is None:
        center_y = CENTER[1]
    x1 = max(0.0000001, x1)
    y1 = max(0.0000001, y1)
    # Distance from the center to (x1, y1)
    d = ((2 * pi) * sqrt((x1**2) + (y1**2))) / 80
    x2 = x1 + (d / (sqrt(1 + ((x1 / y1) ** 2))))
    y2 = y1 + (((-1 * x1) / y1) * (x2 - x1))

    x3 = x1 - (d / (sqrt(1 + ((x1 / y1) ** 2))))
    y3 = y1 + (((-1 * x1) / y1) * (x3 - x1))
    return [(x2, y2), (x3, y3)]


# Main loop for visualization
if __name__ == "__main__":
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # Fetch data from Raspberry Pi
        data = fetch_data()
        if i == 19:
            pygame.draw.polygon(
                screen, (0, 255, 0), ((CENTER[0], CENTER[1]), (0, 0), (WIDTH, 0))
            )
            # Draw the radar visualization
            for i, x in enumerate(data):
                m = (2 * i) / 19 - 1
                point = calculate_point(m, x)
                extra_points = calculate_triangle_points(point[0], point[1])
                p1 = (CENTER[0], CENTER[1])
                p2 = (
                    extra_points[0][0] * 5 + CENTER[0],
                    extra_points[0][1] * 5 + CENTER[1],
                )
                p3 = (
                    extra_points[1][0] * 5 + CENTER[0],
                    extra_points[1][1] * 5 + CENTER[1],
                )

                pygame.draw.polygon(screen, (0, 0, 0), (p1, p2, p3))
            print(f"Minimum Distance: {min(data)}cm.")
        pygame.display.flip()
        clock.tick(30)
