import pygame
import threading
from pyo import *
import numpy as np

# Pygame setup
pygame.init()
window_size = 500
screen = pygame.display.set_mode((window_size, window_size))
pygame.display.set_caption("Spatial Audio Visualization")

# Constants
listener_position = (window_size // 2, window_size // 2)
listener_size = 20

# Initialize the pyo server
s = Server().boot()
s.start()

# Global Variables
source_position = [0, 0]


def update_sound():
    global source_position

    # Speed of sound in meters per second
    speed_of_sound = 343

    # Listener ear positions
    ear_distance = 0.2
    left_ear_position = (-ear_distance / 2, 0)
    right_ear_position = (ear_distance / 2, 0)

    # Continuous sine wave
    sine = Sine(freq=1100, mul=0.5)

    # Delay objects to adjust for distance
    delay_left = Delay(sine, delay=0, feedback=0)
    delay_right = Delay(sine, delay=0, feedback=0)

    # Outputs to the left and right channels
    delay_left.out(0)
    delay_right.out(1)

    while True:
        # Calculate the distance from the sound source to each ear
        distance_left = np.sqrt(
            (left_ear_position[0] - source_position[0]) ** 2
            + (left_ear_position[1] - source_position[1]) ** 2
        )
        distance_right = np.sqrt(
            (right_ear_position[0] - source_position[0]) ** 2
            + (right_ear_position[1] - source_position[1]) ** 2
        )

        # Convert distances to delays in seconds
        delay_left_time = distance_left / speed_of_sound
        delay_right_time = distance_right / speed_of_sound

        # Update the delay objects with the new delays
        delay_left.delay = float(delay_left_time)
        delay_right.delay = float(delay_right_time)

        # Apply volume scaling based on distance
        sine.mul = 0.5 / max(float(distance_left), float(distance_right), 0.1)


def main():
    global source_position
    running = True

    # Start sound update in a separate thread
    sound_thread = threading.Thread(target=update_sound, daemon=True)
    sound_thread.start()

    while running:
        screen.fill((255, 255, 255))  # White background

        # Draw the listener (red square)
        pygame.draw.rect(
            screen,
            (255, 0, 0),
            (
                listener_position[0] - listener_size // 2,
                listener_position[1] - listener_size // 2,
                listener_size,
                listener_size,
            ),
        )

        # Draw the sound source (black circle)
        mouse_x, mouse_y = pygame.mouse.get_pos()
        source_position = [
            mouse_x - listener_position[0],
            mouse_y - listener_position[1],
        ]
        pygame.draw.circle(screen, (0, 0, 0), (mouse_x, mouse_y), 10)

        pygame.display.flip()

        # Handle quitting the application
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()
    s.stop()


if __name__ == "__main__":
    main()
"""

from pyo import *
from time import sleep

s = Server(audio="portaudio", nchnls=2).boot()
s.start()

# Play sound only in the left ear
print("Playing in left!")
left = Sine(freq=440, mul=0.5).out(0)
sleep(2)  # Listen for 2 seconds
print("Stopping left!")
left.stop()

print("Playing on right!")
# Play sound only in the right ear
right = Sine(freq=440, mul=0.5).out(1)
sleep(2)
right.stop()
print("Stopped right!")
s.stop()
"""
