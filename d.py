import pygame
import time
count = 0
pygame.mixer.init()
music = pygame.mixer.Sound("/Users/joylee/Downloads/AI/warning.wav")
if count == 0:
    music.play()
    time.sleep(5)
   