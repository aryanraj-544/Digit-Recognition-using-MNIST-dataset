from matplotlib import testing
import pygame, sys, os
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WINDOWSZEX = 640
WINDOWSZEY = 480
BOUNDRYINC = 5
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)

IMAGESAVE = False

MODEL = load_model("bestmodel.h5")

LABELS = {0: "Zero", 1: "One",
          2: "Two", 3: "Three",
          4: "Four", 5: "Five", 
          6: "Six", 7: "Seven", 
          8: "Eight", 9: "Nine"}

# Initialize pygame
pygame.init()
ttf_path = os.path.join(sys.path[0], "OpenSans-Regular.ttf")
FONT = pygame.font.Font(ttf_path, 18)
# FONT = pygame.font.Font("freesansbold.tff", 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSZEX, WINDOWSZEY))

pygame.display.set_caption("Digit Board")

iswriting = False
img_count = 1
PREDICT = True
number_xcord = []
number_ycord = []

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4,0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0]-BOUNDRYINC, 0), min(WINDOWSZEX, number_xcord[-1]+BOUNDRYINC)
            rect_min_y, rect_max_y = max(number_ycord[0]-BOUNDRYINC, 0), min(number_ycord[-1]+BOUNDRYINC, WINDOWSZEY)

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("test.png")
                img_count +=1

            if PREDICT:
                image = cv2.resize(img_arr, dsize=(28,28))
                image = np.pad(image, (10,10), 'constant',  constant_values=0)
                image = cv2.resize(image, dsize=(28,28))/255

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])

                textSurface = FONT.render(label, True, RED, WHITE)
                textRectObj = pygame.Surface.get_rect()
                textRectObj.left , textRectObj.bottom = rect_min_x, rect_max_y

                DISPLAYSURF.blit(textSurface, textRectObj)

            if event.type == KEYDOWN:
                if event.unicode == "n":
                    DISPLAYSURF.fill(BLACK)

            pygame.display.update()