import numpy as np
import cv2
from __main__ import *
import matplotlib.pyplot as plt
import imutils


print('Imports Complete')

print('CV2 version')
print(cv2.__version__)

flag = False
prgRun = True

def main(prgRun):
    problem = 1
    # problem = int(input('Which part would you like to run? \nEnter 1 for ngihtime image . \nEnter 2 for the firt part of the lane finder. \nEnter 3 for the second part of the lane finder (Challenge video): '))

    video = cv2.VideoCapture('detectbuoy.avi')

    # Read until video is completed
    while (video.isOpened()):
        # Capture frame-by-frame
        ret, frame = video.read()
        if ret == True:
            frame = imutils.resize(frame, width=320, height=180)
            frame.shape
            ogframe = frame
            clnframe = frame
            resetframe = frame
            cv2.imshow('Original Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break


    prgRun=False
    return prgRun







print('Function Initializations complete')

if __name__ == '__main__':
    print('Start')
    prgRun = True
    while prgRun == True:
        prgRun = main(prgRun)

    print('Goodbye!')
    cv2.destroyAllWindows()