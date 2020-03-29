import numpy as np
import cv2
from __main__ import *
import matplotlib.pyplot as plt
import imutils
import os

from getData import *
from histogramImages import*

# Hello?

print('Imports Complete')

print('CV2 version')
print(cv2.__version__)

flag = False
prgRun = True
getdata = False

def main(prgRun):
    problem = 1

    Ohb, Ohg, Ohr = orangeHist()
    Ghb, Ghg, Ghr = greenHist()
    Yhb, Yhg, Yhr = yellowHist()

    video = cv2.VideoCapture('detectbuoy.avi')

    # Read until video is completed
    while (video.isOpened()):
        # Capture frame-by-frame
        ret, frame = video.read()
        if ret == True:
            # frame = imutils.resize(frame, width=320, height=180)
            frame.shape
            ogframe = frame
            clnframe = frame
            resetframe = frame
            # cv2.imshow('Original Frame', frame)
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break

            if getdata==True:
                buildData(frame)

            else:



                # Gaussian = getgauss(frame)
                # Expectation = getEM()

                circleframe = frame

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