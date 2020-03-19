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

    #Correct image
    if problem ==1:
        print('Being Problem')


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