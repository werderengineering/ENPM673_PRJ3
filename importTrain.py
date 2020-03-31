import cv2
import os
import numpy as np
def import_train_samples():
    train = []
    for filename in os.listdir("orangeSamples"):
        image = cv2.imread(os.path.join("orangeSamples", filename))
        nx,ny,ch = image.shape
        image = np.reshape(image, (nx * ny, ch))
        for i in range(image.shape[0]):
            train.append(image[i,:])
    train = np.array(train)

    return train

np.save('orangeTrain.npy', import_train_samples())
#
# n=np.load('yellowTrain.npy',allow_pickle=True)
# print(n)
