import numpy as np
import ImageSegmentationByEM_unfinished
from scipy.stats import multivariate_normal as mvn

import random
#
# x = np.array([100, 150, 200])
# print(x.shape == (3, ))
# points_set = np.array([x, x, x, x, x, x])
# points_set[0] = points_set[0] + 10
# points_set[1] = points_set[1] + 20
# points_set[2] = points_set[2] - 10
# points_set[3][1] = points_set[3][1] + 20
# points_set[4] = points_set[4] + 50

k = 2
dimension = 3
points_set = np.load("greenTrain.npy", allow_pickle=True)
mean_list, covar_list, alpha_list = ImageSegmentationByEM_unfinished.likelyhoodMaximization(points_set, k)
print("outcome:     ")
print(mean_list)
print(covar_list)
print(alpha_list)

# a = np.eye(3, 3)
# a = np.array([a, a, a])
# b = np.array([[1, 2, 3]]).T
# c = a.copy()
# for i in range(3):
#     c[i, :] = np.multiply(b[i, :], a[i])
# print(c)
