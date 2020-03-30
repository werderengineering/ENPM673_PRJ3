import numpy as np
import ImageSegmentationByEM
import random
# #
a = np.array([100, 150, 200])
# a = np.array([1, 2, 3]).reshape((1,3))
# mean = np.array([4, 5, 6]).reshape((1, 3))
# print(a.T)
# b = np.dot((a).T, (mean))
# print(b)
# print(a.shape)
print(type(a))
points_set = [a, a, a, a, a]
points_set[0] = points_set[0] + 10
points_set[1] = points_set[1] + 20
points_set[2][0] = points_set[2][0] + 10
points_set[3][1] = points_set[3][1] + 20
points_set[4][2] = points_set[4][2] + 30
print(points_set)

k = 2
dimension = 3

mean_list, covar_list, alpha_list = ImageSegmentationByEM.likelyhoodMaximization(points_set, k)
print("outcome:     ")
print(mean_list)
print(covar_list)
print(alpha_list)
