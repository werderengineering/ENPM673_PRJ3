import numpy as np
import ImageSegmentationByEM
import random
# #
# a = np.array([[0, 1, 3]])
points_set = [1, 2, 4, 6, 8]
# b = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# c = np.dot(np.dot(a,b),a.T)
# #
# print(a.shape)
# print(b.shape)


# a = np.array([[0, 0, 0]])
# mean = np.array([[0, 0, 0]])
# covar = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# a = np.array([[0]])
# mean = np.array([[0]])
# covar = np.array([[1]])
#
# pdf = ImageSegmentationByEM.GaussianPDF(a, mean, covar)
# print(pdf)


# b = [a, a, a]
# sums = [sum(pair) for pair in zip(*b)]
# print(sums)
# for i in range(3):
#     b[i] = [pair/sum for pair, sum in zip(b[i], sums)]
#     print(b[i])
#
# print(b)

# mean_set = random.choices(points_set, k=3)
# print(mean_set)
k = 2
dimension = 3
covar_set = [np.eye(dimension, dimension) for index_cluster in range(k)]
print(covar_set)

for covar in covar_set:
    ran = 255 * np.random.rand(dimension)
    ran = list(ran.astype(int))
    covar = np.fill_diagonal(covar, ran)

print(covar_set)
