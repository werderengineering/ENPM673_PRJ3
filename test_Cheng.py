import numpy as np
import ImageSegmentationByEM

# #
# a = np.array([[0, 1, 3]])
a = [0, 2, 4]
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
b = [a, a, a]
sums = [sum(pair) for pair in zip(*b)]
print(sums)
c = [pair/sum for pair, sum in zip(zip(*b), sums)]
print(c)
