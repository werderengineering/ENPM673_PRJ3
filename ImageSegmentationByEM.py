import numpy as np
import matplotlib.pyplot as plt
import random

def tryCustomGeneratedData(size=(50, 50), showFigure=True):
    """generate a group of data follow gaussian distribution.
    show a multivariate normal, multinormal or Gaussian distribution in plot
    Args:
        size: how many data you wanna generate
    Return:
    """
    mean = (0, 6)
    cov = [[0.1, 0], [0, 0.1]]
    random = np.random.multivariate_normal(mean, cov, size)

    if showFigure:
        random_1D = random.reshape(random.size)
        ar = np.arange(random.size)  # just as an 1D array
        fig = plt.figure()
        plt.scatter(random_1D, ar)
        plt.title("Gaussian Distribution")
        plt.show()


def GaussianPDF(x, mean, covar):
    """ccompute the pdf of a n-Dimension gaussian distribution
    Args:
        x: one single n-D point, values of different Dimension in row
        mean: mean of the n-Dimension gaussian distribution
        covar: covariance matrix of the n-Dimension gaussian distribution
    Return:
        the pdf of the this point given mean and covariance
    """
    dimension = x.shape[1]
    assert x.shape == (1, dimension), "feature vector size: " + str(x.shape)
    assert mean.shape == (1, dimension), "mean size: " + str(mean.shape)
    assert covar.shape == (dimension, dimension), "covariance size: " + str(mean.shape)
    offset = x - mean  # offset between the point and mean
    pdf = np.exp(-1 / 2 * np.dot(np.dot(offset, covar), offset.T))
    pdf = 1 / ((2 * np.pi) ** (dimension / 2) * np.sqrt(np.linalg.det(covar))) * pdf
    return pdf


def responsibilities(point_set, mean_set, covar_set, alpha_set):
    """compute the conditional probability of a color label given the
    color observation, aka the Posterior (responsibilities)
    Args:
        point_set: a set contains points in column
        mean_set: k means of the n-Dimension gaussian distribution in this point set
        covar_set: k covariance matrix of the n-Dimension gaussian distribution in this point set
        alpha_set: k alpha represent weight of each n-Dimension gaussian distribution in the whole mixture distribution
    Return:
        the pdf of the this point given mean and covariance
    """
    # make sure the input list is correct
    assert point_set[0].size == mean_set[0].size    # dimension
    assert mean_set[0].size[1] == covar_set[0].size[0]  # dimension
    assert mean_set[0].size[1] == covar_set[0].size[1]  # dimension
    assert len(mean_set) == len(covar_set) == len(alpha_set)    # k

    # extract parameters
    k = len(mean_set)
    # compute the responsibilities before normalization
    responsibilities = []
    for index_cluster in range(k):
        """for each cluster, compute the weightedLikelihoods, 
        and store them in the dictionary called responsibilities"""
        mean = mean_set[k]
        covar = covar_set[k]
        alpha = alpha_set[k]
        weightedLikelihoods = [alpha*GaussianPDF(point, mean, covar) for point in point_set]
        responsibilities[k] = weightedLikelihoods
    # responsibilities normalization
    sums_responsibilityAllCluster = [sum(responsibility) for responsibility in zip(*responsibilities)]  # for each of data point, get sum of all culster responsibility
    for index_cluster in range(k):
        responsibilities[index_cluster] = [responsibility / sums_responsibilityAllCluster for responsibility, sums_responsibilityAllCluster in zip(responsibilities[index_cluster], sums_responsibilityAllCluster)]
    # shape check
    assert len(responsibilities) == len(mean_set)
    assert len(responsibilities[0]) == len(point_set[0].size)

    return responsibilities


def likelyhoodMaximization(point_list, k):
    """  find the parameters weight factors, means, covariances such that those would maximize
        the corectness of  conditional probability of a color label given the color observation.
        Args:
            point_list: a set contains points in column
        Return:
            the pdf of the this point given mean and covariance
        """
    # make sure the input list is correct
    assert type(point_list) == list  # input format is ok
    assert point_list[0].size == 3  # dimension check

    # initialization of means and covariances
    dimension = point_list[0].size
    alpha_list = [1/k] * k
    mean_list = random.choices(point_list, k=k)
    covar_list = [np.eye(dimension, dimension) for index_cluster in range(k)]
    for index, covar in enumerate(covar_list):
        var_rand = 255 * np.random.rand(dimension)
        var_rand = list(var_rand.astype(int))
        covar_list[index] = np.fill_diagonal(covar, var_rand)

    # iteration for coverage

    # calculate the normalized responsibilities of each cluster
    respons = responsibilities(point_list, mean_list, covar_list, alpha_list)
    # update means, covariances, and weight factors
    point_array = np.asarray(point_list)
    mean_array_list = [np.asarray(mean) for mean in mean_list]

    for index in range(k):
        respon = respons[index]
        # make respon as numpy array
        mean_array_list[index] = np.dot(point_array)/sum(respon)
        # numpy array dot

    return mean_list, covar_list