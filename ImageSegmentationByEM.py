import numpy as np
import matplotlib.pyplot as plt


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
    color observation and is called the Posterior (responsibilities)
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

    # compute the responsibilities before normalization
    responsibilities = {}
    for k in range(len(mean_set)):
        mean = mean_set[k]
        covar = covar_set[k]
        alpha = alpha_set[k]
        weightedLikelihoods = [alpha*GaussianPDF(point, mean, covar) for point in point_set]
        responsibilities[k] = weightedLikelihoods
    # responsibilities normalization


    return responsibilities


def weightedLikelihood(point, mean, covar, alpha):
    """compute the conditional probability of color observation given the color label
     (likelihood) of a color label, aka the prior
        Args:
            point_set: a set contains points in column
            mean: means of the n-Dimension gaussian distribution
            covar: covariance matrix of the n-Dimension gaussian distribution
            alpha: alpha represent weight of this n-Dimension gaussian distribution in the whole mixture distribution
        Return:
            the pdf of the this point given mean and covariance
        """
    assert point == mean.size  # dimension
    assert mean.size[1] == covar.size[0]  # dimension
    assert mean.size[1] == covar.size[1]  # dimension
    weightedLikelihood = GaussianPDF(point, mean, covar)
    return alpha*weightedLikelihood