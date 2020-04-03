import numpy as np
import matplotlib.pyplot as plt
import random

flag = True


def GaussianPDF(x, mean, covar):
    """ccompute the pdf of a n-Dimension gaussian distribution
    Args:
        x: one single n-D point or a array of points, values of different Dimension in row
        mean: mean of the n-Dimension gaussian distribution
        covar: covariance matrix of the n-Dimension gaussian distribution
    Return:
        the pdf of the this point given mean and covariance
    """
    assert type(x) == np.ndarray  # check list type
    assert type(mean) == np.ndarray
    assert type(covar) == np.ndarray
    dimension = x[0].shape[0]  # get dimension form point
    NumSample = len(x)
    assert x[0].shape == (dimension,), "feature vector size: " + str(x[0].shape) + " should be " + str(
        (dimension,))  # check first point shape again
    assert mean.shape == (dimension,), "mean size: " + str(mean.shape)  # check mean shape
    assert covar.shape == (dimension, dimension)  # check covariance shape

    """change the type and shape of input point and mean"""
    x = np.asarray(x)
    mean = np.asarray(mean)
    x = x.reshape((NumSample, dimension))
    mean = mean.reshape((1, dimension))
    """calculate the pdf"""
    offset = x - mean  # offset between the point and mean
    pdf = np.multiply(np.dot(offset, np.linalg.inv(covar)), offset)
    pdf = np.exp(-1 / 2 * np.sum(pdf, axis=1))
    pdf = 1 / ((2 * np.pi) ** (dimension / 2) * np.sqrt(np.linalg.det(covar))) * pdf
    # x[[np.isfinite(x)]]

    """output check"""
    assert type(pdf) == np.ndarray, "this type is " + str(type(pdf))
    assert pdf.shape == (NumSample,), "the pdf shape is " + str(pdf.shape)
    assert np.isfinite(pdf)
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
    assert point_set[0].size == mean_set[0].size  # dimension
    assert mean_set[0].size == covar_set[0].shape[0]  # dimension
    assert mean_set[0].size == covar_set[0].shape[1]  # dimension
    assert mean_set.shape[0] == covar_set.shape[0] == alpha_set.shape[0]  # k

    # extract parameters
    numSampling = point_set.shape[0]
    k, dimension = mean_set.shape
    # compute the responsibilities before normalization
    respons = np.zeros((numSampling, k))  # (k, N) list
    for index_cluster in range(k):
        """for each cluster, compute the weightedLikelihoods,
        and store them in the dictionary called responsibilities"""
        mean = mean_set[index_cluster]
        covar = covar_set[index_cluster]
        alpha = alpha_set[index_cluster]
        """check the determinant of covariance matrix, if too small, error"""
        lowerBound_covar = 0.0001
        assert abs(np.linalg.det(covar)) > lowerBound_covar, "The determinant of covariance: \n" + str(
            covar) + " at cluster #" + str(index_cluster) + " is " + str(np.linalg.det(covar)) + " smaller than " + str(
            lowerBound_covar)
        weightedLikelihoods = alpha * GaussianPDF(point_set, mean, covar)
        respons[:, index_cluster] = weightedLikelihoods
    # responsibilities normalization after figured out weight likelihood of each point for all cluster
    respons = respons / np.sum(respons, axis=1).reshape(numSampling, 1)

    # output check
    assert type(respons[0]) == np.ndarray  # check type of list
    assert respons.shape == (numSampling, k)  # check k
    assert respons.shape[0] == point_set.shape[0]  # check number of sample

    return respons


def initilizeGassuainClusterModelParameters(point_list, dimension, k):
    """  initialization of means and covariances
        Args:
            :param point_list: a set contains points in column, eack point has to be a numpy array
            :param dimension: dimension of point
            :param k: number of cluster
        Return:
            a list of mean, a list of covariance, a list of weight factor
        """
    assert point_list[0].size == dimension  # dimension check

    alpha_list = np.asarray([1 / k] * k)
    mean_list = np.asarray(random.choices(point_list, k=k))
    covar_list = [np.eye(dimension, dimension) for index_cluster in range(k)]
    for index, covar in enumerate(covar_list):
        var_rand = 255 * np.random.rand(dimension)
        var_rand = list(var_rand.astype(int))
        np.fill_diagonal(covar, var_rand)
        covar_list[index] = covar
    covar_list = np.asarray(covar_list)
    assert mean_list.shape[0] == k
    assert covar_list.shape[0] == k
    assert alpha_list.shape[0] == k
    assert mean_list[0].size == dimension
    assert covar_list[0].shape == (dimension, dimension)

    return mean_list, covar_list, alpha_list


def likelyhoodMaximization(point_list, k):
    """  find the parameters weight factors, means, covariances such that those would maximize
        the corectness of  conditional probability of a color label given the color observation.
        Args:
            point_list: a set contains points in column, eack point has to be a numpy array
        Return:
            the pdf of the this point given mean and covariance
        """
    # make sure the input list is correct
    assert type(point_list) == np.ndarray  # input format is ok
    print(point_list[0])
    NumSample, dimension = point_list.shape
    assert point_list.shape[0] > point_list.shape[1]  # number of sample should be bigger than dimension

    # initialization of means and covariances
    mean_list, covar_list, alpha_list = initilizeGassuainClusterModelParameters(point_list, dimension, k)
    if flag:
        print(str(NumSample) + " of " + str(dimension) + "d points found")
        print("Means: \n" + str(mean_list))
        print("Covars: \n" + str(covar_list))
        print("Alphas: \n" + str(alpha_list))
    """iteration until coverage
    calculate responsibilities of each cluster
    keep update means, covariances, and weight factor
    stop at 
    """
    for interator in range(1000):
        # calculate the normalized responsibilities of each point of each cluster: first index cluster, second index point
        respons = responsibilities(point_list, mean_list, covar_list, alpha_list)
        if flag:
            print("responsibilities" + str(respons))
            print(" sum of each row (point): " + str([sum(responsibility) for responsibility in zip(*respons)]))
            print(" sum of each col (cluster): " + str([sum(responsibility) for responsibility in respons]))
        # update means, covariances, and weight factors
        mean_previous_list = mean_list.copy()  # store the previous means

        for index in range(k):  # for each cluster
            respon = respons[:, index].reshape((NumSample, 1))
            assert respon.shape == (NumSample, 1)  # check the Num of samples
            assert point_list.shape[0] == NumSample  # check the Num of samples

            """update means"""
            mean_list[index, :] = (np.dot(point_list.T, respon) / np.sum(respon, axis=0)).reshape((dimension))
            """update covariance"""

            # compute the covariance of each point from point list
            covar_list_eachPoint = [np.dot((point.reshape((1, 3)) - mean_list[index].reshape((1, 3))).T,
                                           (point.reshape((1, 3)) - mean_list[index].reshape((1, 3)))) for point in
                                    point_list]
            covar_list_eachPoint = np.asarray(covar_list_eachPoint)
            assert len(covar_list_eachPoint) == NumSample
            assert covar_list_eachPoint[0].shape == (dimension, dimension)

            # get the product of each point's responsibility and covariance
            for i in range(NumSample):
                covar_list_eachPoint[i, :] = np.multiply(respon[i, :], covar_list_eachPoint[i, :])
            covar_list[index] = np.sum(covar_list_eachPoint, axis=0) / np.sum(respon, axis=0)
            """update weight factor"""
            alpha_list[index] = 1 / NumSample * np.sum(respon, axis=0)
        """check the stop criteria"""
        Convergence = sum(
            [np.linalg.norm(mean - mean_previous) for mean, mean_previous in zip(mean_list, mean_previous_list)])
        if Convergence > 0.01:
            continue
        else:
            print("Convergenced after " + str(interator) + " iteration")
            break
    return mean_list, covar_list, alpha_list