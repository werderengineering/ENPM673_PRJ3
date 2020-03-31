import numpy as np
import pylab as plt
import os
import cv2
from scipy.stats import multivariate_normal as mvn
import random
########################################################################################
########################################################################################
#Set Train = 1 fo
#training
Train = 1
K = 2



#Set values here
testfolder = "frames"
outfolder = "EM Output"
########################################################################################
########################################################################################
flag = True

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
    dimension = x.size
    x = x.reshape((1, dimension))
    mean = mean.reshape((1, dimension))
    assert x.shape == (1, dimension), "feature vector size: " + str(x.shape)
    assert mean.shape == (1, dimension), "mean size: " + str(mean.shape)
    assert covar.shape == (dimension, dimension), "covariance size: " + str(mean.shape)
    offset = x - mean  # offset between the point and mean
    pdf = np.exp(-1 / 2 * np.dot(np.dot(offset, np.linalg.inv(covar)), offset.T))
    pdf = 1 / ((2 * np.pi) ** (dimension / 2) * np.sqrt(np.linalg.det(covar))) * pdf
    pdf = float(pdf[0,0])
    assert type(pdf) == float, "this type is " + str(type(pdf))
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
    assert mean_set[0].size == covar_set[0].shape[0]  # dimension
    assert mean_set[0].size == covar_set[0].shape[1]  # dimension
    assert len(mean_set) == len(covar_set) == len(alpha_set)    # k

    # extract parameters
    k = len(mean_set)
    # compute the responsibilities before normalization
    respons = []    # (k, N) list
    for index_cluster in range(k):
        """for each cluster, compute the weightedLikelihoods,
        and store them in the dictionary called responsibilities"""
        mean = mean_set[index_cluster]
        covar = covar_set[index_cluster]
        alpha = alpha_set[index_cluster]
        """check the determinant of covariance matrix, if too small, error"""
        lowerBound_covar = 0.001
        assert np.linalg.det(covar) > lowerBound_covar, "The determinant of covariance: \n" + str(covar) + " at cluster #" + str(index_cluster) + " is " + str(np.linalg.det(covar)) + " smaller than " + str(lowerBound_covar)
        weightedLikelihoods = [alpha*GaussianPDF(point, mean, covar) for point in point_set]
        respons.append(weightedLikelihoods)
    # responsibilities normalization
    sums_responsibilityAllCluster = [sum(responsibility) for responsibility in zip(*respons)]  # for each of data point, get sum of all culster responsibility
    for index_cluster in range(k):
        """for each responsibility, divide it by the sum of responsibility from all culsters"""
        respon = respons[index_cluster]
        # if flag:
        #     print(respon)
        #     print(sums_responsibilityAllCluster)
        respons[index_cluster] = [(responsibility / sums_responsibility_eachPooint_AllCluster) for responsibility, sums_responsibility_eachPooint_AllCluster in zip(respon, sums_responsibilityAllCluster)]

    # shape check
    assert len(respons) == len(mean_set)   # check k
    assert type(respons[0]) == list    # check type of list
    assert type(respons[0][0]) == float # check responsibility type
    assert len(respons[0]) == len(point_set)    # check number of sample

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
    assert type(point_list) == list  # input format is ok
    assert point_list[0].size == dimension  # dimension check

    alpha_list = [1 / k] * k
    mean_list = random.choices(point_list, k=k)
    covar_list = [np.eye(dimension, dimension) for index_cluster in range(k)]
    for index, covar in enumerate(covar_list):
        var_rand = 255 * np.random.rand(dimension)
        var_rand = list(var_rand.astype(int))
        np.fill_diagonal(covar, var_rand)
        covar_list[index] = covar

    assert len(mean_list) == k
    assert len(covar_list) == k
    assert len(alpha_list) == k
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
    assert type(point_list) == list  # input format is ok
    assert point_list[0].size == 3  # dimension check
    NumSample = len(point_list)
    dimension = point_list[0].size
    # initialization of means and covariances
    mean_list, covar_list, alpha_list = initilizeGassuainClusterModelParameters(point_list, dimension, k)
    if flag:
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
        mean_previous_list = mean_list.copy()   # store the previous means
        point_array = np.asarray(point_list)    # cast list of points to array of points

        for index in range(k):  # for each cluster
            respon = respons[index]
            assert len(respon) == NumSample  # check the Num of samples
            assert point_array.shape[0] == NumSample     # check the Num of samples

            """update means"""
            respon_array = np.asarray(respon).reshape((NumSample, 1))
            mean_list[index] = ( np.dot(point_array.T, respon_array)/sum(respon) ).reshape((dimension))
            """update covariance"""
            # compute the covariance of each point from point list
            covar_list_eachPoint = [np.dot((point.reshape((1,3))-mean_list[index].reshape((1,3))).T, (point.reshape((1,3))-mean_list[index].reshape((1,3)))) for point in point_list]
            assert len(covar_list_eachPoint) == NumSample
            assert covar_list_eachPoint[0].shape == (dimension, dimension)

            # sum up all the product of each point's responsibility and covariance
            covar_sum = sum([respon_onePoint*covar_onePoint for respon_onePoint, covar_onePoint in zip(respon, covar_list_eachPoint)])
            covar_list[index] = covar_sum/sum(respon)
            """update weight factor"""
            alpha_list[index] = 1/NumSample * sum(respon)
        """check the stop criteria"""
        Convergence = sum([np.linalg.norm(mean-mean_previous) for mean, mean_previous in zip(mean_list, mean_previous_list)])
        if Convergence > 0.01:
            continue
        else:
            print("Convergenced")
            break

    return mean_list, covar_list, alpha_list
# def EMalgo(xtrain,K,iters):
#
#
# 	n,d = xtrain.shape
# 	# mean = np.array([xtrain[np.random.choice(n, K)]])
# 	mean = xtrain[np.random.choice(n, K),:]
# 	Sigma = [80*np.eye(d)] * K
# 	for i in range(K):
# 		Sigma[i]=np.multiply(Sigma[i],np.random.rand(d,d))
# 		#Sigma[i]=Sigma[i]*Sigma[i].T
# 	w = [1./K] * K
# 	z = np.zeros((n, K))
#
# 	log_likelihoods = []
#
# 	while len(log_likelihoods) < iters:
# 		for k in range(K):
# 			#x_mean = np.matrix(xtrain - mean[k])
# 			#Sinv = np.linalg.pinv(Sigma[k])
# 			tmp = w[k] * mvn.pdf(xtrain, mean[k], Sigma[k],allow_singular=True)#((2.0 * np.pi) ** (-d / 2.0)) * (1.0 / (np.linalg.det(Sigma[k]) ** 0.5)) * np.exp(-0.5 * np.sum(np.multiply(x_mean * Sinv, x_mean), axis=1))
# 			z[:,k]=tmp.reshape((n,))
#
#
# 		log_likelihood = np.sum(np.log(np.sum(z, axis = 1)))
#
# 		print ('{0} -> {1}'.format(len(log_likelihoods),log_likelihood))
#
# 		if log_likelihood>-205555: break
#
# 		log_likelihoods.append(log_likelihood)
#
#
#
#
#
#
#
# 		z = (z.T / np.sum(z, axis = 1)).T
#
# 		N_ks = np.sum(z, axis = 0)
#
# 		for k in range(K):
# 			mean[k] = 1. / N_ks[k] * np.sum(z[:, k] * xtrain.T, axis = 1).T
# 			x_mean = np.matrix(xtrain - mean[k])
# 			Sigma[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mean.T,  z[:, k]), x_mean))
# 			w[k] = 1. / n * N_ks[k]
#
# 		if len(log_likelihoods) < 2 : continue
# 		if len(log_likelihoods)>10000 or np.abs(log_likelihood - log_likelihoods[-2]) < 0.0001: break
#
# 	# plt.plot(log_likelihoods)
# 	# plt.title('Log Likelihood vs iteration plot')
# 	# plt.xlabel('Iterations')
# 	# plt.ylabel('log likelihood')
# 	# plt.show()
# 	np.save('weights',w)
# 	np.save('sigma',Sigma)
# 	np.save('mean',mean)
#
# xtrain = np.load('xtrain.npy',allow_pickle = True)
#
# if(Train == 1):
# 	EMalgo(xtrain,K,1000000)
#
# w=np.load('weights.npy')
# Sigma=np.load('sigma.npy')
# mean=np.load('mean.npy')
point_list = np.load('yellowTrain.npy')
mean, Sigma, w  = likelyhoodMaximization(list(point_list), K)
w = np.asarray(w)
Sigma = np.asarray(Sigma)
mean = np.asarray(mean)
# np.save('w.npy',w)
# np.save('Sigma.npy',Sigma)
# np.save('mean.npy',mean)

for filename in os.listdir(testfolder):

	img = cv2.imread(os.path.join(testfolder,filename))
	cv2.imshow('Test', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	basename = os.path.basename(filename)
	print (basename)
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
	nr, nc, d = img.shape
	n=nr*nc
	xtest=np.reshape(img,(n,d))
	likelihoods=np.zeros((K,n))
	log_likelihood=np.zeros(n)
	for k in range(K):
		#x_mean = np.matrix(xtrain - mean[k])
		#Sinv = np.linalg.pinv(Sigma[k])
		likelihoods[k] = w[k] * mvn.pdf(xtest, mean[k], Sigma[k],allow_singular=True)#((2.0 * np.pi) ** (-d / 2.0)) * (1.0 / (np.linalg.det(Sigma[k]) ** 0.5)) * np.exp(-0.5 * np.sum(np.multiply(x_mean * Sinv, x_mean), axis=1))
		log_likelihood = likelihoods.sum(0)

	log_likelihood = np.reshape(log_likelihood, (nr, nc))

	log_likelihood[log_likelihood > np.max(log_likelihood) / 5] = 255
	output_img = np.zeros(img.shape)
	output_img[:, :, 2] = log_likelihood
	output_img = np.array(output_img)
	output_img = cv2.resize(output_img, (400, 300))

	cv2.imshow('RED',output_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.imwrite(os.path.join(outfolder, basename[:-4]+'RED'+'.png'), output_img)