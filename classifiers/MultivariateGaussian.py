## Multivariate Gaussian classs
## TODO: how to estimate covariance matrix by ME?
##
## Jimmie Felidae @ Feb. 17, 2014

import numpy as np
from MachineLearningBase import MachineLearningBase

class MultivariateGaussian(MachineLearningBase):
    @staticmethod
    def _multigauss_fn(means, cov):
        """
        In:  means - 1D
             cov - 2D matrix
        """
        # TODO: covarience
        k = len(means)   # dimension
        means = np.matrix(means)
        cov = np.matrix(cov)
        c1 = np.power((2*np.pi), -k/2) * np.power(np.linalg.det(cov), -0.5)  # coeff
        invcov = cov.I
        def inner_fn(xs):
            deltas = np.matrix(xs) - means
            y = c1 * np.exp((deltas * invcov * deltas.T) * (-0.5))
            return y
        return inner_fn

    def train(self, vectors, classes):
        categorized_data = {}
        for index, class_ in enumerate(classes):
            if class_ not in categorized_data:
                categorized_data[class_] = []
            categorized_data[class_].append(vectors[index])
        vlen = len(vectors[0])
        def extract_i(i):
            def inner_fn(vector):
                return vector[i]
            return inner_fn
        self._pdfs = {}
        for class_ in categorized_data:
            class_vectors = categorized_data[class_]
            means = np.zeros(vlen)
            cov = np.zeros((vlen, vlen))
            for i in range(0, vlen):
                feature_i = map(extract_i(i), class_vectors)
                mean_i = np.mean(feature_i)
                std_i = np.std(feature_i)
                means[i] = mean_i
                cov[i][i] = std_i
            self._pdfs[class_] = self._multigauss_fn(means, cov)
    def predict(self, vectors):
        results = [0] * len(vectors)   # preallocate mem
        for index, vector in enumerate(vectors):
            max_ = 0
            max_class = None
            for class_ in self._pdfs:
                p = self._pdfs[class_](vector)
                if p > max_:
                    max_ = p
                    max_class = class_
            results[index] = max_class
        return results
