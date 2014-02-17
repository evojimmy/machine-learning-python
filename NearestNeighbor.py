## Nearest Neighbor class
## TODO: more efficient neighbor-finding
##
## Jimmie Felidae @ Feb. 17, 2014

import math
from MachineLearningBase import MachineLearningBase

class NearestNeighbor(MachineLearningBase):
    @staticmethod
    def _euclidean_distance(U, V):
        return math.sqrt(sum(map(lambda p: (p[0]-p[1])*(p[0]-p[1]), zip(U, V))))
    def train(self, vectors, classes):
        self._storage = zip(vectors, classes)
    def predict(self, vectors):
        results = [0] * len(vectors)  # preallocate mem
        for index, vector in enumerate(vectors):
            min = float("inf")
            min_class = None
            for item in self._storage:
                d = self._euclidean_distance(vector, item[0])
                if d < min:
                    min = d
                    min_class = item[1]
            results[index] = min_class
        return results
