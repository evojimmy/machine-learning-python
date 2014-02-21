## Base Machine Learning class
## it should depict the learning process and provide
## performance metrics.
##
## TODO: add more evaluation metrics
##
## Jimmie Felidae @ Feb. 17, 2014


class MachineLearningBase(object):
    def train(self, vectors, classes):
        """
        In:  vectors - 2D list
             classes - 1D list
        """
        raise NotImplementedError("Not implemented in base class")
    def predict(self, vectors):
        """
        In:  vectors - 2D list
        Out: classes - 1D list
        """
        raise NotImplementedError("Not implemented in base class")
    def evaluate(self, ground_truths, predicts):
        # TODO
        return self.evaluate_error_rate(ground_truths, predicts)
    def evaluate_accuracy(self, ground_truths, predicts):
        correct = len(filter(lambda p: p[0] == p[1], zip(ground_truths, predicts)))
        all = len(ground_truths)
        return float(correct) / float(all)
    def evaluate_error_rate(self, ground_truths, predicts):
        return 1 - self.evaluate_accuracy(ground_truths, predicts)
