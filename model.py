from tools import getThreshold
from preprocess import Data


class Model:
    def __init__(self, model):
        self.model = model

    def train(self, train_data: Data, valid_data: Data):
        raise NotImplementedError

    def score(self, data: Data):
        raise NotImplementedError

    def predict(self, data: Data, threshold):
        scores = self.score(data)
        predictions = []
        for i in range(len(scores)):
            if scores[i]<threshold:
                predictions.append(-1)
            else:
                predictions.append(1)
        return predictions

    def get_best_threshold(self, data: Data):
        assert data.targets
        scores = self.score(data)
        valid_pred_true = [(i, j) for i, j in zip(scores, data.targets)]
        return getThreshold(valid_pred_true)