import sys
sys.path.append("..")
from preprocess import PreprocessedData, Data
from model import Model
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class DecisionTreeModel(Model):
    def __init__(self, model):
        super().__init__(model)
        self.model: DecisionTreeClassifier = model

    def train(self, train_data: Data, valid_data: Data):
        train_x_array, train_y_array = self.process_data(train_data)
        self.model.fit(X=train_x_array, y=train_y_array)

    def score(self, data: Data):
        x_array, y_array = self.process_data(data)
        return self.model.predict_proba(x_array)[:, 0]

    def evaluate(self, data: Data):
        # x_array, y_array = self.process_data(data)
        y_pred = self.predict(data, threshold=0.5)
        y_true = data.targets
        acc = accuracy_score(y_true, y_pred)
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f'acc: {acc} p: {p} r: {r} f1: {f1}')


    @staticmethod
    def process_data(data: Data):
        targets = np.array(data.targets)
        xs = np.array(data.statistics)
        # print(targets.shape, titles.shape)
        # data_ = np.hstack((xs[:, np.newaxis], targets[:, np.newaxis]))
        return xs, targets

if __name__=="__main__":
    preprocess_data = PreprocessedData(path='../Data/train.json')
    model = DecisionTreeModel(DecisionTreeClassifier())
    model.train(preprocess_data.train_data, preprocess_data.valid_data)
    model.evaluate(preprocess_data.valid_data)

