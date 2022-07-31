import sys
sys.path.append("..")
from preprocess import PreprocessedData, Data
from model import Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle

class sklearnModel(Model):
    def __init__(self, model):
        super().__init__(model)
        self.model: DecisionTreeClassifier = model

    def train(self, train_data: Data, valid_data: Data):
        train_x_array, train_y_array = self.process_data(train_data)
        self.model.fit(X=train_x_array, y=train_y_array)

    def score(self, data: Data):
        x_array, y_array = self.process_data(data)
        return self.model.predict_proba(x_array)[:, 0]

    def predict(self, data: Data, threshold=0.5):
        x_array, y_array = self.process_data(data)
        return self.model.predict(x_array)

    def evaluate(self, data: Data):
        # x_array, y_array = self.process_data(data)
        y_pred = self.predict(data)
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
    print(preprocess_data.data.statistics.shape)
    model = sklearnModel(LogisticRegression())
    model.train(preprocess_data.train_data, preprocess_data.valid_data)
    scores = model.score(preprocess_data.data)
    with open('train_lr_pred.pkl', 'wb') as f:
        pickle.dump(scores, f)
    model.evaluate(preprocess_data.valid_data)
    # test set
    test_pre_data = PreprocessedData(path='/home/cza/ccks/Data/val.unlabel.json', mode='test')
    test_scores = model.score(test_pre_data.data)
    # with open('test_lr_pred.pkl', 'wb') as f:
    #     pickle.dump(test_scores, f)

