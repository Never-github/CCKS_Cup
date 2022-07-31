from preprocess import PreprocessedData
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from tools import evaluation_func, transform_prediction_to_json
bert_pred = pickle.load(open('train_bert_pred.pkl', 'rb'))
lr_pred = pickle.load(open('train_lr_pred.pkl', 'rb'))
print(bert_pred.shape, lr_pred.shape)
preprocess_data = PreprocessedData(path='../Data/train.json')
y = np.array(preprocess_data.data.targets)
print(y.shape)
X = np.hstack((bert_pred[:, np.newaxis], lr_pred[:, np.newaxis]))
split = int(np.floor(len(X)*0.95))
train_x, val_x = X[:split], X[split:]
train_y, val_y = y[:split], y[split:]
lr = LogisticRegression()
lr.fit(train_x, train_y)
# scores = lr.predict_proba(X)[:, 0]
val_preds = lr.predict(val_x)
evaluation_func(val_y, val_preds)

test_preprocess_data = PreprocessedData(path='../Data/val.unlabel.json', mode='test')
test_bert_pred = pickle.load(open('valid_bert_target.pkl', 'rb'))
test_lr_pred = pickle.load(open('test_lr_pred.pkl', 'rb'))
print(test_bert_pred.shape, test_lr_pred.shape)
test_x =  np.hstack((test_bert_pred[:, np.newaxis], test_lr_pred[:, np.newaxis]))
test_preds = lr.predict(test_x)
transform_prediction_to_json(test_preds, test_preprocess_data.data.urls, )



