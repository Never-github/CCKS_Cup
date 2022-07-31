import numpy as np
from tensorflow import keras
from keras.metrics import Precision, Recall
import sys
sys.path.append("..")
from preprocess import PreprocessedData, Data
from model import Model
from pykeen.triples import TriplesFactory
from tqdm import tqdm
import pickle
from tools import transform_prediction_to_json, getThreshold, evaluation_func
import os


class RnnModel(keras.models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tf = TriplesFactory.from_path('/home/cza/ccks/Data/triples.txt')
        self.input_dim = self.tf.num_entities + 1
        self.rank = 100
        self.input_length = 20
        self.embedding = keras.layers.Embedding(self.input_dim, self.rank, input_length=self.input_length)
        # self.embedding.set_weights(entities_embeddings)
        self.lstm1 = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, kernel_regularizer=keras.regularizers.l2()))
        self.lstm2 = keras.layers.Bidirectional(keras.layers.LSTM(64, kernel_regularizer=keras.regularizers.l2()))

        self.dense1 = keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2())
        self.dense = keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.l2())

    def call(self, inputs, training=None, mask=None):

        embeddings = self.embedding(inputs)
        lstm = self.lstm1(embeddings)
        lstm = keras.layers.Dropout(0.5)(lstm)

        lstm = self.lstm2(lstm)
        lstm = keras.layers.Dropout(0.5)(lstm)

        output = self.dense1(lstm)
        output = keras.layers.Dropout(0.5)(output)

        output = self.dense(output)
        return output


class MyRnn(Model):
    def __init__(self, model):
        super().__init__(model)
        self.model: RnnModel = model
        self.embedding_num = self.model.input_dim
        self.input_length = self.model.input_length

    def train(self, train_data: Data, valid_data: Data, **kwargs):
        train_x, train_y = self.preprocess(train_data)
        valid_x, valid_y = self.preprocess(valid_data)
        self.model.build(input_shape=(None, self.input_length))

        entities_embeddings = pickle.load(open('/home/cza/ccks/test_pre_stratified_complex/entity_embedding.pkl', 'rb'))
        last_embedding = np.random.uniform(-1, 1, (1, self.model.rank))
        entities_embeddings = np.vstack((entities_embeddings, last_embedding))
        self.model.embedding.set_weights([entities_embeddings])
        self.model.embedding.trainable = True
        self.model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy', Precision(), Recall()])
        batch_size = 64

        earlystop = keras.callbacks.EarlyStopping(
            monitor='val_acc',
            patience=3,
            verbose=2,
            mode='max'
        )

        checkpoint = keras.callbacks.ModelCheckpoint(
            kwargs['save_path'],
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            mode='max'
        )

        self.model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=100, validation_data=(valid_x, valid_y),
                       steps_per_epoch=valid_x.shape[0]//batch_size, callbacks=[earlystop, checkpoint])

        # self.model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=100, callbacks=[earlystop, checkpoint], validation_split=0.1)

    def score(self, data: Data, **kwargs):
        load_path = kwargs['load_path']
        self.model.load_weights(load_path)
        x, y = self.preprocess(data)
        scores = self.model.predict(x)
        return scores

    def predict(self, data: Data, threshold, **kwargs):
        scores = self.score(data, load_path=kwargs['load_path'])
        predictions = np.zeros_like(scores)
        predictions[scores > threshold] = 1
        return predictions

    def preprocess(self, data: Data):
        all_triples = data.triples
        targets = data.targets

        entities_of_contents = []
        for triples in tqdm(all_triples):
            triples_id_flatten = []
            for t in triples:
                if t:
                    triples_id_flatten.append(self.model.tf.entity_to_id[t[0]])
                    triples_id_flatten.append(self.model.tf.entity_to_id[t[2]])
                else:
                    triples_id_flatten.extend([self.embedding_num - 1, self.embedding_num - 1])

            entities_of_contents.append(triples_id_flatten)

        return np.array(entities_of_contents), np.array(targets)


if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    preprocess_data = PreprocessedData(path='../Data/train.json')
    rnn = RnnModel()
    model = MyRnn(rnn)
    load_path = '../weights/lstm1'
    save_path = '../weights/lstm1'
    model.train(preprocess_data.train_data, preprocess_data.valid_data, save_path=save_path)

    # valid data evaluation
    valid_scores = model.score(preprocess_data.valid_data, load_path=load_path)
    valid_ys = preprocess_data.valid_data.targets
    threshold = getThreshold([[p, t] for p, t in zip(valid_scores, valid_ys)])
    valid_preds = model.predict(preprocess_data.valid_data, 0.5, load_path=load_path)
    acc, p, r, f1 = evaluation_func(valid_ys, valid_preds)

    with open('log.txt', 'a') as f:
        info = f'model{__file__} ,path: {load_path}, acc: {acc}, p: {p}, r: {r}, f1: {f1} \n'
        print(info)
        f.writelines(info)


    # test_data
    test_pre_data = PreprocessedData(path='/home/cza/ccks/Data/val.unlabel.json', mode='valid')
    test_scores = model.score(test_pre_data.data, load_path=load_path)
    test_preds = np.zeros_like(test_scores)
    test_preds[test_scores > threshold] = 1
    test_ys = test_pre_data.data.targets
    acc, p, r, f1 = evaluation_func(test_ys, test_preds)
    info = f'model{__file__} ,path: {load_path}, acc: {acc}, p: {p}, r: {r}, f1: {f1} \n'
    print(info)


    # transform_prediction_to_json(predictions=test_preds, urls=test_pre_data.data.urls, output_path='result.txt')
