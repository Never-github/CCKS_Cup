#! -*- coding: utf-8 -*-
from bert4keras.backend import keras, set_gelu
from bert4keras.models import build_transformer_model
import sys
sys.path.append("..")
from preprocess import preprocess_data, Data
from model import Model
import numpy as np
from bert4keras.backend import keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from sklearn.metrics import classification_report
from bert4keras.optimizers import Adam
from keras.losses import binary_crossentropy
from preprocess import PreprocessedData
set_gelu('tanh')

def textcnn(inputs,kernel_initializer):
	# 3,4,5
	cnn1 = keras.layers.Conv1D(
			256,
			3,
			strides=1,
			padding='same',
			activation='relu',
			kernel_initializer=kernel_initializer
		)(inputs) # shape=[batch_size,maxlen-2,256]
	cnn1 = keras.layers.GlobalMaxPooling1D()(cnn1)  # shape=[batch_size,256]

	cnn2 = keras.layers.Conv1D(
			256,
			4,
			strides=1,
			padding='same',
			activation='relu',
			kernel_initializer=kernel_initializer
		)(inputs)
	cnn2 = keras.layers.GlobalMaxPooling1D()(cnn2)

	cnn3 = keras.layers.Conv1D(
			256,
			5,
			strides=1,
			padding='same',
			kernel_initializer=kernel_initializer
		)(inputs)
	cnn3 = keras.layers.GlobalMaxPooling1D()(cnn3)

	output = keras.layers.concatenate(
		[cnn1,cnn2,cnn3],
		axis=-1)
	output = keras.layers.Dropout(0.2)(output)
	return output


def build_bert_model(config_path, checkpoint_path):
	bert = build_transformer_model(
		config_path=config_path,
		checkpoint_path=checkpoint_path,
		model='bert',
		return_keras_model=False,
	)


	cls_features = keras.layers.Lambda(
		lambda x:x[:,0],
		name='cls-token'
		)(bert.model.output) #shape=[batch_size,768]
	all_token_embedding = keras.layers.Lambda(
		lambda x:x[:,1:-1],
		name='all-token'
		)(bert.model.output) #shape=[batch_size,maxlen-2,768]

	cnn_features = textcnn(
		all_token_embedding,bert.initializer) #shape=[batch_size,cnn_output_dim]
	concat_features = keras.layers.concatenate(
		[cls_features,cnn_features],
		axis=-1)

	dense = keras.layers.Dense(
			units=512,
			activation='relu',
			kernel_initializer=bert.initializer
		)(concat_features)

	output = keras.layers.Dense(
			units=1,
			activation='sigmoid',
			kernel_initializer=bert.initializer
		)(dense)

	model = keras.models.Model(bert.model.input,output)

	return model


class data_generator(DataGenerator):
	"""
    数据生成器
    """
	def __iter__(self, random=False):
		batch_token_ids, batch_segment_ids, batch_labels = [], [], []
		for is_end, (text, label) in self.sample(random):
			token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)#[1,3,2,5,9,12,243,0,0,0]
			batch_token_ids.append(token_ids)
			batch_segment_ids.append(segment_ids)
			batch_labels.append([label])
			if len(batch_token_ids) == self.batch_size or is_end:
				batch_token_ids = sequence_padding(batch_token_ids)
				batch_segment_ids = sequence_padding(batch_segment_ids)
				batch_labels = sequence_padding(batch_labels)
				yield [batch_token_ids, batch_segment_ids], batch_labels
				batch_token_ids, batch_segment_ids, batch_labels = [], [], []

class myBert(Model):
	def __init__(self, model):
		super().__init__(model)

	def train(self, train_data: Data, valid_data: Data, **kwargs):
		"""

		:param train_data:
		:param valid_data:
		:param kwargs: save_path, load_path
		:return:
		"""
		self.model.load_weights(kwargs['load_path'])
		train_, valid_ = self.process_data(train_data, valid_data)
		train_generator = data_generator(train_, kwargs['batch_size'])
		valid_generator = data_generator(valid_, kwargs['batch_size'])
		self.model.compile(
			# loss='sparse_categorical_crossentropy',
			loss=binary_crossentropy,
			optimizer=Adam(5e-6),
			metrics=['accuracy'],
		)

		earlystop = keras.callbacks.EarlyStopping(
			monitor='val_loss',
			patience=3,
			verbose=2,
			mode='min'
		)

		checkpoint = keras.callbacks.ModelCheckpoint(
			kwargs['save_path'],
			monitor='val_loss',
			verbose=1,
			save_best_only=True,
			mode='min'
		)

		self.model.fit_generator(
			train_generator.forfit(),
			steps_per_epoch=len(train_generator),
			epochs=100,
			validation_data=valid_generator.forfit(),
			validation_steps=len(valid_generator),
			shuffle=True,
			callbacks=[earlystop, checkpoint]
		)

	def score(self, data: Data):
		pass



	@staticmethod
	def process_data(train_data: Data, valid_data: Data):
		targets = np.array(train_data.targets)
		titles = np.array(train_data.titles)
		# print(targets.shape, titles.shape)
		train_ = np.hstack((titles[:, np.newaxis], targets[:, np.newaxis]))

		test_targets = np.array(valid_data.targets)
		test_titles = np.array(valid_data.titles)
		test_ = np.hstack((test_titles[:, np.newaxis], test_targets[:, np.newaxis]))
		return train_, test_

if __name__ == '__main__':
	config_path= 'chinese_L-12_H-768_A-12/bert_config.json'
	checkpoint_path='chinese_L-12_H-768_A-12/bert_model.ckpt'
	dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'

	load_path = '../weights/bert_.h5'
	save_path = '../weights/bert_.h5'

	bert = build_bert_model(config_path, checkpoint_path)
	model = myBert(bert)

	tokenizer, maxlen = Tokenizer(dict_path), 256

	preprocess_data = PreprocessedData()
	model.train(preprocess_data.train_data, preprocess_data.valid_data, load_path=load_path, save_path=save_path)





