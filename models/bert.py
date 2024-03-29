#! -*- coding: utf-8 -*-
import os
import random

from bert4keras.backend import keras, set_gelu
from bert4keras.models import build_transformer_model
import sys
sys.path.append("..")
from preprocess import PreprocessedData, Data
from model import Model
import numpy as np
from bert4keras.backend import keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from sklearn.metrics import classification_report
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from keras.losses import binary_crossentropy, sparse_categorical_crossentropy
from pathlib import Path
import re
import pickle
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from tools import evaluation_func, getThreshold, transform_prediction_to_json, fbeta_score
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
	# output = keras.layers.Softmax(axis=-1)(output)

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
		if Path(kwargs['load_path']).exists():
			self.model.load_weights(kwargs['load_path'])
			print("loading model ")
			print("loading model ")
			print("loading model ")
		train_ = self.process_data(train_data)
		valid_ = self.process_data(valid_data)
		train_val_ = np.vstack((train_, valid_))
		# true_valid = self.process_data(kwargs['true_valid_data'])
		train_generator = data_generator(train_, kwargs['batch_size'])
		valid_generator = data_generator(valid_, kwargs['batch_size'])
		train_val_generator = data_generator(train_val_, kwargs['batch_size'])
		# true_valid_generator = data_generator(true_valid, kwargs['batch_size'])
		AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
		self.model.compile(
			# loss='sparse_categorical_crossentropy',
			loss=binary_crossentropy,
			# loss = sparse_categorical_crossentropy,
			optimizer=Adam(1e-5),
			# optimizer=AdamLR(learning_rate=1e-5, lr_schedule={
			# 	1000: 1,
			# 	2000: 0.1
			# }),
			# optimizer=Adam(lr=5e-6),
			metrics=[Accuracy(), Precision(), Recall(), fbeta_score],
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
			train_val_generator.forfit(),
			steps_per_epoch=len(train_generator),
			epochs=1,
			validation_data=valid_generator.forfit(),
			validation_steps=len(valid_generator),
			shuffle=True,
			callbacks=[earlystop, checkpoint],
		)

	def score(self, data: Data, **kwargs):
		self.model.load_weights(kwargs['load_path'])
		data_ = self.process_data(data)
		d_generator = data_generator(data_, kwargs['batch_size'])
		scores = []
		for x, y in d_generator:
			scores.extend(self.model.predict(x))
		return np.array(scores).flatten()

	@staticmethod
	def process_data(train_data: Data):
		targets = np.array(train_data.targets)
		titles = np.array(train_data.titles)
		contents = np.array(train_data.contents)
		# print(targets.shape, titles.shape)
		contents_head = np.array([c for c in contents])
		# contents_tail = np.array([c[-128:] for c in contents])

		titles_contents = np.array([t+'[SEP]'+c_h for t, c_h in zip(titles, contents_head)])
		# print(titles_contents)
		#abstracts = np.array(train_data.abstracts)
		#titles_abstracts = np.array([(t+c) for t, c in zip(titles, abstracts)])

		train_ = np.hstack((titles_contents[:, np.newaxis], targets[:, np.newaxis]))
		# train_ = [(i, j) for i, j in zip(titles_abstracts, targets)]

		# test_targets = np.array(valid_data.targets)
		# test_titles = np.array(valid_data.titles)
		# test_contents = np.array(valid_data.contents)
		# test_titles_contents = np.array([(t + '[SEP]' + c)[:768] for t, c in zip(test_titles, test_contents)])
		# test_ = np.hstack((test_titles_contents[:, np.newaxis], test_targets[:, np.newaxis]))
		return train_

def pre_tokenize(text):
	"""单独识别出[xxx]的片段
	"""
	tokens, start = [], 0
	for r in re.finditer('\[[^\[]+\]', text):
		tokens.append(text[start:r.start()])
		tokens.append(text[r.start():r.end()])
		start = r.end()
	if text[start:]:
		tokens.append(text[start:])
	return tokens

if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"]="0"

	config_path= '../chinese_L-12_H-768_A-12/bert_config.json'
	checkpoint_path='../chinese_L-12_H-768_A-12/bert_model.ckpt'
	dict_path = '../chinese_L-12_H-768_A-12/vocab.txt'

	load_path = '../weights/bert_sub_tc_all_ep1'
	save_path = '../weights/bert_sub_tc_all_ep1'

	bert = build_bert_model(config_path, checkpoint_path)
	model = myBert(bert)

	tokenizer, maxlen = Tokenizer(dict_path, pre_tokenize=pre_tokenize), 420
	batch_size = 6

	# train bert
	train_preprocess_data = PreprocessedData()
	valid_preprocess_data = PreprocessedData(path='/home/cza/ccks/Data/val.unlabel.json', mode='valid')
	model.train(train_preprocess_data.train_data, train_preprocess_data.valid_data, load_path=load_path, save_path=save_path, batch_size=batch_size)
	# scores = model.score(preprocess_data.data, load_path=load_path, batch_size=batch_size)
	# with open('train_bert_pred.pkl', 'wb') as f:
	# 	pickle.dump(scores, f)

	# valid data evaluation
	valid_scores = model.score(train_preprocess_data.valid_data, load_path=load_path, batch_size=batch_size)
	valid_preds = np.zeros_like(valid_scores)
	valid_ys = train_preprocess_data.valid_data.targets
	threshold = getThreshold([[p, t] for p, t in zip(valid_scores, valid_ys)])
	threshold = 0.5
	valid_preds[valid_scores>threshold] = 1
	acc, p, r, f1 = evaluation_func(valid_ys, valid_preds)
	with open('log.txt', 'a') as f:
		info = f'model{__file__} ,path: {save_path}, max_len: {maxlen}, batch_size: {batch_size}, acc: {acc}, p: {p}, r: {r}, f1: {f1} \n'
		f.writelines(info)

	# test_data
	test_pre_data = PreprocessedData(path = '/home/cza/ccks/Data/test.unlabel.json', mode='test')
	test_scores = model.score(test_pre_data.data, load_path=load_path, batch_size=batch_size)
	test_preds = np.zeros_like(test_scores)
	test_preds[test_scores>threshold] = 1
	with open('test_bert_pred.pkl', 'wb') as f:
		pickle.dump(test_preds, f)
	transform_prediction_to_json(predictions=test_preds, urls=test_pre_data.data.urls, output_path='result.txt')
