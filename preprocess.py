import copy
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import logging
import re
import jieba

# logger = logging.Logger()

class Data:
    def __init__(self, urls, titles, contents, triples, statistics, abstracts, targets=None):
        self.urls = urls
        self.titles = titles
        self.contents = contents
        self.triples = triples
        self.statistics = statistics
        self.abstracts = abstracts
        self.targets = targets


class PreprocessedData:
    def __init__(self, path='/home/cza/ccks/Data/train.json', mode='train', seed=42):
        self.path = path
        self.mode = mode
        self.seed = seed  # make sure train_val_split correct
        self.sentiment_train_path = '/home/cza/ccks/Data/sentiment_train_data'

        self.train_url_to_json = '/home/cza/ccks/Data/train_url_to_triples.json'
        self.valid_url_to_json = '/home/cza/ccks/Data/valid_url_to_triples.json'
        self.test_url_to_json = '/home/cza/ccks/Data/test_url_to_triples.json'

        self.stop_vocab_path = '/home/cza/ccks//Data/stopwords-master/cn_stopwords.txt'

        self.stop_words = self.build_stop_vocab()
        self.data = Data(*self.fit())
        self.train_data , self.valid_data = self.train_val_split(self.data, seed=self.seed, split=0.8)

    def build_stop_vocab(self):
        stop_words = set()
        with open(self.stop_vocab_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                stop_words.add(line)
        return stop_words

    def filter_stop_words(self, content, max_len):

        cut_list = jieba.cut(content[:max_len])
        new_content = ''
        for word in cut_list:
            if word not in self.stop_words:
                new_content += word
        return new_content


    def get_title_data(self):
        """
        :return:  titles: list
        """
        titles = []
        max_len = 0
        with open(self.path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data_ = json.loads(line)
                title = data_['title']
                max_len = max(max_len, len(title))
                # print(len(title))
                title = self.filter_stop_words(title, 100)
                title_sub = re.sub("[^\u4e00-\u9fa5]", '', title)
                titles.append(title_sub)
        print(max_len)
        return titles

    def get_content_data(self, max_len=368):
        """
        :return: contents: list
        """
        # import jieba

        contents = []
        with open(self.path, 'r', ) as f:
            lines = f.readlines()
            for line in lines:
                data_ = json.loads(line)
                content = data_['content']
                content = self.filter_stop_words(content, 1000)
                content_sub = re.sub("[^\u4e00-\u9fa5]", '', content)[:max_len]
                contents.append(content_sub)
        return contents

    def get_statistic_data(self):
        """
        :return: statistics: list
        """
        from statitics_utils import is_entities_in_content
        statistics = []
        with open(self.path, 'r') as f:
            lines = list(f.readlines())

            for line in tqdm(lines):
                data_ = json.loads(line)
                title_length = len(data_['title'])
                content_length = len(data_['content'])
                entities_num = len(data_['entities'].keys())
                entities_in_titles = is_entities_in_content(data_['entities'].keys(), data_['content'])
                statistics.append([title_length, content_length, entities_num, entities_in_titles])

        # with open(self.sentiment_train_path, 'rb') as f:
        #     sentiment_data = pickle.load(f)
        #
        # statistics = np.hstack([statistics, sentiment_data])

        return statistics


    def get_triples_data(self, max_triples=10):
        """
        :return: triples: list
        """
        all_triples = []
        if self.mode == 'train':
            path = self.train_url_to_json
        elif self.mode == 'valid':
            path = self.valid_url_to_json
        else:
            path = self.test_url_to_json

        with open(path, 'r') as f:
            dic = json.load(f)
            for url, triples in tqdm(dic.items()):
                temp_triples = copy.deepcopy(triples)
                while len(temp_triples) < max_triples:
                    temp_triples.append([])
                all_triples.append(temp_triples[:max_triples])
                    # print(url)
        return all_triples

    def get_target(self):
        """
        :return: targets: list
        """
        targets = []
        with open(self.path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data_ = json.loads(line)
                targets.append(int(data_['label']))
        return targets

    def get_abstract(self, max_len=368):
        """

        :return:
        """

        abstracts = []
        if self.mode=='train':
            df = pd.read_csv("/home/cza/ccks/Data/train.tsv", sep="\t")['abstract']
        else:
            # df = pd.read_csv("/home/cza/ccks/Data/val.unlabel.tsv", sep="\t")['abstract']
            df = []
        for ab in df:
            # print(ab)
            ab_sub = re.sub("[^\u4e00-\u9fa5]", '', ab)[:max_len]
            abstracts.append(ab_sub)
        return abstracts

    def get_urls(self):
        urls = []
        with open(self.path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data_ = json.loads(line)
                urls.append(data_['url'])
        return urls

    def fit(self):
        """
        :return: (titles, contents, triples, statistics) or Nothing
        """
        urls = self.get_urls()
        print("prepare url complete")
        titles = self.get_title_data()
        print('prepare title complete')
        contents = self.get_content_data()
        print('prepare content complete')
        triples = self.get_triples_data()
        print('prepare triple complete')
        statistics = self.get_statistic_data()
        print('prepare statistic complete')
        abstracts = self.get_abstract()
        print('prepare abstract complete')

        if self.mode.lower() == 'train':
            targets = self.get_target()
        elif self.mode.lower() == 'valid':
            with open('/home/cza/ccks/models/valid_bert_target.pkl', 'rb') as f:
                targets = pickle.load(f)
        else:
            targets = np.ones_like(urls)
        return urls, titles, contents, triples, statistics, abstracts, targets


    @staticmethod
    def train_val_split(data: Data, seed=42, split=0.9):
        """
        :return: train_data_list, valid_data_list
        """

        data_list = list(data.__dict__.values())

        b = int(split * len(data_list[0]))

        for d in data_list:
            # assert len(d) == b
            np.random.seed(seed)
            np.random.shuffle(d)

        return Data(*[d[:b] for d in data_list]), Data(*[d[b:] for d in data_list])

if __name__=="__main__":
    preprocess_data = PreprocessedData(mode='test')
    # targets = preprocess_data.train_data.targets
    # contents = preprocess_data.train_data.contents
    # print(len(contents), len(targets))
    # titles = []
    # with open('Data/train.json', 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         print("*"*10)
    #         data_ = json.loads(line)
    #         entities = data_['entities']
    #         content = data_['content']
    #         for k, v in entities.items():
    #             print(k, v)
    #         print("*"*10)
    #         break
