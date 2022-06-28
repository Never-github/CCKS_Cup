import json
import pandas as pd
import numpy as np
import logging


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


        self.data = Data(*self.fit())
        self.train_data , self.valid_data = self.train_val_split(self.data, seed=self.seed, split=0.9)

    def get_title_data(self):
        """
        :return:  titles: list
        """
        titles = []
        with open(self.path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data_ = json.loads(line)
                titles.append(data_['title'])
        return titles

    def get_content_data(self):
        """
        :return: contents: list
        """
        contents = []
        with open(self.path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data_ = json.loads(line)
                contents.append(data_['content'])
        return contents

    def get_statistic_data(self):
        """
        :return: statistics: list
        """
        statistics = []
        with open(self.path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data_ = json.loads(line)
                title_length = len(data_['title'])
                content_length = len(data_['content'])
                entities_num = len(data_['entities'].keys())
                statistics.append([title_length, content_length, entities_num])
                #
        return statistics

    def get_triples_data(self):
        """
        :return: triples: list
        """
        triples = []
        pass
        return triples

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

    def get_abstract(self):
        """

        :return:
        """

        abstracts = []
        df = pd.read_csv("/home/cza/ccks/Data/train.tsv", sep="\t")['abstract']
        for ab in df:
            abstracts.append(ab)
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

        np.random.seed(seed)
        for d in data_list:
            # assert len(d) == b
            np.random.shuffle(d)

        return Data(*[d[:b] for d in data_list]), Data(*[d[b:] for d in data_list])

if __name__=="__main__":
    # preprocess_data = PreprocessedData()
    # targets = preprocess_data.train_data.targets
    # contents = preprocess_data.train_data.contents
    # print(len(contents), len(targets))
    titles = []
    with open('Data/train.json', 'r') as f:
        lines = f.readlines()
        for line in lines:
            print("*"*10)
            data_ = json.loads(line)
            entities = data_['entities']
            content = data_['content']
            for k, v in entities.items():
                print(k, v)
            print("*"*10)
            break
