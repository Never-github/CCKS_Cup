import json

import numpy as np
import logging


# logger = logging.Logger()

class Data:
    def __init__(self, titles, contents, triples, statistics, targets=None):
        self.titles = titles
        self.contents = contents
        self.triples = triples
        self.statistics = statistics
        self.targets = targets


class PreprocessedData:
    def __init__(self, path='Data/train.json', mode='train', seed=42):
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
        pass
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
                targets.append(data_['label'])
        return targets

    def fit(self):
        """

        :return: (titles, contents, triples, statistics) or Nothing
        """
        titles = self.get_title_data()
        print('prepare title complete')
        contents = self.get_content_data()
        print('prepare content complete')
        triples = self.get_triples_data()
        print('prepare triple complete')
        statistics = self.get_statistic_data()
        print('prepare statistic complete')
        if self.mode.lower() == 'train':
            targets = self.get_target()
            return titles, contents, triples, statistics, targets
        else:
            return titles, contents, triples, statistics


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
            data_ = json.loads(line)
            print(data_['entities'])
            titles.append(data_.keys())
            break
