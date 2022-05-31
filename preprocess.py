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
    def __init__(self, path='data/train.json', mode='train', seed=42):
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
        pass
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
        pass
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
        return targets

    def fit(self):
        """

        :return: (titles, contents, triples, statistics) or Nothing
        """
        titles = self.get_title_data()
        contents = self.get_content_data()
        triples = self.get_triples_data()
        statistics = self.get_statistic_data()
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
            assert len(d) == b
            np.random.shuffle(d)

        return Data(*[d[:b] for d in data_list]), Data(*[d[b:] for d in data_list])

if __name__=="__main__":
    preprocess_data = PreprocessedData()
    targets = preprocess_data.train_data.targets
    contents = preprocess_data.train_data.contents
    print(len(contents), len(targets))