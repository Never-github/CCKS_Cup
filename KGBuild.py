import json
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

class KGBuilder:
    def __init__(self):
        self.graph = defaultdict(list)
        self.triples = []
        self.entities_num_per_text = []

    def build(self, path='Data/train.json'):
        with open(path, 'r') as f:
            lines = list(f.readlines())
            for line in tqdm(lines):
                data_ = json.loads(line)
                entities = data_['entities']
                self.entities_num_per_text.append(len(entities))
                for k, v in entities.items():
                    co_occur = list(set(v['co-occurrence']))
                    self.graph[k].extend(co_occur)


        return self.graph

    @staticmethod
    def transform_dataset_to_triples(path='Data/test.unlabel.json', output='Data/test_url_to_triples.json'):
        """
        dataset to triples
        save as json
        :param path:
        :return:
        """
        url_to_triples = dict()
        with open(path, 'r') as f:
            lines = list(f.readlines())

            for line in tqdm(lines):
                data_ = json.loads(line)
                entities = data_['entities']
                url = data_['url']
                triples = []
                for h, v in entities.items():
                    co_occur = list(set(v['co-occurrence']))
                    for t in co_occur:
                        triples.append([h, 'rel', t])
                url_to_triples[url] = triples

        with open(output, 'w') as f:
            json.dump(url_to_triples, f)



    def transform_graph_to_triples(self):
        for h, tails in self.graph.items():
            for t in tails:
                self.triples.append([h, 'co_occur', t])


    def save_triples_as_txt(self,output_path="Data/triples.txt"):
        with open(output_path, 'a') as f:
            for h, r, t in self.triples:
                line = h + '\t' + r + '\t' + t + '\n'
                f.writelines(line)
        print('triples ok')






if __name__=="__main__":
    kg = KGBuilder()
    # # kg.build('Data/train.json')
    # # kg.build('Data/val.unlabel.json')
    # # all_subs_num = sum(kg.entities_num_per_text)
    # # kg_ent_num = len(kg.graph)
    # # print(kg.graph)
    # # print(all_subs_num, kg_ent_num)
    # # print(kg_ent_num/all_subs_num)
    # # kg.transform_graph_to_triples()
    # # print(len(kg.triples))
    # # kg.save_triples_as_tsv()
    kg.transform_dataset_to_triples()

    # with open('Data/train_url_to_triples.json', 'r') as f:
    #     dic = json.load(f)
    #     min_length = 10000
    #     for k, v in dic.items():
    #         min_length = min(min_length, len(v))
    #         if len(v)<5:
    #             print(len(v), v)
    #     print(min_length)




