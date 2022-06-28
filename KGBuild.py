import json
from collections import defaultdict
from tqdm import tqdm


class KGBuilder:
    def __init__(self, path='Data/train.json'):
        self.path = path
        self.graph = defaultdict(list)
        self.entities_num_per_text = []

    def build(self):
        with open(self.path, 'r') as f:
            lines = list(f.readlines())
            for line in tqdm(lines):
                data_ = json.loads(line)
                entities = data_['entities']
                self.entities_num_per_text.append(len(entities))
                for k, v in entities.items():
                    co_occur = v['co-occurrence']
                    self.graph[k].extend(co_occur)
        return self.graph


if __name__=="__main__":
    kg = KGBuilder()
    kg.build()
    all_subs_num = sum(kg.entities_num_per_text)
    kg_ent_num = len(kg.graph)
    print(kg.graph)
    print(all_subs_num, kg_ent_num)
    print(kg_ent_num/all_subs_num)


