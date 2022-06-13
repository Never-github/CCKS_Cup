import jsonlines
import pandas as pd
import numpy as np
from bert4keras.snippets import to_array


def getThreshold(rrank):
    distanceFlagList = rrank
    distanceFlagList = sorted(distanceFlagList, key=lambda sp: sp[0], reverse=False)

    threshold = distanceFlagList[0][0] - 0.01
    minValue = 0
    currentValue = 0
    for i in range(1, len(distanceFlagList)):
        if distanceFlagList[i - 1][1] == 1:
            currentValue += 1
        else:
            currentValue -= 1

        if currentValue < minValue:
            threshold = (distanceFlagList[i][0] + distanceFlagList[i - 1][0]) / 2.0
            minValue = currentValue
    # print('threshold... ', threshold)
    return threshold


def read_data(filedir='Data/train.json'):
    items = list(jsonlines.open(filedir))
    keys = items[0].keys()
    data_dict = {}
    for key in keys:
        data_dict[key] = [item[key] for item in items]
    return pd.DataFrame(data_dict)


####################################################
#|                 text preprocessing             |#
####################################################
from document_extract import document_extract


def generate_abstract(text, max_budget, tokenizer, model):
    text = preprocess_text(text)
    sents = get_sentences(text)
    sent_embeddings = get_sent_embeddings(sents, tokenizer, model)
    sent_lengths = get_sent_length(sents)
    indices = document_extract(sent_embeddings, sent_lengths, max_budget)
    return recover_text(sents, indices)
    

segment_list = "。？！"


def preprocess_text(sent : str):
    sent = sent.replace('<br/>', '')
    return sent


def get_sentences(text):
    si = [i+1 for i in range(len(text)) \
        if text[i] in segment_list] # segment indices
    si.insert(0, 0)
    if text[-1] not in segment_list:
        si.append(len(text))
    return [text[si[i]:si[i+1]] for i in range(len(si)-1)]


def get_sent_embeddings(sents, tokenizer, model):
    sent_embeddings = [encode_sentence(s, tokenizer, model) for s in sents]
    return np.stack(sent_embeddings, axis=0)


def encode_sentence(sent, tokenizer, bert_model):
    token_ids, segment_ids = tokenizer.encode(sent)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    embedding = bert_model.predict([token_ids, segment_ids])
    return np.concatenate([embedding[0, 0], embedding[0, -1]], axis=0)


def get_sent_length(sents):
    return [len(s) for s in sents]


def recover_text(sents, indices):
    return ''.join([sents[i] for i in sorted(list(indices), reverse=False)])


