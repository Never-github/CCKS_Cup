# import jieba
# import jiagu
import json
import pickle

import numpy as np
from tqdm import tqdm

def get_sentiment_per_text(text):
    category, score = jiagu.sentiment(text)
    category = 1 if category=='positive' else 0
    return [category, score]

def save_sentiment_data_as_pickle(path='Data/train.json', pkl_path='Data/sentiment_train_data'):
    title_sentiment_data = []
    with open(path, 'r') as f:
        lines = list(f.readlines())
        for line in tqdm(lines, desc='title'):
            data_ = json.loads(line)
            title = data_['title']
            title_sentiment_data.append(get_sentiment_per_text(title))
    content_sentiment_data = []
    all_sentences = get_sentences(path=path, num=10)
    for sentences in tqdm(all_sentences, desc='content'):
        temp = []
        assert len(sentences)==10, len(sentences)
        for s in sentences:
            c, score = get_sentiment_per_text(s)
            temp.extend([c, score])
        content_sentiment_data.append(temp)

    print(np.array(title_sentiment_data).shape, np.array(content_sentiment_data).shape)
    sentiments = np.hstack([np.array(title_sentiment_data), np.array(content_sentiment_data)])


    with open(pkl_path, 'wb') as f:
        pickle.dump(sentiments, f)



def is_entities_in_content(entities, content):
    for e in entities:
        if e in content:
            return 1
    return 0

def get_articles_att():
    with open('Data/train.json', 'r') as f:
        lines = list(f.readlines())
        contents = []
        for line in lines:
            data_ = json.loads(line)
            contents.append(data_['content'])
        all_contents = ' '.join(contents)

        print(len(set(jieba.cut(all_contents))))

def get_sentences(path='Data/train.json', num = 10):
    with open(path, 'r') as f:
        lines = list(f.readlines())

        sentences_lenth = []
        all_sentences = []
        for line in tqdm(lines):
            data_ = json.loads(line)
            content = data_['content']
            sentences = content.split("<br/>")
            if len(sentences)<num:
                sentences = content.split("。")
            sentences_lenth.append((len(sentences)))
            all_sentences.append(sentences[:num])
        print(min(sentences_lenth), max(sentences_lenth))
    return all_sentences





if __name__=="__main__":
    save_sentiment_data_as_pickle()