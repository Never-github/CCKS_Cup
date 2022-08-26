# CCKS 2022 
## 基于知识图谱的优质文章识别
### knowledge graph enhanced language model
two kinds of information are considered: text and knowledge graph. 


### preprocesss
easily get training and testing data through preprocess.py

### text
BERT+TextCNN: title+content
implemented in models/bert.py

### knowledge graph
train embeddings for entities with representation learning: TransE and ComplEx
a BiLSTM model is applied to finetune the embeddings
implemented in models/lstm.py

### final
both the above two methods can obtain considerable performance.
fuse the two models to perform better


##  Learning with Knowledge Graph Guided Attention
another form of knowledge graph enhanced language model

a new learning strategy

coming soon...
