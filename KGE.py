import numpy as np
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.training import training_loop
from pykeen.datasets.nations import NATIONS_TRAIN_PATH, NATIONS_TEST_PATH
import pickle

tf = TriplesFactory.from_path('Data/triples.txt')

print(tf.num_entities)
print(tf.entities_to_ids(['.com', '007', '007']))

training, testing = tf.split()
result = pipeline(
    training=training,
    testing=testing,
    model='ComplEx',
    model_kwargs=dict(
        embedding_dim=100,
    ),
    # stopper='early',
    negative_sampler='basic',
    epochs=200,  # short epochs for testing - you should go higher
)
result.save_to_directory('test_pre_stratified_complex')
print(result.get_metric('hits@'))
print(result.get_metric('mrr'))
print(result.get_metric('hits@1'))
print(result.get_metric('hits@3'))
print(result.get_metric('hits@10'))

import torch
#
# model = torch.load('test_pre_stratified_transe/training_triples/base.pth')

model = result.model
entity_embedding_tensor = model.entity_representations[0](indices=None).detach().cpu().numpy()
relation_embedding_tensor = model.relation_representations[0](indices=None).detach().cpu().numpy()

with open('test_pre_stratified_complex/entity_embedding.pkl', 'wb') as f:
    pickle.dump(entity_embedding_tensor, f)
with open('test_pre_stratified_transe/relation_embedding.pkl', 'wb') as f:
    pickle.dump(relation_embedding_tensor, f)



