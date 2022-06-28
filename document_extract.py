import tensorflow as tf
import numpy as np


def sim(sent1, sent2):
    """ similarity between two sentences
    
    naive choice is cosine similarity. could be replaced.
    """
    return np.dot(sent1, sent2)


def coverage_score(sentences, selection, alpha):
    """ the coverage score of a subset of sentences 
    
        it measure how much the selection covers all the information of the whole corpus.
    """
    N, S = len(sentences), len(selection)
    score = 0
    for i in range(N):
        score += tf.minimum(sum([sim(sentences[i], selection[j]) for j in range(S)]),
                            alpha * sum([sim(sentences[i], sentences[j]) for j in range(N)])
                            )

    return score


def document_extract(sentence_embeddings, sentence_lengths, max_budget, alpha=0.9, paragraph_partitions=None):
    """ extract key sentences from the whole corpus

    sentence_embeddings: dtype is numpy.ndarray ; dimension = [len_of_sequence, embedding_dim]
    sentence_lengths: a list of integers, each entry is the length of the corresponding sentence
    max_budget: the maximal length of the extracted text
    
    """
    N = len(sentence_embeddings)
    indices = set(range(N))
    selection = set()
    selection_length = 0
    while True:
        no_fit = True
        for i in indices - selection:
            if selection_length + sentence_lengths[i] > max_budget:
                continue

            selection_indices = list(selection.union(set([i])))
            if no_fit:    
                best_increment = coverage_score(sentence_embeddings, sentence_embeddings[selection_indices], alpha)
                best_increment /= selection_length + sentence_lengths[i]
                best_id = i
            else:
                increment = coverage_score(sentence_embeddings, sentence_embeddings[selection_indices], alpha)
                increment /= selection_length + sentence_lengths[i]
                if increment > best_increment:
                    best_increment = increment
                    best_id = i
            
            no_fit = False

        selection.add(best_id)
        selection_length += sentence_lengths[best_id]

        if no_fit:
            break

    return selection


