from sentence_transformers import SentenceTransformer
from sentence_transformers import util as sentence_util
from flair.models import SequenceTagger
from flair.data import Sentence

import torch
import numpy as np

llm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def query_correspondance(existing_concepts, query):
    concept_embeddings = llm.encode(existing_concepts, convert_to_tensor=True)
    target_embedding= llm.encode(query, convert_to_tensor=True)
    scores = sentence_util.pytorch_cos_sim(target_embedding, concept_embeddings)
    sorted_scores, idx = torch.sort(scores, descending=True)
    sorted_scores, idx = sorted_scores.flatten(), idx.flatten()
    idx = idx.detach().cpu().numpy()
    # from IPython import embed; embed()
    return np.array(existing_concepts)[idx]

def chunk_query(query):
    tagger = SequenceTagger.load('flair/chunk-english')
    sentence = Sentence(query)
    tagger.predict(sentence)
    sentence_dic = {}
    for label in sentence.get_labels():
        sentence_dic[label.data_point.text] = label.value
    return sentence_dic

# this needs to be handle multiple instances of the same class
def create_keyword_dic(relevent_objs, sentence_dic):
    keywords = {}
    for phrase, label in sentence_dic.items():
        if label == 'NP': #noun phrase
            for obj in relevent_objs:
                if obj in phrase:
                    keywords[phrase] = obj
    return keywords
