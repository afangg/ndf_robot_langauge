from sentence_transformers import SentenceTransformer
from sentence_transformers import util as sentence_util
from flair.models import SequenceTagger
from flair.data import Sentence

import torch
import numpy as np
# def process_query(self, existing_concepts, query):
#     concepts = list(self.demo_dic.keys())
#     n = len(concepts)
#     concept_embeddings = self.ll_model.encode(concepts, convert_to_tensor=True)

#     while True:
#         query_text = input('Please enter a query\n')
#         target_embedding= self.ll_model.encode(query_text, convert_to_tensor=True)
#         scores = sentence_util.pytorch_cos_sim(target_embedding, concept_embeddings)
#         sorted_scores, idx = torch.sort(scores, descending=True)
#         sorted_scores, idx = sorted_scores.flatten(), idx.flatten()
#         corresponding_concept = None
#         for i in range(n):
#             print('Corresponding concept:', concepts[idx[i]])
#             query_text = input('Corrent concept? (y/n)\n')
#             if query_text == 'n':
#                 continue
#             elif query_text == 'y':
#                 corresponding_concept = concepts[idx[i]]
#                 break
        
#         if corresponding_concept:
#             break

#     demos = self.demo_dic[corresponding_concept] if corresponding_concept in self.demo_dic else []
#     if not len(demos):
#         log_warn('No demos correspond to the query!')
        
#     log_debug('Number of Demos %s' % len(demos)) 

#     if demos is not None and self.table_model is None:
#         demo_file = np.load(demos[0], allow_pickle=True)
#         if 'table_urdf' in demo_file:
#             self.table_model = demo_file['table_urdf'].item()
#     return demos, corresponding_concept

def query_correspondance(self, existing_concepts, query):
    concept_embeddings = self.ll_model.encode(existing_concepts, convert_to_tensor=True)
    target_embedding= self.ll_model.encode(query, convert_to_tensor=True)
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

