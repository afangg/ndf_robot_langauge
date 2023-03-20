from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
from flair.models import SequenceTagger
from flair.data import Sentence

import torch
import numpy as np

llm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
ARTICLES = {'a', 'an', 'the'}

def query_correspondance(existing_concepts, query):
    '''
    Finds the cosine similarity between the existing concepts and the input query text.
    Returns the list of concepts sorted in decreasing order of similarity

    return: list of concepts sorted by similarity to the query
    '''

    #existing_concepts = existing_concepts.replace('_', ' ')
    query = query.replace('_', ' ')
    concept_embeddings = llm.encode(existing_concepts, convert_to_tensor=True)
    target_embedding= llm.encode(query, convert_to_tensor=True)
    scores = pytorch_cos_sim(target_embedding, concept_embeddings)
    sorted_scores, idx = torch.sort(scores, descending=True)
    sorted_scores, idx = sorted_scores.flatten(), idx.flatten()
    idx = idx.detach().cpu().numpy()
    return np.array(existing_concepts)[idx]

def chunk_query(query):
    '''
    Decomposes a query text into a dictionary mapping phrases to part of speech and returns it
    ex. Phrase tags: adjectival, adverbial, noun phrase, preposition, particle, verb phrase, etc.

    return: {phrase: label}
    '''
    tagger = SequenceTagger.load('flair/chunk-english')
    sentence = Sentence(query)
    tagger.predict(sentence)
    sentence_dic = {}
    for label in sentence.get_labels():
        sentence_dic[label.data_point.text] = label.value
    return sentence_dic

# this needs to be handle multiple instances of the same class
def create_keyword_dic(relevent_objs, sentence_dic):
    '''
    Associates each noun phrase with a object class it corresponds to and returns the pairs

    @relevent_objs: set of object classes 
    @sentence_dic: {phrase: label}
obj
    return: [(obj_class, noun phrase)] 
    '''
    keywords = []
    verb_flag = False
    for phrase, label in sentence_dic.items():
        if label == 'NP': #noun phrase
            for obj in relevent_objs:
                if obj in phrase:
                    split_phrase = phrase.split(' ')
                    phrase = [word for word in split_phrase if word not in ARTICLES]
                    phrase = ' '.join(phrase)
                    keywords.append((obj, phrase, verb_flag))
            if verb_flag:
                verb_flag = not verb_flag

        if label == 'VBP':
            verb_flag = True
        
    return keywords
