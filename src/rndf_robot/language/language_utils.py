from sentence_transformers import SentenceTransformer
from sentence_transformers.util import dot_score
from flair.models import SequenceTagger
from flair.data import Sentence

import torch
import numpy as np
ARTICLES = {'a', 'an', 'the'}

tagger = SequenceTagger.load('flair/chunk-english')

class MiniLM:
    def __init__(self) -> None:
        self.llm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def find_correspondence(self, existing_concepts, query):
        '''
        Finds the cosine similarity between the existing concepts and the input query text.
        Returns the list of concepts sorted in decreasing order of similarity

        return: list of concepts sorted by similarity to the query
        '''

        query = query.replace('_', ' ')
        concept_embeddings = self.llm.encode(existing_concepts, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
        target_embedding= self.llm.encode(query, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
        scores = dot_score(target_embedding, concept_embeddings)
        sorted_scores, idx = torch.sort(scores, descending=True)
        sorted_scores, idx = sorted_scores.flatten(), idx.flatten()
        idx = idx.detach().cpu().numpy()
        return np.array(existing_concepts)[idx]

def chunk_query(self, query):
    '''
    Decomposes a query text into a dictionary mapping phrases to part of speech and returns it
    ex. Phrase tags: adjectival, adverbial, noun phrase, preposition, particle, verb phrase, etc.

    return: {phrase: label}
    '''
    sentence = Sentence(query)
    self.tagger.predict(sentence)
    sentence_dic = {}
    for label in sentence.get_labels():
        sentence_dic[label.data_point.text] = label.value
    return sentence_dic

# this needs to be able to handle multiple instances of the same class
def create_keyword_dic(relevant_objs, sentence_dic):
    '''
    Associates each noun phrase with a object class it corresponds to and returns the pairs

    @relevant_objs: set of object classes 
    @sentence_dic: {phrase: label}

    return: [(obj_class, noun phrase)] 
    '''
    keywords = []
    verb_flag = False
    for phrase, label in sentence_dic.items():
        if label == 'NP': #noun phrase
            for obj in relevant_objs:
                if obj in phrase:
                    split_phrase = phrase.split(' ')
                    phrase = [word for word in split_phrase if word not in ARTICLES]
                    phrase = ' '.join(phrase)
                    keywords.append((obj, phrase, verb_flag))
            if verb_flag:
                verb_flag = not verb_flag

        if label == 'VP':
            verb_flag = True
        
    return keywords

def identify_classes_from_query(query, corresponding_concept, potential_classes):
    '''
    Takes a query and skill concept and identifies the relevant object classes to execute the skill.
    
    @query: english input for the skill
    @corresponding_concept: concept in the form 'grasp/place {concept}'

    returns: the key for the set of demos relating to the concept (just the {concept} part)
    '''
    concept_key = corresponding_concept[corresponding_concept.find(' ')+1:]
    concept_language = frozenset(concept_key.lower().replace('_', ' ').split(' '))
    relevant_classes = concept_language.intersection(potential_classes)
    chunked_query = chunk_query(query)
    keywords = create_keyword_dic(relevant_classes, chunked_query)
    rank_to_class = get_relevent_nouns(keywords)
    return keywords, rank_to_class

def get_relevent_nouns(keywords, state, assigned=None):
    '''
    @test_objs: list of relevant object classes to determine rank for
    @keywords: list of associated obj class, noun phrase, and verb flag as pairs of tuples in form (class, NP, True/False)
    '''
    # what's the best way to determine which object should be manipulated and which is stationary automatically?
    rank_to_class = {}
    if state == 0:
        # only one noun phrase mentioned, probably the object to be moved
        keyword = keywords.pop()
        rank_to_class[0] = {}
        rank_to_class[0]['description'] = keyword[1]
        rank_to_class[0]['potential_class'] = keyword[0]
    else:
        if state == 1 and assigned:
            old_keywords = keywords.copy()
            keywords = []
            if len(old_keywords) >= 1:
                for pair in old_keywords:
                    # check if the obj class mentioned in noun phrase same as object to be moved
                    if pair[0] in assigned:
                        keywords.append(pair)   

        if len(keywords) == 1:
            priority_rank = 0 if 0 not in rank_to_class else 1
            keyword = keywords.pop()
            rank_to_class[priority_rank] = {}
            rank_to_class[priority_rank]['description'] = keyword[1]
            rank_to_class[priority_rank]['potential_class'] = keyword[0]
        else:
            log_warn('There is still more than one noun mentioned in the query')
            if len(keywords) == 2:
                pair_1, pair_2 = keywords
                if pair_1[2]:
                    classes_to_assign = [pair_1, pair_2]
                elif pair_2[2]:
                    classes_to_assign = [pair_2, pair_1]
                else:
                    log_warn("Unsure which object to act on, just going in order of prompt")
                    classes_to_assign = [pair_2, pair_1]

            else:
                classes_to_assign = keywords
            for i in range(len(classes_to_assign)):
                rank_to_class[i] = {}
                rank_to_class[i]['description'] = classes_to_assign[i][1]
                rank_to_class[i]['potential_class'] = classes_to_assign[i][0]

    target = 'Target - class:%s, descr: %s'% (rank_to_class[0]['potential_class'], rank_to_class[0]['description'])
    log_debug(target)
    if 1 in rank_to_class:
        relation = 'Relational - class:%s, descr: %s'% (rank_to_class[1]['potential_class'], rank_to_class[1]['description'])
        log_debug(relation)