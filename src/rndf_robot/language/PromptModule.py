from airobot import log_debug
from .language_utils import MiniLM

class PromptModule:
    def __init__(self, skill_names) -> None:
        self.langInterpreter = MiniLM()
        self.skill_names = skill_names    

    def prompt_user(self):
        '''
        Prompts the user to input a command and finds the demos that relate to the concept.
        Moves to the next state depending on the input

        return: the concept most similar to their query and their input text
        '''
        log_debug('All demo labels: %s' %self.skill_names)
        while True:
            query = self.ask_query()
            # query = "grasp mug_handle", "grab the mug by the handle"
            if not query: return
            corresponding_concept, query_text = query
            break
        return corresponding_concept, query_text

    def ask_query(self):
        '''
        Prompts the user to input a command and identifies concept

        return: the concept most similar to their query and their input text
        '''
        while True:
            query_text = input('Please enter a query or \'reset\' to reset the scene\n')
            if not query_text: continue
            if query_text.lower() == "reset": return
            ranked_concepts = self.langInterpreter.find_correspondence(self.skill_names, query_text)
            corresponding_concept = None
            for concept in ranked_concepts:
                print('Corresponding concept:', concept)
                correct = input('Corrent concept? (y/n)\n')
                if correct == 'n':
                    continue
                elif correct == 'y':
                    corresponding_concept = concept
                    break
            
            if corresponding_concept:
                break
        return corresponding_concept, query_text
    
    def get_keywords(self, query_text, corresponding_skill, state):
        return self.langInterpreter.identify_classes_from_query(state, query_text, corresponding_skill)