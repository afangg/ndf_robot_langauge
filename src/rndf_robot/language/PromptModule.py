from airobot import log_debug
from .language_utils import MiniLM, identify_classes_from_query

class PromptModule:
    def __init__(self, skill_names, obj_classes) -> None:
        self.skill_names = skill_names   
        self.obj_classes = obj_classes 

    def prompt_user(self):
        '''
        Prompts the user to input a command and finds the demos that relate to the concept.
        Moves to the next state depending on the input

        return: the concept most similar to their query and their input text
        '''
        log_debug('All demo labels: %s' %self.skill_names)
        while True:
            query = self.ask_query()
            # query = "place_relative mug_in tray", "move mug to the tray upside down"
            if not query: return
            best_skill, query_text = query
            break
        return best_skill, query_text.lower()

    def ask_query(self):
        '''
        Prompts the user to input a command and identifies concept

        return: the concept most similar to their query and their input text
        '''
        langInterpreter = MiniLM()

        while True:
            query_text = input('Please enter a query or \'reset\' to reset the scene\n')
            if not query_text: continue
            if query_text.lower() == "reset": return
            ranked_concepts = langInterpreter.find_correspondence(self.skill_names, query_text)
            best_skill = None
            for concept in ranked_concepts:
                print('Corresponding concept:', concept)
                correct = input('Corrent concept? (y/n)\n')
                if correct == 'n':
                    continue
                elif correct == 'y':
                    best_skill = concept
                    break
            
            if best_skill:
                break
        del langInterpreter
        return best_skill, query_text
    
    def get_keywords_and_classes(self, prompt, state):
        best_skill, query_text = prompt
        demo_folder = best_skill[best_skill.find(' ')+1:]
        return identify_classes_from_query(state, 
                                           query_text, 
                                           demo_folder, 
                                           self.obj_classes)
    
    def decompose_demo(self, prompt):
        #ie. grasp mug_handle
        best_skill, _ = prompt
        action, folder_name = best_skill[:best_skill.find(' ')], best_skill[best_skill.find(' ')+1:]
        folder_name = folder_name.split('_')
        _, geometry = folder_name[0], folder_name[1:][0]
        return action, geometry
