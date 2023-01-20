from sentence_transformers import SentenceTransformer, util
import torch
sentences = ['place the mug', 'grasp mug', 'hang mug', 'push mug', 'drop mug']

# sentences = ['put down the mug']
target = 'pick up the mug'
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#Compute embedding for both lists
target_embedding= model.encode(target, convert_to_tensor=True)
concept_embeddings = model.encode(sentences, convert_to_tensor=True)
scores = util.pytorch_cos_sim(target_embedding, concept_embeddings)
sorted_scores, idx = torch.sort(scores, descending=True)
sorted_scores, idx = sorted_scores.flatten(), idx.flatten()

for i in range(len(sentences)):
    #print('Embedding for %s: %s' % (sentence, embedding)) 
    print('%s is %s similar to %s' % (target, float(sorted_scores[i]), sentences[idx[i]]))
