from sentence_transformers import SentenceTransformer, util
sentences = ['hang', 'hang cup', 'cup hand', 'hang the mug', 'hung', 'grasp', 'pick', 'hang up', 'hang the cup', 'pick up the mug', 'hang up the mug']
sentences = ['pick up the mug']
target = 'hang the cup'
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#Compute embedding for both lists
target_embedding= model.encode(target, convert_to_tensor=True)
for sentence in sentences:
    embedding = model.encode(sentence, convert_to_tensor=True)
    #print('Embedding for %s: %s' % (sentence, embedding)) 
    score = util.pytorch_cos_sim(target_embedding, embedding)
    print('%s is %s similar to %s' % (target, float(score), sentence))
