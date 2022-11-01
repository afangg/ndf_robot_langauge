from sentence_transformers import SentenceTransformer, util
sentences = ['hang', 'hang the mug', 'hung', 'grasp', 'pick', 'hang up', 'hang the cup', 'pick up the mug', 'hang up the mug']
target = 'hang mug'
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#Compute embedding for both lists
target_embedding= model.encode(target, convert_to_tensor=True)
for sentence in sentences:
    embedding = model.encode(sentence, convert_to_tensor=True)
    score = util.pytorch_cos_sim(target_embedding, embedding)
    print('%s is %s similar to %s' % (target, float(score), sentence))
