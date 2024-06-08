# 문장 유사도
# https://sbert.net/docs/sentence_transformer/pretrained_models.html : multilingual
# step1
from sentence_transformers import SentenceTransformer

# step2
model = SentenceTransformer("all-MiniLM-L6-v2")

# step3
sentences = [
    "The weather is lovely today.",
    "It's so dark outside!",
    "He drove to the stadium.",
]
sentence1 = sentences[0]
sentence2 = sentences[1]

# step4
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)
print(embedding1.shape)
print(embedding2.shape)

# step5
similarities = model.similarity(embedding1, embedding2)
print(similarities)