# embedding_model.py
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "intfloat/multilingual-e5-small"

model = SentenceTransformer(EMBED_MODEL)

def get_model():
    return model
