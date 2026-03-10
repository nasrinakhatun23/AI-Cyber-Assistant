from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(text_list):
    embeddings = model.encode(text_list)
    return embeddings