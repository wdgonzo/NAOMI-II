from src.embeddings import EmbeddingModel
model = EmbeddingModel.load('models/trained_embeddings.pkl')
similar_words = model.get_similar_words('car', top_k=10)
print(similar_words)