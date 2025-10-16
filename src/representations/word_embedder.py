import gensim.downloader as api
from gensim.models import Word2Vec
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

LAB1_ROOT = os.path.join(PROJECT_ROOT, "lab1_22001661_VuongSyViet")
_orig_sys_path = list(sys.path)
try:
    if LAB1_ROOT not in sys.path:
        sys.path.insert(0, LAB1_ROOT)
    from src.preprocessing.simple_tokenizer import SimpleTokenizer
finally:
    sys.path = _orig_sys_path

class WordEmbedder:
    def __init__(self, model_name: str = None):
        """
        Initialize WordEmbedder.
        If model_name is provided, load pre-trained embeddings.
        """
        self.model = None
        self.vector_size = 0
        if model_name:
            self.model = api.load(model_name)
            # For KeyedVectors from gensim.downloader
            self.vector_size = getattr(self.model, "vector_size", 0)

    # Task 1: Pre-trained embeddings
    def get_vector(self, word: str) -> np.ndarray:
        """Return embedding vector of a word, zero vector if OOV."""
        if not self.model or self.vector_size == 0:
            return np.zeros(self.vector_size)

        # If model is a Word2Vec instance, vectors live under .wv
        if hasattr(self.model, "wv"):
            if word in self.model.wv.key_to_index:
                return self.model.wv[word]
            return np.zeros(self.vector_size)
        # Otherwise assume KeyedVectors
        if hasattr(self.model, "key_to_index") and word in self.model.key_to_index:
            return self.model[word]
        return np.zeros(self.vector_size)

    def get_similarity(self, word1: str, word2: str) -> float:
        """Compute cosine similarity between two words."""
        if not self.model or self.vector_size == 0:
            return 0.0

        try:
            if hasattr(self.model, "wv"):
                if word1 in self.model.wv.key_to_index and word2 in self.model.wv.key_to_index:
                    return float(self.model.wv.similarity(word1, word2))
                return 0.0
            # KeyedVectors
            if hasattr(self.model, "key_to_index"):
                if word1 in self.model.key_to_index and word2 in self.model.key_to_index:
                    return float(self.model.similarity(word1, word2))
            return 0.0
        except Exception:
            return 0.0

    def get_most_similar(self, word: str, top_n: int = 10):
        """Return top N most similar words."""
        if not self.model or self.vector_size == 0:
            return []
        try:
            if hasattr(self.model, "wv"):
                if word in self.model.wv.key_to_index:
                    return self.model.wv.most_similar(word, topn=top_n)
                return []
            if hasattr(self.model, "key_to_index") and word in self.model.key_to_index:
                return self.model.most_similar(word, topn=top_n)
            return []
        except Exception:
            return []

    # Task 2: Document embedding
    def embed_document(self, document: str, tokenizer=None) -> np.ndarray:
        """
        Embed a sentence/document by averaging word vectors.
        Returns zero vector if no known words.
        """
        if not self.model:
            raise ValueError("Model not loaded.")

        if tokenizer is None:
            tokenizer = SimpleTokenizer()

        if tokenizer is not None and hasattr(tokenizer, "tokenize"):
            tokens = tokenizer.tokenize(document)
        else:
            tokens = document.lower().split()

        vectors = []
        for token in tokens:
            vec = self.get_vector(token)
            if vec is not None and vec.shape[0] == self.vector_size and np.any(vec):
                vectors.append(vec)

        if not vectors:
            return np.zeros(self.vector_size)
        return np.mean(vectors, axis=0)

    # Task 3: Train Word2Vec on small dataset
    def train_word2vec_small(self, sentences: list, vector_size=100, window=5, min_count=1, sg=1):
        """
        Train Word2Vec on a list of tokenized sentences (small dataset).
        """
        self.model = Word2Vec(sentences, vector_size=vector_size, window=window,
                              min_count=min_count, sg=sg)
        self.vector_size = vector_size
        return self.model

    def save_model(self, path: str):
        """Save the trained model to disk."""
        if self.model:
            self.model.save(path)

    def load_model(self, path: str):
        """Load a trained Word2Vec model from disk."""
        self.model = Word2Vec.load(path)
        # Word2Vec exposes vector size under .wv
        if hasattr(self.model, "wv"):
            self.vector_size = self.model.wv.vector_size
        else:
            self.vector_size = getattr(self.model, "vector_size", 0)