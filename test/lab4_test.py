import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Thư mục lưu kết quả
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

from src.representations.word_embedder import WordEmbedder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

try:
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import Word2Vec as SparkWord2Vec
    from pyspark.sql.functions import split, col
    _spark_available = True
except Exception:
    _spark_available = False

# Task 1 & 2: Pre-trained embeddings and document embedding
print("=== Task 1 & 2: Pre-trained embeddings and document embedding ===")
embedder = WordEmbedder('glove-wiki-gigaword-50')

# Get vector
king_vec = embedder.get_vector('king')
print("Vector of 'king':", king_vec, "...")
np.save(os.path.join(OUTPUT_DIR, 'king_vec_glove.npy'), king_vec)

# Similarity
sim_king_queen = embedder.get_similarity('king', 'queen')
sim_king_man = embedder.get_similarity('king', 'man')
print("Similarity king-queen:", sim_king_queen)
print("Similarity king-man:", sim_king_man)
with open(os.path.join(OUTPUT_DIR, 'similarities.txt'), 'w', encoding='utf-8') as f:
    f.write(f"king-queen: {sim_king_queen}\n")
    f.write(f"king-man: {sim_king_man}\n")

# Most similar words
top_computer = embedder.get_most_similar('computer')
print("Top 10 similar to 'computer':", top_computer)
with open(os.path.join(OUTPUT_DIR, 'most_similar_computer.txt'), 'w', encoding='utf-8') as f:
    for w, score in top_computer:
        f.write(f"{w}\t{score}\n")

# Document embedding
sentence = "The queen rules the country."
doc_vec = embedder.embed_document(sentence)
print("Document vector:", doc_vec, "...")
np.save(os.path.join(OUTPUT_DIR, 'doc_vec.npy'), doc_vec)

# Task 3: Train Word2Vec on small dataset
print("\n=== Task 3: Train Word2Vec on small dataset ===")
data_path = r'F:\NLP\lab3_22001661_VuongSyViet\data\en_ewt-ud-train.txt'
sentences = []
if os.path.isfile(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                sentences.append(tokens)

# Train Word2Vec
small_model = embedder.train_word2vec_small(sentences)
print("Small Word2Vec trained.")

# Save & reload
small_model_path = os.path.join(OUTPUT_DIR, "w2v_small.model")
embedder.save_model(small_model_path)
embedder.load_model(small_model_path)
print("Loaded model vector for 'king':", embedder.get_vector('king'), "...")
np.save(os.path.join(OUTPUT_DIR, 'king_vec_small.npy'), embedder.get_vector('king'))

# Task 4: Train Word2Vec on large dataset with Spark
print("\n=== Task 4: Train Word2Vec on large dataset with Spark ===")
json_path = r'F:\NLP\lab3_22001661_VuongSyViet\data\c4-train.00000-of-01024-30K.json'
if _spark_available and os.path.isfile(json_path):
    spark = SparkSession.builder.appName("Word2Vec-Large").getOrCreate()
    df = spark.read.json(json_path)
    df = df.select("text").na.drop()

    # Simple tokenization
    df = df.withColumn("tokens", split(col("text"), " "))

    # Train Word2Vec
    word2Vec_spark = SparkWord2Vec(vectorSize=100, minCount=5, inputCol="tokens", outputCol="result")
    model_spark = word2Vec_spark.fit(df)

    # Retrieve embedding of 'king'
    king_vec_spark = model_spark.getVectors().filter(col("word") == "king").collect()
    print("Spark Word2Vec vector for 'king':", king_vec_spark)
    with open(os.path.join(OUTPUT_DIR, 'spark_king_vec.txt'), 'w', encoding='utf-8') as f:
        f.write(str(king_vec_spark))
    try:
        spark.stop()
    except Exception:
        pass
else:
    if not _spark_available:
        print("[Info] PySpark not available. Skipping Task 4.")
    else:
        print("[Info] data/c4-train.json not found. Skipping Task 4.")

# Task 5: Visualization
print("\n=== Task 5: Visualization of embeddings ===")
if hasattr(embedder.model, 'wv'):
    vocab_source = embedder.model.wv
else:
    vocab_source = embedder.model
words = list(vocab_source.index_to_key[:200])  # Top 200 words
vectors = [vocab_source[word] for word in words]

# PCA to 2D
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# Scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
for i, word in enumerate(words):
    plt.text(vectors_2d[i, 0]+0.01, vectors_2d[i, 1]+0.01, word, fontsize=9)
plt.title("Word Embedding Visualization (PCA)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'embeddings_pca.png'), dpi=200)
plt.show()