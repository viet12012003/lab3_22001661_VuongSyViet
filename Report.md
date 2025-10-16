# Báo cáo Lab 3 – Word Embeddings with Word2Vec

## 1. Các bước thực hiện

- **[Task 1] Tải và sử dụng model có sẵn (Gensim)**
  - Cài thư viện theo `requirements.txt` (đã pin phiên bản ổn định cho Windows/Python 3.11).
  - Tạo lớp `WordEmbedder` tại `src/representations/word_embedder.py` để:
    - Tải model pretrained `glove-wiki-gigaword-50` qua `gensim.downloader.load()`.
    - Triển khai API khám phá: `get_vector()`, `get_similarity()`, `get_most_similar()` tương thích Gensim 4 (hỗ trợ cả `KeyedVectors` và `Word2Vec`).

- **[Task 2] Nhúng câu/văn bản**
  - Cài đặt `embed_document(document, tokenizer=None)`: ưu tiên dùng `SimpleTokenizer` (Lab 1), bỏ qua OOV, lấy trung bình vector các từ để tạo vector văn bản.

- **[Task 3] Huấn luyện Word2Vec nhỏ (Gensim)**
  - Đọc `data/en_ewt-ud-train.txt` (đường dẫn tuyệt đối trong `test/lab4_test.py`).
  - Huấn luyện với `train_word2vec_small()`, sau đó `save_model()` và `load_model()` để kiểm tra.

- **[Task 4] Huấn luyện Word2Vec lớn (Spark MLlib)**
  - Kiểm tra PySpark, tạo `SparkSession`.
  - Đọc shard JSON `c4-train.00000-of-01024-30K.json`, giữ cột `text`, `na.drop()`, tách `tokens` bằng `split`.
  - Huấn luyện `SparkWord2Vec`, truy xuất vector của từ `'king'` từ `model_spark.getVectors()`.

- **[Task 5] Trực quan hóa**
  - Lấy ~200 từ đầu của vocab, dùng PCA 2D (`sklearn`) và vẽ scatter plot (`matplotlib`) kèm nhãn.

- **Lưu kết quả đầu ra**
  - `output/king_vec_glove.npy` (vector từ pretrained), `output/doc_vec.npy` (nhúng câu)
  - `output/similarities.txt` (độ tương đồng), `output/most_similar_computer.txt` (từ gần nghĩa)
  - `output/w2v_small.model` (model nhỏ), `output/king_vec_small.npy` (vector `'king'` sau khi load model nhỏ)
  - `output/spark_king_vec.txt` (vector `'king'` từ Spark Word2Vec)
  - `output/embeddings_pca.png` (hình trực quan hóa PCA)

---

## 2. Hướng dẫn chạy code

- **Chuẩn bị môi trường (khuyến nghị Python 3.11)**
```powershell
# Tạo và kích hoạt venv (ví dụ .venv311)
py -3.11 -m venv F:\NLP\.venv311
F:\NLP\.venv311\Scripts\activate

# Cài thư viện
pip install --upgrade pip setuptools wheel
pip install -r F:\NLP\lab3_22001661_VuongSyViet\requirements.txt
```

- **Chuẩn bị dữ liệu**
  - `F:/NLP/lab3_22001661_VuongSyViet/data/en_ewt-ud-train.txt`
  - `F:/NLP/lab3_22001661_VuongSyViet/data/c4-train.00000-of-01024-30K.json` (để chạy Task 4; có thể dùng file JSON khác cùng schema)

- **Chạy script**
```powershell
python F:\NLP\lab3_22001661_VuongSyViet\test\lab4_test.py
```
- Kết quả nằm trong `F:/NLP/lab3_22001661_VuongSyViet/output/`.

---

## 3. Phân tích kết quả

- **Pretrained (GloVe 50d) – Similarity & Most Similar**
  - Ví dụ log:
    - `similarity('king','queen') ≈ 0.784` (cao, hợp lý – cùng miền "hoàng gia").
    - `similarity('king','man') ≈ 0.531` (thấp hơn “king–queen”), phản ánh khác biệt vai trò/giới.
    - Top từ gần `computer`: `computers, software, technology, internet, computing, devices, ...` → phù hợp ngữ cảnh công nghệ.
  - Nhận xét: Pretrained GloVe học trên Wikipedia/Gigaword nên giàu thông tin ngữ nghĩa, kết quả lân cận mượt mà.

- **Nhúng văn bản (trung bình vector)**
  - Câu “The queen rules the country.” → vector trung bình của các từ đã biết; là baseline tốt cho so sánh câu, phân cụm.

- **Trực quan hóa PCA 2D**
  - Hình `output/embeddings_pca.png` cho thấy các từ gần nhau theo chủ đề (ví dụ cụm công nghệ).
  - PCA tuyến tính và chỉ 2D nên có thể chồng chéo; tuy nhiên các lân cận gần thường phản ánh đúng quan hệ ngữ nghĩa.

- **So sánh pretrained và model tự huấn luyện**
  - Pretrained: chất lượng cao, bao phủ tốt, lân cận hợp lý.
  - Model nhỏ (Gensim): dữ liệu ít nên chất lượng thấp hơn, phù hợp minh họa pipeline huấn luyện và lưu/tải.
  - Spark Word2Vec (trên shard C4): quy mô lớn hơn so với model nhỏ → biểu diễn phong phú hơn; phụ thuộc preprocessing và phạm vi dữ liệu.

---

## 4. Khó khăn và giải pháp

- **Cài đặt thư viện khoa học trên Windows**
  - Lỗi build `numpy/scipy` trên Python quá mới (3.13). Giải pháp: dùng Python 3.11 + pin phiên bản (`numpy==1.26.4`, `scipy==1.11.4`) để có wheel sẵn; cập nhật `requirements.txt` ổn định.

- **Xung đột không gian tên `src` giữa Lab1 và Lab3**
  - `SimpleTokenizer` (Lab1) import `src.core.interfaces` dễ trỏ nhầm sang `lab3/src`.
  - Giải pháp: import tạm thời bằng cách chèn `LAB1_ROOT` (thư mục cha của `src` Lab1) vào `sys.path`, import `src.preprocessing.simple_tokenizer.SimpleTokenizer`, sau đó khôi phục `sys.path`.

- **Spark trên Windows**
  - Cảnh báo `winutils.exe`/native-hadoop là bình thường khi chạy local; không ảnh hưởng kết quả.
  - Dữ liệu C4 rất lớn → dùng shard nhỏ để chạy thử; đồng bộ điều kiện kiểm tra file với đường dẫn thực tế.

- **Quản lý kết quả**
  - Thêm cơ chế ghi ra `output/`: vector `.npy`, log `.txt`, model `.model`, hình `.png` phục vụ theo dõi và viết báo cáo.

---

## 5. Trích dẫn tài liệu

- **Gensim** (API & pretrained models)
  - Gensim Documentation: https://radimrehurek.com/gensim/
  - Gensim Data (glove-wiki-gigaword-50): https://github.com/RaRe-Technologies/gensim-data
- **Scikit-learn**: PCA
  - https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- **PySpark**: MLlib Word2Vec
  - https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Word2Vec.html
- **Datasets**
  - Universal Dependencies English EWT: https://universaldependencies.org/
  - C4 (Colossal Clean Crawled Corpus): https://www.tensorflow.org/datasets/catalog/c4

---

## 6. Cấu trúc thư mục & mã nguồn chính

- **Mã nguồn**
  - `src/representations/word_embedder.py`: lớp `WordEmbedder` (tải pretrained, API khám phá, nhúng văn bản, train nhỏ, lưu/tải model).
  - `test/lab4_test.py`: kịch bản chạy tuần tự Tasks 1–5, lưu kết quả vào `output/`.
- **Kết quả**
  - `output/`: chứa `.npy`, `.txt`, `.model`, `.png` tương ứng các phần trên.