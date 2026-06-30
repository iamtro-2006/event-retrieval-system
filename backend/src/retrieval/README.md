# Retrieval module

## 1. Lý thuyết cơ bản

Retrieval là quá trình tìm các tài liệu hoặc item có liên quan nhất với một truy vấn đầu vào. Trong hệ thống này, retrieval hoạt động trên không gian vector: mỗi item (keyframe) và mỗi query được biểu diễn dưới dạng vector embedding, sau đó hệ thống tìm những vector gần nhau nhất.

Điểm cốt lõi của retrieval là: thay vì khớp bằng từ khóa thủ công, ta dùng khoảng cách vector để đo mức độ tương đồng ngữ nghĩa.

### Các khái niệm nền tảng
- vector database: hệ thống lưu trữ và truy vấn vector hiệu quả,
- nearest neighbor search: tìm các vector gần nhất với một vector truy vấn,
- cosine similarity: độ đo mức độ giống nhau giữa hai vector,
- ANN (Approximate Nearest Neighbor): kỹ thuật tìm gần đúng để tăng tốc khi dữ liệu lớn.

## 2. Thuật toán và mô hình được ứng dụng

### 2.1 Vector indexing
Sau khi có embedding, hệ thống cần một cơ chế lưu trữ và truy vấn nhanh. Hiện tại hệ thống dùng FAISS, một thư viện nổi tiếng cho ANN search.

FAISS hỗ trợ các kỹ thuật như:
- IVF
- HNSW
- PQ
- flat index

Trong project hiện tại, cấu hình đang dùng HNSW-style indexing với cosine similarity.

### 2.2 Cosine similarity
Độ đo này rất phù hợp cho embedding vì nó tập trung vào hướng của vector, không bị ảnh hưởng quá nhiều bởi độ lớn của vector.

### 2.3 Temporal search
Ngoài retrieval ngữ nghĩa, hệ thống còn có logic tìm kiếm theo thời gian, ví dụ một chuỗi sự kiện xảy ra liên tiếp. Đây là một lớp xử lý bổ sung sau khi có candidate ban đầu.

## 3. Input và output của module

### Input
- embedding đã tạo ở module embeddings,
- metadata về video_id, frame_id, path ảnh/video,
- query đầu vào từ API hoặc frontend.

### Output
- danh sách kết quả retrieval ranked theo độ tương đồng,
- metadata đi kèm cho mỗi kết quả,
- thông tin như timestamp, đường dẫn ảnh, video và context.

## 4. Flow nội bộ

1. Thu thập embedding và metadata.
2. Xây dựng ma trận vector và mapping metadata.
3. Tạo index/vector database.
4. Khi nhận query, encode query thành vector.
5. Tìm các vector gần nhất.
6. Kết hợp kết quả với metadata và trả về payload cho API.

## 5. Các file chính

- models/faiss_index.py: build matrix, metadata và index
- models/retrieval_system.py: engine retrieval chính
- pipelines/build_faiss.py: pipeline build index
- pipelines/build_vector_cache.py: tạo cache vector cho temporal search

## 6. Ghi chú phát triển

- Hiện tại dùng FAISS, nhưng kiến trúc nên giữ ở mức abstraction để sau này có thể đổi sang Milvus, Qdrant hoặc backend khác.
- Nếu đổi backend indexing, ưu tiên sửa ở module này mà không làm ảnh hưởng frontend/API.
