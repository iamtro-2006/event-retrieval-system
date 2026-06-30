# Embeddings module

## 1. Lý thuyết cơ bản

Embedding là cách biểu diễn dữ liệu như một vector số có số chiều cố định để máy tính có thể so sánh và tìm kiếm theo độ tương đồng. Trong hệ thống này, mỗi keyframe hoặc query được chuyển thành một vector đặc trưng, rồi các vector này được dùng để tìm các item có ý nghĩa gần nhau.

Về mặt ý nghĩa, mục tiêu là biến ảnh và text thành một không gian chung mà trong đó:
- các đối tượng tương tự về ngữ nghĩa nằm gần nhau,
- các đối tượng khác nhau nằm xa nhau,
- khoảng cách hoặc độ tương đồng giữa các vector có thể dùng để ranking kết quả.

### Vì sao cần embedding?
Nếu chỉ dùng tên file, metadata thô, hoặc từ khóa đơn giản thì khó xử lý được các truy vấn ngữ nghĩa như:
- “người chạy ngoài trời trong mưa”
- “xe đạp đi qua đường phố”
- “cảnh có nhiều người tụ tập”

Embedding giúp hệ thống hiểu được mối liên hệ ngữ nghĩa giữa query và dữ liệu thay vì chỉ khớp literal text.

## 2. Thuật toán và mô hình được ứng dụng

Module này chủ yếu sử dụng mô hình đa modality dựa trên CLIP/OpenCLIP.

### 2.1 Mô hình hình ảnh và text
Mô hình OpenCLIP học được một không gian chung cho ảnh và text. Nhờ đó:
- ảnh keyframe có thể được encode thành vector,
- câu query có thể được encode thành vector,
- độ tương đồng giữa hai vector có thể tính bằng cosine similarity.

### 2.2 Cosine similarity
Đây là độ đo phổ biến trong retrieval vector:

$$
\text{cosine\_similarity}(u, v) = \frac{u \cdot v}{\|u\|\|v\|}
$$

Giá trị càng gần 1 thì hai vector càng giống nhau về ngữ nghĩa.

### 2.3 Batch encoding
Để tăng tốc, module xử lý dữ liệu theo batch thay vì encode từng ảnh riêng lẻ. Điều này giúp giảm overhead và tăng hiệu suất trên GPU/CPU.

## 3. Input và output của module

### Input
- danh sách ảnh keyframe hoặc frame từ video,
- cấu hình model trong configs/embeddings.yaml,
- đường dẫn đến thư mục ảnh/video đầu vào.

### Output
- file embedding cho từng frame/keyframe,
- metadata đi kèm như đường dẫn ảnh, video_id, frame_id,
- các file này sau đó được dùng bởi module retrieval.

## 4. Flow nội bộ

1. Đọc danh sách input ảnh hoặc frame.
2. Tải mô hình embedding phù hợp.
3. Tiền xử lý ảnh (resize, normalize, chuyển tensor).
4. Encode batch ảnh bằng mô hình.
5. Lưu embedding ra file và đưa kèm metadata.

## 5. Các file chính

- models/embedder.py: load mô hình và encode ảnh/text
- models/encoder.py: wrapper cho việc encode frame từ video
- pipelines/extract_embeddings.py: pipeline tổng hợp cho quá trình tạo embedding

## 6. Ghi chú phát triển

- Nếu thay đổi mô hình embedding, module này là nơi đầu tiên cần cập nhật.
- Nếu đổi định dạng output embedding, retrieval layer cũng phải được điều chỉnh cho phù hợp.
- Khi cần tăng hiệu năng, ưu tiên tối ưu batch size, precision và device selection.
