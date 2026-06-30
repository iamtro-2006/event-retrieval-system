# Rules for Pipeline Development

Tài liệu này quy định cách thiết kế và phát triển pipeline trong backend để hệ thống luôn dễ mở rộng, dễ bảo trì và ít bị phá vỡ khi thay đổi model, thuật toán hoặc thêm chức năng mới.

---

## 1. Nguyên tắc thiết kế chung

### 1.1 Mỗi module chỉ làm một loại việc
Một module trong src/ nên có một trách nhiệm chính rõ ràng.

Ví dụ:
- embeddings: chuyển dữ liệu thành vector
- keyframes: chọn frame đại diện cho video
- retrieval: tìm item gần nhất trong không gian vector
- logic: xử lý query, temporal, scoring, reranking
- utils: tiện ích dùng chung

Không nên để một module vừa đọc video, vừa encode ảnh, vừa build index và vừa trả API.

### 1.2 Pipeline phải có flow tuyến tính và rõ ràng
Một pipeline tốt nên tuân theo cấu trúc:

```text
input -> preprocess -> feature extraction -> indexing -> retrieval -> postprocess -> output
```

Mỗi bước chỉ nhận input đã chuẩn hóa từ bước trước và chỉ sinh output phù hợp cho bước sau.

### 1.3 Không để business logic chui vào API layer
API layer chỉ nên:
- nhận request,
- validate input,
- gọi pipeline/service,
- trả response.

Logic xử lý retrieval, OCR, temporal, reranking phải nằm trong module nghiệp vụ ở src/.

---

## 2. Quy tắc về file và module khi thêm chức năng mới

### 2.1 Khi thêm một chức năng mới, phải tạo đầy đủ các tầng cần thiết
Ví dụ: thêm OCR cho video/keyframe.

Một chức năng mới nên có cấu trúc như sau:

```text
src/ocr/
  __init__.py
  README.md
  models/
    detector.py
    recognizer.py
  pipelines/
    extract_ocr.py
```

### 2.2 Các file mới nên tuân theo một contract cố định
Mỗi file mới nên có 3 phần rõ ràng:
- input: dữ liệu đầu vào cụ thể
- process: thuật toán hoặc mô hình xử lý
- output: dữ liệu đầu ra có schema rõ ràng

Ví dụ cho OCR:
- input: ảnh hoặc frame video
- process: preprocess image -> OCR model -> postprocess text
- output: list[dict] gồm text, confidence, bbox, source_image

### 2.3 Không làm chồng logic giữa module mới và module cũ
Nếu thêm OCR, không nên nhét toàn bộ OCR logic vào retrieval hoặc main.py.

Mọi logic liên quan đến OCR phải nằm ở module OCR, và retrieval chỉ dùng output của OCR khi cần.

---

## 3. Ví dụ thực tế: khi thêm chức năng OCR

### 3.1 Mục tiêu
Thêm khả năng đọc text từ keyframe/video để hỗ trợ tìm kiếm theo nội dung text trong hình.

### 3.2 Các file cần xem xét

#### File mới đề xuất
- src/ocr/models/detector.py
  - phát hiện vùng text trong ảnh
- src/ocr/models/recognizer.py
  - nhận diện ký tự/chuỗi text từ vùng đã detect
- src/ocr/pipelines/extract_ocr.py
  - pipeline tổng hợp chạy OCR cho một batch ảnh
- src/ocr/README.md
  - mô tả lý thuyết, input/output, flow nội bộ

#### File đã có cần cập nhật
- src/embeddings/README.md
  - nếu muốn dùng OCR text embedding hoặc text features riêng
- src/retrieval/README.md
  - nếu muốn index OCR text cùng embedding hình ảnh
- main.py
  - nếu thêm endpoint như /api/ocr/search
- configs/app.yaml hoặc configs/embeddings.yaml
  - thêm cấu hình cho OCR model và threshold

### 3.3 Flow thiết kế chung cho OCR

```text
Input image/frame
  -> preprocess image
  -> detect text regions
  -> recognize text per region
  -> normalize text output
  -> return OCR result + confidence + bbox
```

### 3.4 Mẫu thiết kế dữ liệu đầu ra

```python
{
    "image_path": "...",
    "text": "giấy tờ xe",
    "confidence": 0.94,
    "bbox": [[x1, y1], [x2, y2]],
    "source": "keyframe"
}
```

### 3.5 Quy tắc khi thêm OCR vào pipeline chung
- OCR phải là một bước độc lập, không gắn với retrieval trực tiếp.
- OCR output phải có schema chuẩn và có thể dùng lại ở nhiều nơi.
- Nếu OCR được dùng trong search, nên tạo một lớp trung gian để chuyển OCR result thành feature/indexable data.

---

## 4. Quy tắc cho thuật toán và mô hình

### 4.1 Mỗi thuật toán phải có mục đích rõ ràng
Khi viết một thuật toán, phải trả lời 3 câu hỏi:
- nó giải quyết vấn đề gì?
- đầu vào là gì?
- đầu ra là gì?

### 4.2 Không dùng mô hình quá phức tạp nếu không cần
Nếu một task đơn giản có thể xử lý bằng rule-based hoặc heuristic, ưu tiên cách đơn giản trước.
Nếu task cần hiểu ngữ nghĩa hoặc hình ảnh phức tạp, mới dùng model học sâu.

### 4.3 Mọi mô hình mới phải có fallback path
Ví dụ:
- nếu OCR model fail, có thể dùng heuristic hoặc trả về empty result thay vì crash toàn bộ pipeline.
- nếu indexing backend fail, có thể fallback sang một cấu hình thấp hơn hoặc log error rõ ràng.

---

## 5. Quy tắc cho indexing/vector database

### 5.1 Layer indexing phải độc lập khỏi API và UI
Code indexing không nên biết frontend đang dùng gì.

### 5.2 Không phụ thuộc cứng vào FAISS
Hiện tại dùng FAISS, nhưng code nên thiết kế theo abstraction:

```python
class IndexBackend:
    def add(self, vectors, metadata):
        ...

    def search(self, vector, top_k):
        ...

    def delete(self, ids):
        ...
```

Như vậy, đổi từ FAISS sang Milvus/Qdrant sẽ chỉ cần đổi implementation, không cần đổi toàn bộ pipeline.

### 5.3 Metadata phải luôn đi cùng vector
Mỗi vector phải có metadata đủ để phục vụ truy xuất và hiển thị kết quả:
- video_id
- frame_id
- timestamp
- image path
- source

---

## 6. Quy tắc cho data flow

### 6.1 Dữ liệu giữa module phải có schema rõ ràng
Nên dùng:
- dict
- dataclass
- pandas DataFrame
- typed payload

Không nên truyền dữ liệu theo kiểu tự do hoặc phụ thuộc vào cấu trúc cũ.

### 6.2 Mỗi bước nên có output có thể kiểm tra được
Ví dụ:
- keyframes module output phải có list of keyframe records
- embeddings module output phải có vector array và metadata
- retrieval module output phải có ranked results

### 6.3 Nếu có nhiều bước xử lý, nên có một lớp trung gian
Ví dụ:
- OCR result -> OCR feature builder -> retrieval input

Điều này giúp module tăng tính tách biệt và dễ thay đổi.

---

## 7. Quy tắc cho config và tech stack

### 7.1 Config phải dùng YAML và có cấu trúc rõ ràng
Thông số mới như:
- model name
- threshold
- batch size
- top_k
- device
- precision
- index type

nên được đặt trong config thay vì hardcode trong code.

### 7.2 Tech stack chung nên giữ nhất quán
Backend hiện tại nên ưu tiên các công nghệ sau:
- Python 3.10/3.11
- FastAPI cho API
- PyTorch / OpenCLIP cho model inference
- FAISS cho vector search hiện tại
- NumPy / Pandas cho xử lý dữ liệu
- PyYAML cho config
- OpenCV / Pillow cho ảnh
- Uvicorn cho server runtime

Nếu thay đổi kỹ thuật mới, nên vẫn giữ nguyên nguyên tắc kiến trúc và không làm loạn cấu trúc hiện tại.

---

## 8. Quy tắc kiểm thử và đánh giá

### 8.1 Mỗi module cần có một cách kiểm tra tối thiểu
Ví dụ:
- OCR module: test với một vài ảnh mẫu và so sánh kết quả text
- embeddings module: test output shape và dtype
- retrieval module: test top-k và metadata mapping

### 8.2 Khi đổi model hoặc thuật toán, nên lưu baseline
Cần có:
- output cũ,
- output mới,
- config cũ và mới,
- sample đầu vào.

Đây là cách để đánh giá cải tiến một cách khách quan.

---

## 9. Quy tắc ghi chép khi phát triển

### 9.1 Mỗi module nên có README
README nên nêu rõ:
- lý thuyết cơ bản,
- thuật toán / mô hình,
- input/output,
- flow nội bộ,
- file chính,
- điểm cần chú ý khi nâng cấp.

### 9.2 Khi sửa behavior, cập nhật tài liệu liên quan
Nếu thay đổi pipeline, phải cập nhật:
- README chính,
- README module liên quan,
- cấu hình nếu cần,
- sample hoặc example nếu có.

---

## 10. Kết luận

Một pipeline tốt không chỉ chạy được, mà còn phải:
- có trách nhiệm module rõ ràng,
- có flow dữ liệu rõ ràng,
- có schema rõ ràng,
- dễ thay model/backend,
- dễ thêm chức năng mới mà không làm vỡ hệ thống cũ.

Nói ngắn gọn: khi thêm chức năng mới, hãy luôn thiết kế như một bước mới trong pipeline, không phải một mớ logic rời rạc được cắm vào đâu đó.
