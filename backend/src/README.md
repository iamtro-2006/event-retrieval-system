# src module overview

Thư mục src/ là nơi chứa toàn bộ logic nghiệp vụ của backend. Mỗi submodule có trách nhiệm riêng trong pipeline từ video đến kết quả truy xuất.

## Cấu trúc tổng quát

- embeddings/: xử lý embedding hình ảnh/text
- keyframes/: phát hiện scene, chọn keyframe và lưu metadata
- retrieval/: xây dựng indexing/vector database và thực hiện retrieval
- logic/: tách query, temporal matching, scoring, reranking hỗ trợ
- utils/: tiện ích dùng chung như config, logger, video I/O
- asr/, ocr/, reranker/: các module mở rộng cho các dạng tín hiệu khác

## Nguyên tắc sử dụng

Mỗi module nên được xem như một thành phần độc lập với:
- input rõ ràng
- output rõ ràng
- config rõ ràng
- dependency rõ ràng

Không nên module nào phụ thuộc trực tiếp vào output dạng UI hoặc frontend-specific structure.

---

## Flow chung qua các module

```text
video input
  -> keyframes module
  -> embeddings module
  -> retrieval module
  -> logic module
  -> API response
```

---

## Lưu ý khi đọc và mở rộng

- Nếu bạn muốn hiểu một bước cụ thể trong pipeline, hãy bắt đầu từ module tương ứng.
- Nếu muốn đổi indexing backend, tập trung vào retrieval/.
- Nếu muốn đổi model embedding, tập trung vào embeddings/.
- Nếu muốn đổi cách chọn keyframe, tập trung vào keyframes/.
