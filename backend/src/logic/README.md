# Logic module

## 1. Lý thuyết cơ bản

Logic module đóng vai trò xử lý các bước sau khi retrieval ban đầu trả về candidate. Mục tiêu là cải thiện chất lượng kết quả bằng cách tách query, hiểu cấu trúc ngữ nghĩa và kết hợp các tín hiệu khác như thời gian, ngữ cảnh và điểm số.

Trong hệ thống retrieval, một câu query thường không chỉ đơn giản là một khái niệm đơn lẻ. Nó có thể gồm:
- nhiều ý nhỏ,
- nhiều sự kiện liên tiếp,
- các mốc thời gian,
- các điều kiện ngữ nghĩa khác nhau.

Do đó, module này giúp chuyển query thành các biểu diễn dễ xử lý hơn.

## 2. Thuật toán và mô hình được ứng dụng

### 2.1 Query parsing
Query parsing giúp tách một câu truy vấn phức tạp thành các phần nhỏ hơn. Ví dụ:
- “người đi xe đạp trên đường phố, sau đó dừng ở trạm xe buýt”
có thể được chia thành các đơn vị ý nghĩa riêng.

Việc này giúp retrieval có thể tìm ra các candidate phù hợp hơn cho từng phần của query.

### 2.2 Temporal search
Temporal search xử lý các truy vấn có yếu tố thời gian hoặc chuỗi sự kiện. Nó không chỉ quan tâm đến độ giống ngữ nghĩa mà còn đến thứ tự và khoảng thời gian giữa các sự kiện.

Đây là một bước quan trọng khi query mô tả một chuỗi hành động liên tiếp.

### 2.3 Scoring và reranking
Sau khi có các candidate từ retrieval, module này dùng logic scoring để điều chỉnh thứ hạng. Việc này có thể dựa trên:
- điểm số từ vector search,
- độ phù hợp temporal,
- độ liên quan ngữ nghĩa,
- context xung quanh frame.

## 3. Input và output của module

### Input
- query từ người dùng,
- candidate từ module retrieval,
- metadata và timestamp của các item.

### Output
- query plan hoặc các phần query con,
- danh sách kết quả đã sắp xếp lại,
- thông tin temporal/context dùng cho API.

## 4. Flow nội bộ

1. Nhận query đầu vào.
2. Tách query thành các phần con hoặc sự kiện.
3. Gọi retrieval hoặc xử lý candidate phù hợp.
4. Tính điểm và điều chỉnh thứ tự kết quả.
5. Trả về kết quả đã được refine cho API.

## 5. Các file chính

- query_parser.py: tách query thành các đơn vị con
- temporal_search.py: xử lý temporal matching
- scoring.py: tính và kết hợp điểm số
- frame_context.py: lấy context xung quanh frame

## 6. Ghi chú phát triển

- Đây là nơi phù hợp để cải tiến ranking và temporal logic.
- Nếu chỉ sửa ở API layer thì sẽ khó bảo trì và khó so sánh chất lượng.
