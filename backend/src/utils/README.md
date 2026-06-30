# Utils module

## 1. Lý thuyết cơ bản

Utils module cung cấp các thành phần nền tảng cho toàn bộ hệ thống. Nó không trực tiếp thực hiện retrieval hay embedding, nhưng lại là lớp hỗ trợ quan trọng để các module khác chạy đúng, nhất quán và dễ bảo trì.

Trong một hệ thống ML/AI, các tiện ích chung như:
- config loading,
- logging,
- seed control,
- device selection,
- file và video I/O,

là những phần rất cần được tách riêng để tránh lặp code và tạo sự phụ thuộc không cần thiết.

## 2. Thuật toán và mô hình được ứng dụng

Module này chủ yếu dùng các kỹ thuật và cấu trúc hỗ trợ, không phải các mô hình học sâu. Tuy nhiên, các hàm trong module vẫn đóng vai trò quan trọng cho toàn bộ pipeline:

- config parsing: đọc cấu hình YAML và chuẩn hóa tham số,
- logger setup: ghi log để theo dõi pipeline,
- seed fixing: đảm bảo kết quả có thể tái lập khi cần,
- video IO helpers: đọc và chuẩn hóa video để đưa vào mô hình phát hiện cảnh hoặc embedding.

## 3. Input và output của module

### Input
- file cấu hình YAML,
- đường dẫn video và dữ liệu,
- các tham số runtime như device, batch size, log level.

### Output
- object config dùng trong pipeline,
- logger instance,
- device object CPU/GPU,
- helper functions để đọc video và xử lý đường dẫn.

## 4. Flow nội bộ

1. Đọc và chuẩn hóa cấu hình đầu vào.
2. Khởi tạo logging và random seed.
3. Cung cấp các hàm tiện ích cho module khác dùng lại.

## 5. Các file chính

- config.py: đọc cấu hình hệ thống
- logger.py: thiết lập logging
- video_io.py: đọc và chuẩn hóa video
- device.py: chọn device phù hợp
- seed.py: kiểm soát randomness

## 6. Ghi chú phát triển

- Module này nên được giữ ổn định vì nhiều module khác phụ thuộc vào nó.
- Khi thêm helper mới, cần đảm bảo nó không làm tăng phụ thuộc vòng tròn giữa các module.
