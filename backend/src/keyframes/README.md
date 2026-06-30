# Keyframes module

## 1. Lý thuyết cơ bản

Keyframe là một frame đại diện cho một đoạn video hoặc một cảnh nhất định. Thay vì lưu toàn bộ khung hình của video, hệ thống chỉ giữ các frame quan trọng để giảm dung lượng, tăng tốc truy xuất và làm đơn giản hóa việc biểu diễn nội dung video.

Ý tưởng chính là:
- không phải mọi frame đều có giá trị như nhau,
- một cảnh video thường có nhiều frame lặp lại hoặc ít thông tin mới,
- chỉ cần chọn một vài frame đại diện cho mỗi đoạn cảnh.

Việc chọn keyframe giúp hệ thống:
- giảm số lượng dữ liệu phải index,
- làm cho retrieval nhanh và hiệu quả hơn,
- giữ được nội dung quan trọng của video.

## 2. Thuật toán và mô hình được ứng dụng

Module này sử dụng hai lớp kỹ thuật chính:

### 2.1 Scene detection
Scene detection dùng để chia video thành các đoạn cảnh có ý nghĩa. Mục tiêu là xác định những vùng video mà nội dung tương đối ổn định, thay vì chia theo frame đơn lẻ.

Trong hệ thống hiện tại, module này dùng TransNetV2, một mô hình học sâu được huấn luyện để phát hiện boundary giữa các cảnh trong video.

### 2.2 Keyframe selection
Sau khi có các scene, cần chọn frame tiêu biểu cho mỗi scene. Điều này thường dựa trên các tiêu chí như:
- độ khác biệt hình ảnh giữa các frame,
- histogram similarity,
- khoảng cách giữa các frame trong một scene,
- chất lượng hình ảnh và tính đại diện.

Hệ thống hiện tại chọn keyframe bằng cách đánh giá các frame trong scene và giữ những frame có tính phân biệt cao, giảm các frame trùng lặp.

## 3. Input và output của module

### Input
- video gốc từ data/raw/videos,
- cấu hình trong configs/kf_extraction.yaml,
- trọng số mô hình TransNetV2.

### Output
- file scene detection (.scenes.txt),
- file map keyframe (.csv),
- ảnh keyframe đã được lưu,
- metadata cho bước embedding và retrieval tiếp theo.

## 4. Flow nội bộ

1. Đọc video đầu vào.
2. Dùng TransNetV2 phát hiện scene.
3. Xác định các scene và độ dài tương ứng.
4. Chọn frame đại diện cho từng scene.
5. Lưu ảnh keyframe và bản đồ mapping.

## 5. Các file chính

- models/detector.py: phát hiện scene bằng TransNetV2
- models/selector.py: lựa chọn keyframe
- pipelines/extract_keyframes.py: pipeline tổng hợp

## 6. Ghi chú phát triển

- Nếu thay đổi tiêu chí chọn keyframe, ảnh hưởng trực tiếp đến chất lượng retrieval.
- Nếu tăng/giảm ngưỡng scene detection, cần đánh giá lại trên sample dữ liệu.
- Output của module này là input quan trọng cho module embeddings.
