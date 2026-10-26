# Notebooks

- `vlm_keyframes.ipynb`: tải từng video từ Drive, OmniShotCut → lấy mẫu 3 FPS → lọc chất lượng trên GPU → PE-Core-L14-336 → clustering → anchor theo shot + MMR → dedup 0,97 theo độ mới và độ nét → JPEG/ZIP → upload → cleanup.
- `compress_video.ipynb`: tải từng video từ Drive nguồn, NVDEC → CUDA resize → HEVC NVENC 480p, upload vào `videos/` của Drive đích. Cell cuối in tiến độ rclone và FFmpeg.
- `extract_embedding.ipynb`: chọn `DATASET=V3C` hoặc `DATASET=LHE`, trích xuất Qwen3-VL-Embedding-2B qua vLLM từ ZIP keyframe và upload ZIP vector.
- `extract_ocr_paddle.ipynb`: tải ZIP keyframe theo dải video ID từ Drive, clone nhánh `vbs-dev`, cài PP-OCRv6 trong venv riêng để tránh xung đột NCCL với Torch GPU của Kaggle, dùng model chọn trong `configs/ocr_extraction.yaml`, rồi upload JSON OCR thô theo lô 10 video.
- `extract_asr.ipynb`: tải video V3C theo dải ID từ Drive, chạy Faster-Whisper large-v3 INT8 với Python có sẵn của Kaggle và PyAV <19, rồi upload chỉ JSON ASR theo lô 10 video; model và video đã tải được dùng lại khi chạy tiếp trong cùng phiên.
- `extract_caption.ipynb`: tải video V3C cùng Drive nguồn của ASR, clone nhánh `vbs-dev`, tách general shot bằng OmniShotCut rồi caption bằng VideoLLaMA3-7B trên A100 (BF16). Shot lưu `scenes/V3C/<id>.scenes.txt` (frame đầu/cuối inclusive); caption lưu `captions/V3C/<id>.json` gồm `video_id`, `start`, `end`, `caption`. Upload chỉ caption JSON lên Drive đích theo lô tối đa 10 video; checkpoint `.partial.json` cho phép tiếp tục video dở dang. Không tạo venv.
- `host_server.ipynb`: chọn V3C hoặc LHE, tải ZIP embedding và map của dataset đó, build index/cache rồi chạy API và vLLM. Cần điền Drive folder ID embedding/map khi chọn LHE.

Chạy caption extraction trực tiếp trong repo: từ `backend/`, dùng `python -m scripts.caption.extract.run --dataset v3c` (hoặc `lhe`). Mặc định đọc `../data/videos/<DATASET>`, dùng/tạo shot tại `../data/scenes/<DATASET>`, và ghi JSON tại `../data/captions/<DATASET>`. CLI hỗ trợ `--video-id`, `--input-dir`, `--scenes-dir`, `--output-dir`, `--overwrite`. Chỉ gồm pipeline trích xuất; chưa lập chỉ mục hoặc thêm nhánh truy vấn caption.

## VLM keyframes trên Kaggle T4

Notebook có 5 cell code:
1. Config: mặc định chỉ LHE01 (`START_VIDEO_ID=END_VIDEO_ID=1`), có thể chọn range đến LHE75; sample_fps=3.0, batch embedding 16, decode batch 8, mmr_budget_fraction=.5, anchors_per_shot=3, `mmr_relevance_weight=.4`, `final_similarity_threshold=.97`, max scene 20 giây và source/output folder IDs. Cell in `pipeline_id` để dễ nhận ra notebook cũ.
2. Cài dependencies trực tiếp bằng pip vào Python hiện tại của Kaggle; dùng PyTorch/torchvision và scientific stack sẵn có, không tạo venv hoặc ép Python 3.11.
3. Rclone config và danh sách video. Nguồn `1yPb3EQBPKfKtlP0PGGZHmKDF3pghnRCN`; đích `11__IfRj1eLK4GZWEeNLs1SQK-DowEU1K`.
4. Source canonical nhúng sẵn và kiểm tra imports/CUDA. Khi cập nhật source thì chạy lại cell 4 trước cell 5.
5. Từng video: download → extract → package → upload bốn file → xóa job; lỗi thì giữ job và dừng. Chạy lại cell 5 để tiếp tục.

Rclone remote mặc định `iamtro`. Gắn rclone.conf hoặc tạo Kaggle Secret `RCLONE_CONFIG`. T4 GPU 0 được dùng; hai T4 không tự gộp VRAM hoặc tăng gấp đôi tốc độ.

NVDEC decode/RGB conversion vào GPU; PyTorch GPU resize/normalize, PE-Core-L embedding, clustering, histogram và JPEG encoding qua torchvision CUDA/nvJPEG. Bộ đệm decoder tối đa 2 × batch, window Omni giữ hữu hạn; không có cache toàn bộ RGB trên RAM/đĩa. Chỉ preprocess/embedding mẫu 3 FPS trong general shots. Decode NVDEC vẫn đọc stream liên tục; bộ đệm theo decode batch 8, không tăng theo FPS. Các mẫu được gom đủ batch embedding 16 trước khi inference. Embedding float32 giữ trong RAM: M × 1024 × 4 byte với M là số mẫu, không phải số frame nguồn. Đoạn 20 giây ở 60 FPS có khoảng 60 mẫu thay vì 1200 frame; ma trận khoảng cách clustering cũng giảm tương ứng. Scene quá ngắn lấy một frame giữa. Model detector được giải phóng trước khi load PE-Core. PE-Core được giải phóng trước clustering. CPU còn demux/điều phối, xử lý các tie số học gần nhau, ghi byte JPEG đã encode trên GPU với tối đa hai tác vụ, pickle, ZIP STORE và upload. Không cam kết GPU utilization 100%.

OmniShotCut dùng checkpoint v1.5 pin revision và cùng loop inference, overlap/padding/query decoding/stitching upstream. Chỉ thay frame iterable và validation transform để xử lý tensor GPU. Tensor resize/NVDEC color conversion có thể khác pixel nhỏ với FFmpeg/PIL; không hứa kết quả bitwise hoặc biên cut trùng tuyệt đối. Không hỗ trợ NVDEC thì fail rõ ràng, không fallback CPU. Cần NVIDIA driver/libnvcuvid và codec được T4 hỗ trợ; import smoke test chưa thay thế kiểm tra decode thật.

Omni shot ngắn dưới 1 giây được gộp với shot kế bên đến khi không còn shot dưới ngưỡng (trừ video chỉ có một shot rất ngắn). Quality gate và PE-Core chạy trong cùng một lượt NVDEC trên GPU. Gate đếm bin màu trên toàn khung hình (<=5 là gần đồng màu), còn cháy sáng, tương phản grayscale và variance of Laplacian dùng vùng giữa; không loại chỉ vì ảnh tối. Shot có toàn bộ mẫu 3 FPS đồng màu bị bỏ. Nếu shot còn mẫu không đồng màu nhưng tất cả trượt các tiêu chí chất lượng khác, giữ mẫu không đồng màu có điểm cao nhất. Ảnh được xuất bằng lượt NVDEC riêng và mã hóa JPEG trên CUDA; CPU chỉ ghi byte ảnh. Clustering theo chunk 20 giây tạo candidate; mỗi shot còn lại giữ tối đa ba anchor, rồi MMR toàn cục chọn thêm đến 50% tổng candidate (hoặc nhiều hơn nếu anchor bắt buộc vượt ngân sách), dùng công thức 0,4 đại diện - 0,6 giống ảnh đã chọn. Cuối cùng, mọi cặp PE-Core cosine >0,97 được xét theo độ khác biệt với keyframe lân cận (60%) và độ nét đo ở quality gate (40%); không decode hay embed lại. Shot có thể mất ảnh nếu ảnh trùng với shot khác. Chưa có GPU T4 tại máy local để xác nhận inference end-to-end.

Output mỗi video (LHE chỉ là tên dataset, không còn method/pipeline riêng):
```
keyframes/LHE/LHE01.zip
map_keyframes/LHE/LHE01.csv
scenes/LHE/LHE01.scenes.txt
embeddings/LHE/LHE01.pkl
```
Scene TXT ghi các khoảng general sau khi gộp shot ngắn, trước quality gate. Shot bị gate loại vẫn có thể còn trong TXT; CSV map hiện không có `scene_id`.
ZIP chỉ chứa ảnh JPG flat. Map cột `keyframe_id,video_id,frame_idx,timestamp_sec,fps`; scene inclusive start/end chỉ các range general; PKL là dict `format=vlm-sampled-v1`, `features` numpy float32 M×1024, `sharpness` float32 M, `frame_indexes`, `timestamps`, `sample_ranges`, `sample_fps`, `source_total_frames`, `source_fps`, `model_name`. Dòng i là mẫu thứ i; `frame_indexes[i]` và `timestamps[i]` mới là định danh/thời gian gốc. CSV map vẫn có cùng cột, frame_idx và fps gốc; timestamp_sec lấy từ PTS thực, không lấy chỉ số mẫu hoặc chia cho 3. Cấu trúc bốn file upload không đổi, nhưng consumer đọc PKL phải dùng schema sampled mới. Ảnh gốc quality95. Không xuất shot MP4. Upload map sau cùng, chỉ xóa workspace sau cả 4 copyto thành công; lỗi giữ nguyên workspace. Không xóa file Drive. Workspace mới `/kaggle/working/vlm_pipeline`, không dùng cache SurgVLP cũ. Cache full-frame cũ hoặc khác sampling plan tự được tính lại; scene vẫn dùng lại. Cache trước khi thêm điểm nét cần tính lại quality+embedding một lần. Map cũ chỉ được skip nếu cache sampled và selection_policy cùng khớp. Đổi ngân sách MMR, số anchor hoặc ngưỡng dedup sẽ dùng lại embedding và chọn/xuất ảnh lại; đổi tiêu chí lọc chất lượng có thể cần tính lại embedding. selection_policy được lưu trong cùng PKL, không sinh sidecar; chỉ áp dụng khi cache/video còn local. Video đã upload/cleanup thì không còn cache local để dùng lại. Timestamp lấy từ packet PTS qua ffprobe (chỉ demux, không CPU decode), sắp theo presentation order và chuẩn hóa về frame đầu. Timeline yêu cầu mỗi displayed frame có một packet PTS duy nhất, kiểm tra số lượng theo metadata và NVDEC. Video có PTS thiếu/trùng hoặc packet/frame không khớp sẽ báo lỗi thay vì ghi map sai; VFR được hỗ trợ khi thỏa điều kiện đó. Cache hiện chưa xác minh hash nội dung source; dùng cùng video khi resume.

Notebook VLM và HEVC chứa sẵn mã cần chạy trên Kaggle, gồm rclone và các helper. Notebook VLM cũng nhúng bản source pipeline dưới dạng ZIP trong cell 4; khi sửa pipeline ở repo, phải cập nhật payload trong cell đó trước khi chạy bản mới. Các bước smoke/worker chạy bằng `sys.executable` trong process mới để đọc packages sau cài đặt.





