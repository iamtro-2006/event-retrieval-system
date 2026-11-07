# Notebooks

- `vlm_keyframes.ipynb`: từng video Drive → OmniShotCut clean_shot → timestamp 3 FPS / vùng tối đa 30 giây → GPU quality + PE-Core-L14-336 → P3 common/medoid/unique + MMR budget=None → dedup cosine >0,96 chỉ giữa shot kề nhau → JPG → scene cuối → ZIP/upload/cleanup.
- `compress_video.ipynb`: tải từng video từ Drive nguồn, NVDEC → CUDA resize → HEVC NVENC 480p, upload vào `videos/` của Drive đích. Cell cuối in tiến độ rclone và FFmpeg.
- `extract_embedding.ipynb`: chọn `DATASET=V3C` hoặc `DATASET=LHE`, trích xuất Qwen3-VL-Embedding-2B qua vLLM từ ZIP keyframe và upload ZIP vector.
- `extract_ocr_paddle.ipynb`: tải ZIP keyframe theo dải video ID từ Drive, clone nhánh `vbs-dev`, cài PP-OCRv6 trong venv riêng để tránh xung đột NCCL với Torch GPU của Kaggle, dùng model chọn trong `configs/ocr_extraction.yaml`, rồi upload JSON OCR thô theo lô 10 video.
- `extract_asr.ipynb`: tải video V3C theo dải ID từ Drive, chạy Faster-Whisper large-v3 INT8 với Python có sẵn của Kaggle và PyAV <19, rồi upload chỉ JSON ASR theo lô 10 video; model và video đã tải được dùng lại khi chạy tiếp trong cùng phiên.
- `extract_caption.ipynb`: tải video V3C cùng Drive nguồn của ASR, clone nhánh `vbs-dev`, tách general shot bằng OmniShotCut rồi caption theo batch bằng InternVL3.5-2B qua vLLM. Prompt nằm trong một cell riêng để thử nghiệm mà không cần sửa repo trên GitHub. Shot lưu `scenes/V3C/<id>.scenes.txt` (frame đầu/cuối inclusive); caption lưu theo model và prompt trong `captions_2b/<run_id>/V3C/<id>.json` gồm `video_id`, `start`, `end`, `caption`. Upload chỉ caption JSON lên Drive đích theo lô tối đa 10 video; checkpoint `.partial.json` cho phép tiếp tục video dở dang. Không tạo venv.
- `embed_asr_caption_kaggle.ipynb`: chọn `DATASET`, `START_VIDEO_ID`, `END_VIDEO_ID` theo cùng quy ước với `extract_embedding.ipynb` (V3C 5 chữ số, LHE `LHE` + 2 chữ số; range gồm hai đầu); chỉ tải, embed và upload đúng các video trong range. Cần ASR/caption JSON cùng map keyframe; mỗi segment lưu vector riêng theo ordinal JSON tại `asr/<DATASET>/<video_id>/<segment_number>.npy` và/hoặc `caption/<DATASET>/<video_id>/<segment_number>.npy`. Caption shot luôn được map keyframe nếu video có keyframe; segment thiếu vector khi build host chỉ bị bỏ riêng.
- `hierarchical_search_kaggle.ipynb`: notebook Kaggle chính. Tải visual/ASR/caption/OCR/map cùng vector ASR/caption theo dataset và range; dựng index rồi khởi động Qwen vLLM, Elasticsearch và API. Cell cài Codex CLI nếu thiếu; cell kế tiếp đăng nhập bằng device code và xác nhận trạng thái. Nếu tổ chức không bật device authentication, cần đăng nhập trên host agent bằng `codex login`.

Chạy caption extraction trực tiếp trong repo: từ `backend/`, dùng `python -m scripts.caption.extract.run --dataset v3c` (hoặc `lhe`). Mặc định đọc `../data/videos/<DATASET>`, dùng/tạo shot tại `../data/scenes/<DATASET>`, và ghi JSON tại `../data/captions/<DATASET>`. CLI hỗ trợ `--video-id`, `--input-dir`, `--scenes-dir`, `--output-dir`, `--overwrite`. Chỉ gồm pipeline trích xuất; chưa lập chỉ mục hoặc thêm nhánh truy vấn caption.

Caption speed labels: sau shot detection, SpeedPredictor gắn nhãn toàn bộ caption segment trong RAM trước khi nạp InternVL3.5/vLLM. Giữ nguyên 16 frame theo checkpoint, nhịp tham chiếu 7.5 FPS, 3 lượt dự đoán thích nghi (window count, neo cửa sổ cuối, lấy đều frame, tổng hợp log-score, giới hạn factor theo thời lượng và FPS thực) theo code tác giả; không nội suy vị trí, không dùng 40 frame/5 giây. NVDEC và chuẩn hóa/resize/inference trên CUDA, không fallback CPU. Chỉ giữ một segment đã resize trong GPU khi dự đoán. Shot dưới 2.133s hoặc không tạo được cửa sổ hợp lệ báo unknown, không thêm từ chỉ tốc độ. Ngưỡng >1.8 timelapse, <0.6 slow-motion. Console mặc định hiển thị progress bar OMNI/SPEED/CAPTION; bật VERBOSE_LOGS để xem log chi tiết hoặc PRINT_CAPTIONS để xem caption cuối; không tạo file log hay file score/label. Notebook dùng subprocess -u, PYTHONUNBUFFERED và flush từng dòng. JSON cuối giữ bốn trường cũ, partial JSON phục vụ resume. Run ID gồm cấu hình nội dung caption để không dùng nhầm kết quả 40-frame cũ. Cần kiểm định lại ngưỡng trên luồng suy luận gốc; không phải xác nhận giám định video đã chỉnh tốc độ.

## VLM keyframes trên Kaggle T4

`vlm_keyframes.ipynb` là notebook duy nhất cho bản VLM mới hợp nhất V3C.
Luồng canonical: OmniShotCut clean_shot → lấy mẫu theo PTS 3 FPS → GPU quality
+ PE-Core-L14-336 → P3 common/medoid/unique, MMR budget=None → dedup cosine
>0.96 chỉ giữa shot kề nhau → CUDA JPEG → scene/map → ZIP/upload/cleanup.
Giữ clean shot gốc, vùng sampling tối đa 30 giây; không gộp shot ngắn và không
dedup trong cùng shot. Shot toàn mẫu đồng màu bị bỏ, shot còn tín hiệu có
fallback nonflat. Không còn notebook thử nghiệm V3C hoặc đường clustering cũ.

### Chạy notebook

1. Config: chọn `DATASET="V3C"` hoặc `LHE`; mặc định V3C2 07476–07485.
   `DATASET_GROUPS`, `VIDEO_ID_RANGES`, `VIDEO_ID_LIST` chọn nguồn V3C (list ghi đè range).
   LHE dùng START_VIDEO_ID/END_VIDEO_ID trong 1..75. Pipeline ID `vlm-p3-quality-v2`.
2. Cài trực tiếp vào Python Kaggle, dùng scientific/PyTorch stack sẵn có.
3. Rclone: V3C dùng SFTP config `/kaggle/input/datasets/khanho/sftp-rclone/rclone.conf`,
   remote `stfp`, layout `V3C1|V3C2|V3C3/videos/<ID>/<ID>.<extension>`.
   ID giới hạn trong collection, ưu tiên MP4 → MOV → MKV → WEBM → AVI;
   liệt kê đệ quy theo chunk 200, dự phòng từng folder. Hỗ trợ các đuôi trong
   `remote_jobs.VIDEO_EXTENSIONS`; nhận đuôi không đảm bảo codec được NVDEC hỗ trợ.
   LHE dùng Drive nguồn `1yPb3EQBPKfKtlP0PGGZHmKDF3pghnRCN`, tên `LHE##.mp4`.
   Drive đích `11__IfRj1eLK4GZWEeNLs1SQK-DowEU1K`, remote `iamtro`;
   gắn rclone.conf hoặc Kaggle Secret `RCLONE_CONFIG`. SFTP/Drive dùng config riêng.
4. Giải nén source canonical nhúng sẵn, kiểm tra imports/CUDA. Không cần clone repo.
5. Chạy batch: `VIDEO_BATCH_SIZE=1`, `UPLOAD_WORKERS=4` mặc định.
   Tải đủ batch → extract tuần tự trong một persistent worker → package cả batch →
   upload song song trong từng folder, chờ xong lần lượt keyframes → scenes →
   embeddings → map_keyframes → cleanup. Không thêm tầng V3C/LHE trên Drive.
   `UPLOAD_DONE` chỉ cập nhật sau upload đủ bốn folder và cleanup thành công.
   Lỗi giữ workspace `jobs/<V3C|LHE>/<id>` để chạy lại; chưa tải batch tiếp.

### Model, timeline và input

OmniShotCut v1.5 pin revision, cùng overlap/padding/stitching upstream;
clean_shot lọc intra label general. Omni và PE-Core load lazy một lần, cùng giữ
trên GPU 0 qua các video và batch, inference tuần tự. Worker đóng khi xong/lỗi/interrupt.
NVDEC decode/RGB, resize/normalize, quality, embedding và nvJPEG chạy GPU;
P3/k-medoids/AUCC, demux/điều phối, ZIP STORE và ghi byte ảnh dùng CPU.
Decode batch 8, embedding batch 16, tối đa hai tác vụ ghi ảnh; không cache toàn bộ RGB.
Hai T4 không tự gộp VRAM. Chưa benchmark tốc độ/VRAM hay chạy end-to-end GPU local.

Quality: RGB histogram 8×8×8 (>5 bin); ảnh rộng tối đa 320, Laplacian >=20,
brightness 10..245, loại khi entropy <1.2 đồng thời edge density <.002.
Canny PyTorch GPU 80/160; điểm .35 sharpness + .25 exposure + .20 entropy + .20 edge.
Resize/color conversion có thể lệch OpenCV/PIL, không bảo đảm kết quả bitwise.
Dedup ưu tiên quality rồi frame sớm, chỉ so với ảnh đã giữ; không gộp bắc cầu.

Timeline dùng ffprobe packet PTS/flags, không decode CPU để kiểm tra.
Loại packet discard; chỉ bỏ PTS trùng cuối của AVC MP4/MOV khi byte NAL xác nhận
end-of-sequence/stream. Duplicate khác và mismatch frame vẫn báo lỗi.
SimpleDecoder giới hạn batch cuối theo display total; ThreadedDecoder kiểm tra EOF.
Raw shot cache ở `.cache/omni_clean_shot_timeline_v2/`; scene inclusive chỉ được
publish sau export và chỉ giữ shot có keyframe cuối, map publish sau cùng.

Codec/profile không hỗ trợ T4 NVDEC (DNxHD/ProRes, H.264 10-bit/4:4:4,
HEVC 4:2:2...) chuyển tạm trong `jobs/.../transcoded/`: FFmpeg CPU decode →
h264_nvenc, p1/CQ18/yuv420p, không ép FPS/resize. Kiểm tra frame count/PTS tương đối
(tolerance 2 microseconds), rồi smoke test NVDEC/JPEG trước model. OOM, mismatch
và error 300 chung chung không kích hoạt chuyển mã. Bản gốc giữ nguyên;
manifest cho retry, cleanup sau upload thành công. Cần FFmpeg có h264_nvenc.
Chuyển màu/nén có thể ảnh hưởng pixel và lựa chọn keyframe.

### Mã nguồn và kiểm tra

- `src/keyframe_extraction/`: thuật toán P3, quality, export, model cache,
  worker protocol, input codec và remote jobs.
- `scripts/keyframe_extraction/vlm_worker.py`: worker GPU canonical.
- `scripts/keyframe_extraction/vlm_run_cell.py`: cell xử lý notebook.
- `scripts/keyframe_extraction/build_vlm_keyframes.py`: đóng gói trực tiếp từ
  source repo theo manifest; không lấy source từ ZIP cũ, không giữ code merge/migration.

Sau sửa source, từ backend chạy:

```bash
python scripts/keyframe_extraction/build_vlm_keyframes.py
python -m unittest tests.test_vlm_keyframes tests.test_vlm_remote_jobs tests.test_video_io -q
```

Chạy lại cell code 4 rồi 5 để dùng bundle mới. Ba suite kiểm tra P3/quality,
source notebook, resident worker, thứ tự upload/cleanup, packet timeline và codec
fallback. Kiểm tra CPU/mock không thay thế chạy pipeline thật trên Kaggle T4.

## Caption InternVL3.5 + vLLM

Notebook `extract_caption.ipynb` dùng duy nhất `OpenGVLab/InternVL3_5-2B` qua vLLM; đã bỏ backend Transformers tuần tự và notebook thử nghiệm model cũ. Chạy lại cell 2 → 3 → 4 → 5 khi nâng cấp. Sửa prompt/batch/FPS ở cell 3 rồi chạy lại cell 5 trong cùng phiên.

Cell 2 cài `backend/requirements-caption.txt`, pin vLLM 0.11.2 và Transformers 4.57.1; vLLM yêu cầu bộ Torch 2.9.0/torchvision 0.24.0 tương ứng. Vì vậy bộ Torch có sẵn trên Kaggle có thể bị thay thế. Không cài chồng `backend/requirements.txt` vào runtime caption. Notebook kiểm tra imports/CUDA trong process mới và không import Torch/vLLM vào kernel. Dùng runtime Linux có NVIDIA GPU/driver tương thích; chưa chạy end-to-end trên GPU máy phát triển.

Mặc định lấy mẫu đều **1.5 FPS, tối đa 16 frame/shot**, theo FPS thực. Mỗi frame là một tile theo config model (448×448), đánh số và có timestamp. Dùng đầu vào multi-image theo thứ tự frame, tương ứng cách video chat của model; vLLM xử lý image tokens/normalization. Một tile/frame, không thinking, không quantization. BF16 trên GPU hỗ trợ, FP16 trên T4. Giữ nguyên prompt người dùng và giới hạn 220 output tokens; nếu caption bị giới hạn độ dài sẽ có WARNING để tăng `MAX_NEW_TOKENS`.

Tự nhận GPU visible ở runtime: **1 GPU → TP=1, tối đa 3 shot đồng thời, 6 shot/batch; 2 GPU → TP=2, tối đa 6 shot đồng thời, 12 shot/batch**. Nếu thấy hơn hai GPU thì chỉ dùng hai GPU đầu. TP=2 chia model qua cả hai GPU, không phải hai engine độc lập. GPU memory utilization áp dụng cho từng GPU; số shot đồng thời là giới hạn toàn engine, không nhân thêm lần nữa. SpeedPredictor/OmniShotCut vẫn dùng GPU 0 theo thứ tự trước khi nạp vLLM. Notebook in tên/VRAM thực tế và kế hoạch trước inference, CLI tự xác nhận lại GPU count. Chưa benchmark một/hai GPU Kaggle, không khẳng định TP nhanh gấp đôi vì còn giao tiếp giữa GPU. Một thread lấy frame/resize cho batch kế tiếp trong lúc GPU sinh caption, chỉ giữ tối đa một batch chờ. Reader không bị dùng đồng thời bởi nhiều thread; tokenizer chuẩn bị prompt tách riêng tokenizer của engine. Chỉ giữ ảnh đã resize trong hàng đợi để giảm RAM/IPC. Engine được dùng lại cho toàn bộ lô video; không gc/empty_cache sau mỗi shot. Bật chunked prefill, prefix caching và CUDA graph/compilation. Cache tiền xử lý ảnh tắt vì hầu hết frame chỉ được dùng một lần. Không tự giảm frame hoặc retry engine sau CUDA OOM.

Các biến tại cell 3:

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `CAPTION_BATCH_SIZE` | auto | 6/12 shot mỗi lần cho 1/2 GPU; có thể đặt số nguyên |
| `TENSOR_PARALLEL_SIZE` | auto | 1 GPU dùng TP=1, ≥2 GPU dùng TP=2 |
| `MAX_NUM_SEQS` | auto | 3/6 shot đồng thời cho TP=1/2 |
| `GPU_MEMORY_UTILIZATION` | 0.88 | Ngân sách GPU của vLLM |
| `MAX_MODEL_LEN` | 6144 | Context gồm image/text/output tokens |
| `MAX_NUM_BATCHED_TOKENS` | auto | 4096/8192 cho TP=1/2; giảm khi thiếu VRAM |
| `PRINT_CAPTIONS` | False | True để in từng caption cuối |
| `VERBOSE_LOGS` | False | True để bật log INFO chi tiết thay cho bar |
| `ENFORCE_EAGER` | False | True tắt graph/compile khi thiếu VRAM |
| `MAX_FRAMES` | 16 | Giảm 12/8 nếu muốn thử đánh đổi chất lượng/VRAM |
| `MIN_COLOR_BINS` | 5 | Loại frame khi số bin màu có pixel **<5**; đúng 5 bin vẫn giữ |

OOM trong engine: đọc traceback, giảm concurrency/token budget hoặc bật eager và chạy lại CLI (cell 3 + 5). OOM khi khởi động: có thể cần giảm GPU memory utilization để chừa chỗ graph/vision encoder; tăng tỷ lệ chỉ khi log báo thiếu KV-cache và GPU còn đủ chỗ. Không có cam kết nhanh gấp bao nhiêu khi chưa benchmark. Lần đầu compile/load lâu hơn các batch sau.

Mặc định chỉ hiện progress bar cho load model, OMNI decode/windows, SPEED và CAPTION; warning/error/traceback luôn giữ lại. Decode upstream không cung cấp số frame theo thời gian nên bar OMNI decode là 0/1 video, không bịa phần trăm frame. Model loading gồm download/compile/capture, dùng bar 0/1 model để theo dõi thời gian chờ. Event tiến độ được worker stream qua stdout, notebook dựng một bar cho từng stage/video, throttle .5s; không in bar mới cho mỗi event. Chạy CLI trực tiếp dùng tqdm terminal. Chỉ chế độ verbose mới bật diagnostics NVDEC và log từng frame/pass/score. `PRINT_CAPTIONS=True` in caption cuối từng shot. Không ghi file log.

CAPTION bar tính số cửa sổ đã xử lý (gồm shot skipped), có postfix tổng inferred/skipped và tốc độ batch. Batch đã checkpoint mới tăng bar; lỗi không tự đẩy bar lên 100%. Checkpoint sau mỗi batch thành công; lỗi giữa batch thì chỉ batch đó phải chạy lại. Số frame/thứ tự/nhãn/schema không đổi khi tắt log. Đặt `CAPTION_VERBOSE=1` cho CLI để debug; `CAPTION_PROGRESS_PROTOCOL=1` dành cho notebook pipe.

SpeedPredictor, prefix timelapse/slow-motion và JSON bốn trường giữ nguyên. Thời gian nội bộ giữ độ chính xác của biên frame, không làm tròn 3 chữ số trước khi sample để tránh lấy nhầm frame shot kế. Run ID gồm model/prompt/frame/nhãn tốc độ và backend để tách kết quả local. Chỉnh batch/VRAM không đổi run ID hoặc checkpoint signature, nên có thể giảm concurrency và resume sau OOM; đổi frame/prompt sẽ tạo run mới. **Dùng folder Drive đích riêng để so sánh cùng video**; mặc định JSON cùng ID đã tồn tại vẫn bị bỏ qua. Không tự ghi đè kết quả trên Drive.

Frame caption được lọc trước resize bằng histogram **RGB 8×8×8, [0,256)** trên toàn frame, cùng cách chia bin của quality gate keyframe; gồm cả giá trị kênh 255. Chỉ áp dụng tiêu chí màu này, không thêm lọc độ nét/sáng/tương phản. Ngưỡng caption theo yêu cầu là **<5 thì loại, ≥5 giữ**, khác điều kiện `>5` ở gate keyframe. Không resize trước khi đếm vì nội suy có thể tạo thêm màu. Giữ nguyên tập mẫu 1.5 FPS, không lấy bù hoặc lặp frame; timestamp các frame còn lại vẫn là thời gian gốc.

Khi VERBOSE_LOGS=True, `CAPTION-FILTER` in frame index/timestamp/bin count cho frame bị loại và tổng sampled/kept/dropped; mặc định các dòng này ẩn. Shot bị loại hết frame mẫu không gọi generate và không xuất segment; toàn video bị loại thì JSON là `[]` hợp lệ. Checkpoint nội bộ vẫn đánh dấu cửa sổ skipped để resume đúng vị trí và không lệch nhãn speed ở shot sau; JSON cuối chỉ có bốn trường cũ. Run ID và checkpoint signature chứa version/ngưỡng filter để tách khỏi kết quả trước khi lọc.
