# Notebooks

- `vlm_keyframes.ipynb`: tải từng video từ Drive, OmniShotCut → lấy mẫu 3 FPS → lọc chất lượng trên GPU → PE-Core-L14-336 → clustering → anchor theo shot + MMR → dedup 0,97 theo độ mới và độ nét → JPEG/ZIP → upload → cleanup.
- `compress_video.ipynb`: tải từng video từ Drive nguồn, NVDEC → CUDA resize → HEVC NVENC 480p, upload vào `videos/` của Drive đích. Cell cuối in tiến độ rclone và FFmpeg.
- `extract_embedding.ipynb`: chọn `DATASET=V3C` hoặc `DATASET=LHE`, trích xuất Qwen3-VL-Embedding-2B qua vLLM từ ZIP keyframe và upload ZIP vector.
- `extract_ocr_paddle.ipynb`: tải ZIP keyframe theo dải video ID từ Drive, clone nhánh `vbs-dev`, cài PP-OCRv6 trong venv riêng để tránh xung đột NCCL với Torch GPU của Kaggle, dùng model chọn trong `configs/ocr_extraction.yaml`, rồi upload JSON OCR thô theo lô 10 video.
- `extract_asr.ipynb`: tải video V3C theo dải ID từ Drive, chạy Faster-Whisper large-v3 INT8 với Python có sẵn của Kaggle và PyAV <19, rồi upload chỉ JSON ASR theo lô 10 video; model và video đã tải được dùng lại khi chạy tiếp trong cùng phiên.
- `extract_caption.ipynb`: tải video V3C cùng Drive nguồn của ASR, clone nhánh `vbs-dev`, tách general shot bằng OmniShotCut rồi caption theo batch bằng InternVL3.5-2B qua vLLM. Prompt nằm trong một cell riêng để thử nghiệm mà không cần sửa repo trên GitHub. Shot lưu `scenes/V3C/<id>.scenes.txt` (frame đầu/cuối inclusive); caption lưu theo model và prompt trong `captions_2b/<run_id>/V3C/<id>.json` gồm `video_id`, `start`, `end`, `caption`. Upload chỉ caption JSON lên Drive đích theo lô tối đa 10 video; checkpoint `.partial.json` cho phép tiếp tục video dở dang. Không tạo venv.
- `host_server.ipynb`: notebook host cũ cho luồng visual embedding/map. Flow ASR/caption hierarchical hiện tại dùng `hierarchical_search_kaggle.ipynb`.
- `embed_asr_caption_kaggle.ipynb`: chọn `DATASET`, `START_VIDEO_ID`, `END_VIDEO_ID` theo cùng quy ước với `extract_embedding.ipynb` (V3C 5 chữ số, LHE `LHE` + 2 chữ số; range gồm hai đầu); chỉ tải, embed và upload đúng các video trong range. Cần ASR/caption JSON cùng map keyframe; mỗi video/source lưu ma trận dưới `asr/<DATASET>/<video_id>.npy` và/hoặc `caption/<DATASET>/<video_id>.npy`.
- `hierarchical_search_kaggle.ipynb`: tải visual/ASR/caption/OCR/map cùng vector ASR/caption, dựng các index và khởi động server Qwen vLLM + Elasticsearch + API.

Chạy caption extraction trực tiếp trong repo: từ `backend/`, dùng `python -m scripts.caption.extract.run --dataset v3c` (hoặc `lhe`). Mặc định đọc `../data/videos/<DATASET>`, dùng/tạo shot tại `../data/scenes/<DATASET>`, và ghi JSON tại `../data/captions/<DATASET>`. CLI hỗ trợ `--video-id`, `--input-dir`, `--scenes-dir`, `--output-dir`, `--overwrite`. Chỉ gồm pipeline trích xuất; chưa lập chỉ mục hoặc thêm nhánh truy vấn caption.

Caption speed labels: sau shot detection, SpeedPredictor gắn nhãn toàn bộ caption segment trong RAM trước khi nạp InternVL3.5/vLLM. Giữ nguyên 16 frame theo checkpoint, nhịp tham chiếu 7.5 FPS, 3 lượt dự đoán thích nghi (window count, neo cửa sổ cuối, lấy đều frame, tổng hợp log-score, giới hạn factor theo thời lượng và FPS thực) theo code tác giả; không nội suy vị trí, không dùng 40 frame/5 giây. NVDEC và chuẩn hóa/resize/inference trên CUDA, không fallback CPU. Chỉ giữ một segment đã resize trong GPU khi dự đoán. Shot dưới 2.133s hoặc không tạo được cửa sổ hợp lệ báo unknown, không thêm từ chỉ tốc độ. Ngưỡng >1.8 timelapse, <0.6 slow-motion. Console mặc định hiển thị progress bar OMNI/SPEED/CAPTION; bật VERBOSE_LOGS để xem log chi tiết hoặc PRINT_CAPTIONS để xem caption cuối; không tạo file log hay file score/label. Notebook dùng subprocess -u, PYTHONUNBUFFERED và flush từng dòng. JSON cuối giữ bốn trường cũ, partial JSON phục vụ resume. Run ID gồm cấu hình nội dung caption để không dùng nhầm kết quả 40-frame cũ. Cần kiểm định lại ngưỡng trên luồng suy luận gốc; không phải xác nhận giám định video đã chỉnh tốc độ.

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






NVDEC caption predictor: cấu hình `speed_predictor.decoder: simple` dùng SimpleDecoder (NVDEC đồng bộ, GPU RGBP, primary CUDA context). Đây là đường thay cho ThreadedDecoder gặp SIGSEGV trong constructor ở môi trường Kaggle đã báo (PyNvVideoCodec 2.2.3 / Torch 2.11 cu130 / Python 3.13). Không khẳng định nguyên nhân sâu hoặc đã sửa lỗi trên GPU khi chưa chạy lại. Các pipeline khác vẫn giữ decoder mặc định của chúng. Cập nhật notebook và chạy lại cell setup/config để helper run() mới có heartbeat/exit code.

SpeedPredictor/NVDEC chạy trong worker process mới không import Decord/PyAV/OpenCV/torchvision/OmniShotCut. Pipeline chính chỉ import Decord sau khi worker đã thoát. Kết quả nhãn truyền qua RAM/stdin/stdout, không lưu file trung gian. Worker crash được chuyển thành lỗi rõ ràng ở parent, log worker vẫn stream trực tiếp. Cách ly xử lý khả năng xung đột native; chưa có kiểm chứng GPU cho môi trường Kaggle gây lỗi.

### Caption InternVL3.5 + vLLM

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
