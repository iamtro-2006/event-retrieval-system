# Tích hợp Meta Perception Encoder Core

Backend `perception_encoder` dùng implementation chính thức của Meta và dùng
chung contract `encode_image` / `encode_text` với OpenCLIP và BLIP-2.
Vì vậy một PE-Core model có thể chạy cả extraction, semantic/image search,
temporal search và multi-model RRF fusion mà không cần một search pipeline mới.

## Model hỗ trợ

Các preset có sẵn: `PE-Core-T16-384`, `PE-Core-S16-384`,
`PE-Core-B16-224`, `PE-Core-L14-336`, `PE-Core-G14-448`, cùng alias Hugging
Face dạng `facebook/PE-Core-*`. Có thể dùng một config PE-Core mới hơn bằng
cách đặt `backend: perception_encoder` và `name` đúng với
`pe.CLIP.available_configs()` của package đang cài.

PE-Lang và PE-Spatial không được adapter này nhận: chúng trả dense/token
features, không có cặp text tower CLIP phù hợp với flat-vector FAISS retrieval
hiện tại.

## Cài dependency

Package chính thức chưa được phát hành như một wheel nhẹ trên PyPI. File
requirements tùy chọn đã pin commit Meta được kiểm thử và `einops` tương thích:

```powershell
python -m pip install --no-deps -r requirements-pe-core.txt
```

Lệnh trên giả định môi trường chính đã được cài bằng `../requirements.txt`.
PE-Core tại revision được pin khai báo **FAIR Noncommercial Research License**;
người triển khai phải đọc license upstream trước khi phân phối hoặc sử dụng
thương mại. Repository này không commit source PE-Core, checkpoint hay index.

Hoặc clone repository rồi khai báo
`perception_encoder.repo_path: D:/path/to/perception_models` trong YAML.

## Chạy end-to-end

Mọi đường dẫn mẫu dùng `../data/...`, được resolve từ thư mục `backend/` và
không phụ thuộc ổ đĩa của tác giả. Entry PE-Core trong
`configs/embeddings.yaml` mặc định tắt. Có thể chạy
riêng entry đó mà không cần đổi `enabled`:

```powershell
cd backend
python scripts/embedding_extraction/run.py --model pe_core_l14_336
python scripts/retrieval/run.py --task build-index --config configs/indexing.pe-core.example.yaml
```

Sau khi index đã tồn tại, bật entry `model_key: pe-core` trong
`configs/app.yaml`. API config sẽ tự trả `pe-core` trong danh sách model khả
dụng và frontend fusion sẽ tự hiện checkbox tương ứng.

Advanced search có thể fuse PE-Core với model khác và OCR/ASR:

```json
{
  "query": "a person riding a bicycle in the rain",
  "semantic_models": ["pe-core", "vitH-378-quickgelu"],
  "weights": {"pe-core": 0.5, "vitH-378-quickgelu": 0.3, "ocr": 0.2},
  "use_ocr": true,
  "use_asr": false,
  "temporal": false
}
```

Mỗi model phải có FAISS index riêng vì dimension/embedding space khác nhau.
Không trộn vector PE-Core và OpenCLIP vào cùng một index; fusion diễn ra ở
tầng rank bằng RRF.
