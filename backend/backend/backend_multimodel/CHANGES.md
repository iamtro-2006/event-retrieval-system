# Multi-model embedding support - what changed

## Tư duy thiết kế
- **Không viết lại** `embedder.py` / `pipeline` / `run.py` - chỉ thêm 1 lớp
  "backend" phía sau `load_clip_model()` để nó có thể nạp bất kỳ model nào,
  không chỉ open_clip.
- **1 file backend/model**, không phải 1 file/mọi thứ: `open_clip_backend.py`,
  `hf_clip_backend.py` (Long-CLIP-L), `blip2_backend.py`, `beit3_backend.py`.
- `registry.py` là factory duy nhất: pipeline chỉ gọi
  `load_model(name, backend, pretrained, ...)`, không cần biết chi tiết từng backend.
- Mọi backend đều trả về object có `.encode_image(batch) -> Tensor`, nên
  `encode_keyframe_images()` trong `embedder.py` **không đổi 1 dòng nào**.

## Model đã hỗ trợ

| Model | Backend | Cách nạp |
|---|---|---|
| ViT-SO400M-16-SigLIP2-384 | `open_clip` | đã có sẵn trong open_clip (`pretrained="webli"`) |
| ViT-H-14-378-quickgelu (DFN) | `open_clip` | đã có sẵn trong open_clip (`pretrained="dfn5b"`) |
| Long-CLIP-L | `hf_clip` | `transformers.AutoModel` (`creative-graphic-design/LongCLIP-L`, trust_remote_code) |
| BLIP-2 (ITC/retrieval) | `blip2` | `transformers.Blip2VisionModelWithProjection` (`Salesforce/blip2-itm-vit-g`) |
| BEiT-3 | `beit3` | code + checkpoint từ `microsoft/unilm/beit3` (không có trên `transformers`/`open_clip`) |

2 model đầu (SigLIP2, DFN) **không cần code mới** - chỉ cần đúng tên/pretrained
trong config, vì open_clip đã hỗ trợ native.

## Cần cài thêm (ngoài open_clip/torch/opencv đã có)
```bash
pip install --break-system-packages transformers timm torchvision
# BEiT-3 (tuỳ chọn - chỉ cần nếu bật beit3_large_itc trong config):
git clone https://github.com/microsoft/unilm.git third_party/unilm
pip install --break-system-packages -r third_party/unilm/beit3/requirements.txt
# tải 1 checkpoint *-itc, ví dụ beit3_large_patch16_384_coco_retrieval.pth
```

## Chạy
```bash
# xem danh sách model (key) đang có trong config
python scripts/embedding_extraction/run.py --list-models

# chạy TẤT CẢ model có enabled: true trong config
python scripts/embedding_extraction/run.py --config configs/embeddings.yaml

# chỉ chạy đúng 1 model theo tên (key trong config) - dù enabled: true/false
python scripts/embedding_extraction/run.py --model siglip2_so400m_384

# chạy nhiều model cụ thể - lặp lại flag hoặc phân tách bằng dấu phẩy
python scripts/embedding_extraction/run.py --model siglip2_so400m_384 --model blip2_itm_vitg
python scripts/embedding_extraction/run.py --model siglip2_so400m_384,blip2_itm_vitg
```

`--model` luôn thắng `enabled` trong config - tức là gõ tên nào thì model đó
chạy, kể cả đang `enabled: false` (ví dụ `beit3_large_itc` sau khi bạn đã
setup repo/checkpoint xong nhưng chưa muốn nó tự chạy mỗi lần `run.py` không
tham số).

Output: `output_embeddings_root/<model_key>/<dataset>/<video>/<frame>.npy`
(mỗi model một thư mục riêng, không đè lên nhau).

## Lưu ý
- `src/utils/device.py` (`resolve_device`) và `src/utils/config.py`
  (`FrameLoaderConfig`) được giả định **đã có sẵn** trong repo gốc của bạn
  (được import bởi code cũ) - không đụng tới trong bản này.
- BLIP-2 Q-Former trả về 32 vector query-token/ảnh, không phải 1 vector.
  Backend đang **mean-pool** thành 1 vector (`pooling: mean` trong config,
  đổi sang `first` nếu muốn nhẹ hơn). Nếu sau này cần giữ full multi-vector
  cho retrieval chính xác hơn, cần sửa bước lưu `.npy` trong pipeline.
- BEiT-3 mặc định `enabled: false` trong config mẫu vì cần clone thêm repo
  ngoài + tải checkpoint riêng - bật lên khi đã setup xong `third_party/unilm/beit3`.
