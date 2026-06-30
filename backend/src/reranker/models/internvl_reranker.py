"""
internvl_reranker_v2.py
────────────────────────────────────────────────────────────────────────────
VLM Structured Reranker — InternVL3-2B (int-8 BitsAndBytes)
Schema v2: multi-criteria scoring, batch processing, full structured output.

Pipeline:
  CLIP search → trả results về frontend ngay
  Backend chạy ngầm InternVL3 rerank → trả structured results đã sắp xếp

Ref:  https://huggingface.co/OpenGVLab/InternVL3-2B
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
	datefmt="%H:%M:%S",
)
log = logging.getLogger("InternVLReranker")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Relevance score thresholds → decision label
_DECISION_THRESHOLDS = [
	(0.75, "strong_match"),
	(0.45, "relevant"),
	(0.20, "weak_match"),
	(0.00, "irrelevant"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Output schema dataclasses (easy to serialise → dict / JSON)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReasoningScores:
	object_match:    float = 0.0
	action_match:    float = 0.0
	scene_match:     float = 0.0
	attribute_match: float = 0.0
	relation_match:  float = 0.0
	lighting_match:  float = 0.0   # v2 extension
	count_match:     float = 0.0   # v2 extension
	text_match:      float = 0.0   # v2 extension (OCR / text in image)
	emotion_match:   float = 0.0   # v2 extension
	style_match:     float = 0.0   # v2 extension

	def to_dict(self) -> dict[str, float]:
		return {k: round(v, 4) for k, v in self.__dict__.items()}

	@property
	def weighted_mean(self) -> float:
		"""
		Trọng số theo độ quan trọng:
		  object   → 0.25
		  action   → 0.20
		  scene    → 0.15
		  attribute→ 0.10
		  relation → 0.10
		  lighting → 0.05
		  count    → 0.05
		  text     → 0.04
		  emotion  → 0.03
		  style    → 0.03
		"""
		weights = [0.25, 0.20, 0.15, 0.10, 0.10, 0.05, 0.05, 0.04, 0.03, 0.03]
		vals    = [
			self.object_match, self.action_match, self.scene_match,
			self.attribute_match, self.relation_match, self.lighting_match,
			self.count_match, self.text_match, self.emotion_match, self.style_match,
		]
		return float(np.dot(weights, vals))


@dataclass
class RerankResult:
	sample_id: str

	# --- rerank block ---
	relevance_score: float = 0.0
	confidence:      float = 0.0
	decision:        str   = "irrelevant"
	reasoning:       ReasoningScores = field(default_factory=ReasoningScores)
	explanation:     str   = ""

	# --- scores block ---
	clip_score:  float = 0.0
	vlm_score:   float = 0.0
	final_score: float = 0.0

	# --- perf block ---
	batch_id:         int   = 0
	sample_latency_ms: float = 0.0

	def to_dict(self) -> dict:
		return {
			"sample_id": self.sample_id,
			"rerank": {
				"relevance_score": round(self.relevance_score, 4),
				"confidence":      round(self.confidence, 4),
				"decision":        self.decision,
				"reasoning":       self.reasoning.to_dict(),
				"explanation":     self.explanation,
			},
			"scores": {
				"clip_score":  round(self.clip_score, 4),
				"vlm_score":   round(self.vlm_score, 4),
				"final_score": round(self.final_score, 4),
			},
			"perf": {
				"batch_id":          self.batch_id,
				"sample_latency_ms": round(self.sample_latency_ms, 2),
			},
		}


@dataclass
class BatchResult:
	query:   str
	results: list[RerankResult] = field(default_factory=list)

	# perf filled in after batch completes
	total_samples:             int   = 0
	total_batches:             int   = 0
	total_time_sec:            float = 0.0
	avg_sample_ms:             float = 0.0
	throughput_samples_per_sec: float = 0.0

	def to_dict(self) -> dict:
		return {
			"query": self.query,
			"perf": {
				"total_samples":              self.total_samples,
				"total_batches":              self.total_batches,
				"total_time_sec":             round(self.total_time_sec, 3),
				"avg_sample_ms":              round(self.avg_sample_ms, 2),
				"throughput_samples_per_sec": round(self.throughput_samples_per_sec, 2),
			},
			"results": [r.to_dict() for r in self.results],
		}

	def to_legacy_list(self) -> list[dict]:
		"""
		Tương thích ngược với code cũ (v1) — trả list[dict] y chang format cũ.

		Code cũ trả list[dict] với:
		  - tất cả fields gốc từ CLIP record (keyframe_path, video_id, ...)
		  - vlm_score         → float
		  - retrieval_score   → final_score (overwrite)
		  - alignment_score   → final_score (overwrite)
		  - rank, display_rank
		  (temporal thêm: video_score, matched_sequence đã có vlm_score per-frame)

		V2 extras được gắn thêm vào mà không conflict:
		  - rerank            → dict (full schema v2)
		  - scores            → {clip_score, vlm_score, final_score}
		  - perf_sample       → {batch_id, sample_latency_ms}

		Dùng method này để KHÔNG cần đổi bất kỳ downstream caller nào.

		Example
		───────
		# Code cũ:
		#   results = reranker.rerank_semantic_frames(candidates, query)
		#   for r in results: print(r["retrieval_score"])
		#
		# Code mới — drop-in replace:
		#   batch   = reranker.rerank_semantic_frames(candidates, query)
		#   results = batch.to_legacy_list()
		#   for r in results: print(r["retrieval_score"])   # ✓ vẫn hoạt động
		"""
		out: list[dict] = []
		for rank, res in enumerate(self.results, start=1):
			original = getattr(res, "_original", {})
			record   = original.copy()

			# Keys mà code cũ inject
			record["vlm_score"]       = res.vlm_score
			record["retrieval_score"] = res.final_score    # code cũ overwrite bằng final
			record["alignment_score"] = res.final_score
			record["rank"]            = rank
			record["display_rank"]    = rank

			# V2 extras — không conflict với key cũ nào
			r_dict = res.to_dict()
			record["rerank"]      = r_dict["rerank"]
			record["scores"]      = r_dict["scores"]
			record["perf_sample"] = r_dict["perf"]

			# Temporal: gắn lại matched_sequence đã enrich
			frames = getattr(res, "_frames", None)
			if frames is not None:
				record["matched_sequence"] = frames
				record["video_score"]      = res.final_score

			out.append(record)
		return out


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

# Cache transform per input_size — tránh rebuild T.Compose mỗi lần gọi _load_pixel_values
_TRANSFORM_CACHE: dict[int, T.Compose] = {}

def _build_transform(input_size: int = 448) -> T.Compose:
	if input_size not in _TRANSFORM_CACHE:
		_TRANSFORM_CACHE[input_size] = T.Compose([
			T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
			T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
			T.ToTensor(),
			T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
		])
	return _TRANSFORM_CACHE[input_size]


def _find_closest_aspect_ratio(
	aspect_ratio: float,
	target_ratios: list[tuple[int, int]],
	width: int,
	height: int,
	image_size: int,
) -> tuple[int, int]:
	best_diff  = float("inf")
	best_ratio = (1, 1)
	area = width * height
	for ratio in target_ratios:
		diff = abs(aspect_ratio - ratio[0] / ratio[1])
		if diff < best_diff:
			best_diff  = diff
			best_ratio = ratio
		elif diff == best_diff:
			if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
				best_ratio = ratio
	return best_ratio


def _dynamic_preprocess(
	image: Image.Image,
	min_num: int = 1,
	max_num: int = 4,          # ≤4 tile → tiết kiệm VRAM hơn code gốc (6 tile)
	image_size: int = 448,
	use_thumbnail: bool = True,
) -> list[Image.Image]:
	orig_w, orig_h = image.size
	target_ratios  = sorted(
		{(i, j)
		 for n in range(min_num, max_num + 1)
		 for i in range(1, n + 1)
		 for j in range(1, n + 1)
		 if min_num <= i * j <= max_num},
		key=lambda x: x[0] * x[1],
	)
	best     = _find_closest_aspect_ratio(orig_w / orig_h, target_ratios, orig_w, orig_h, image_size)
	target_w = image_size * best[0]
	target_h = image_size * best[1]
	resized  = image.resize((target_w, target_h))

	tiles: list[Image.Image] = []
	cols = target_w // image_size
	for i in range(best[0] * best[1]):
		box = (
			(i % cols) * image_size,
			(i // cols) * image_size,
			(i % cols + 1) * image_size,
			(i // cols + 1) * image_size,
		)
		tiles.append(resized.crop(box))

	if use_thumbnail and len(tiles) != 1:
		tiles.append(image.resize((image_size, image_size)))
	return tiles


def _load_pixel_values(
	image_path: str | Path,
	input_size: int = 448,
	max_num: int = 4,
) -> torch.Tensor | None:
	try:
		img    = Image.open(image_path).convert("RGB")
		xform  = _build_transform(input_size)
		tiles  = _dynamic_preprocess(img, image_size=input_size, max_num=max_num)
		tensor = torch.stack([xform(t) for t in tiles])          # (N,3,H,W)
		return tensor.to(torch.bfloat16)
	except Exception as exc:
		log.warning("Cannot open image %s: %s", image_path, exc)
		return None


# ─────────────────────────────────────────────────────────────────────────────
# Structured prompt — multi-criteria, JSON response
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
	"You are a precise visual relevance evaluator. "
	"You respond ONLY with valid JSON — no markdown, no prose."
)

# Giải thích ngắn từng tiêu chí — giúp model hiểu RÕ phải chấm cái gì cho mỗi field,
# thay vì đoán mò từ tên field. Mỗi dòng cố tình ngắn để không làm prompt dài quá
# mức cần thiết (prompt dài không phải bottleneck — generation time mới là bottleneck,
# xem _resolve_eos_token_ids).
_CRITERIA_GUIDE = (
	"object_match: are the specific objects/entities named in the query present?\n"
	"action_match: is the action/event in the query actually happening?\n"
	"scene_match: does the setting/location match (indoor/outdoor, street, etc.)?\n"
	"attribute_match: do described attributes match (color, size, shape, brand)?\n"
	"relation_match: is the spatial/social relation between entities correct?\n"
	"lighting_match: does lighting/time-of-day match if mentioned (day/night)?\n"
	"count_match: does the number of instances match if a count is implied?\n"
	"text_match: if the query mentions visible text/signage, is it present?\n"
	"emotion_match: does mood/emotion expressed match if mentioned?\n"
	"style_match: does visual style match if mentioned (cartoon, photo, etc.)\n"
	"If the query gives no info for a criterion, score it 0.5 (neutral), "
	"do NOT score it 0.0."
)

# Giải thích band quyết định — để model "tự ý thức" được nó nên chấm điểm tổng thể
# rơi vào vùng nào, giảm lệch giữa relevance_score và decision label.
_DECISION_GUIDE = (
	"Overall relevance bands (for your own calibration, you do not output the label):\n"
	"  strong_match >=0.75 : image is an excellent, unambiguous match\n"
	"  relevant     >=0.45 : image matches the core of the query, minor mismatches ok\n"
	"  weak_match   >=0.20 : only a loose/partial match, several elements missing\n"
	"  irrelevant   < 0.20 : image does not depict what the query asks for"
)

def _build_structured_prompt(query: str) -> str:
	"""
	Prompt không dùng JSON template sẵn với số 0.0 — model sẽ copy y chang template
	thay vì evaluate thật. Thay vào đó dùng instruction rõ ràng + ví dụ khác nhau
	để model tự điền giá trị thực, kèm giải thích từng tiêu chí + band quyết định
	để tăng độ chính xác.

	Lưu ý: prompt dài hơn KHÔNG làm chậm pipeline đáng kể — token hoá prompt là một
	forward pass duy nhất, rẻ hơn nhiều so với generation token-by-token. Bottleneck
	thật sự nằm ở số token được SINH RA (xem _resolve_eos_token_ids để fix root cause
	của vấn đề "phải set 2048 token mới ổn").
	"""
	return (
		"<image>\n"
		f'Score how well this image matches the query: "{query}"\n\n'
		f"{_CRITERIA_GUIDE}\n\n"
		f"{_DECISION_GUIDE}\n\n"
		"Give each criterion a score from 0.0 (not matching) to 1.0 (perfect match), "
		"then a confidence (0-1) on your own judgment, and a one-sentence explanation "
		"(max 20 words) of what is actually in the image.\n\n"
		"Example of a good response:\n"
		'{"object_match":0.9,"action_match":0.7,"scene_match":0.8,"attribute_match":0.6,'
		'"relation_match":0.5,"lighting_match":0.4,"count_match":0.8,"text_match":0.1,'
		'"emotion_match":0.3,"style_match":0.5,"confidence":0.8,"explanation":"cars and person visible"}\n\n'
		"Now score THIS image (use your own judgment, not the example values, "
		"output ONLY the JSON object, nothing after the closing brace):\n"
		'{"object_match":'
	)


# ─────────────────────────────────────────────────────────────────────────────
# JSON parser — robust to minor model hallucinations
# ─────────────────────────────────────────────────────────────────────────────

_FLOAT_KEYS = [
	"object_match", "action_match", "scene_match", "attribute_match",
	"relation_match", "lighting_match", "count_match", "text_match",
	"emotion_match", "style_match", "confidence",
]


def _parse_vlm_response(raw: str) -> tuple[ReasoningScores, float, str]:
	"""
	Parse VLM response → (ReasoningScores, confidence, explanation).

	Chiến lược: regex extract từng key trực tiếp từ raw string.
	Hoạt động với:
	  - JSON hoàn chỉnh           {"object_match":0.8,...}
	  - JSON bị truncate          {"object_match":0.8,"action_match":0.   ← hết token
	  - JSON trong markdown fence ```json\\n{...}\\n```
	  - Prose với số inline       object_match: 0.8, action_match: 0.7
	Không phụ thuộc vào dấu } đóng.
	"""
	def _clamp(v: Any, default: float = 0.0) -> float:
		try:
			return max(0.0, min(1.0, float(v)))
		except (TypeError, ValueError):
			return default

	# Strip markdown fences
	cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

	# ── Bước 1: thử json.loads ────────────────────────────────────────────────
	data: dict = {}
	# Prompt kết thúc bằng '{"object_match":' → model trả continuation không có {
	# Thêm prefix để reconstruct JSON hoàn chỉnh
	for attempt in [cleaned, '{"object_match":' + cleaned, cleaned + "}", '{"object_match":' + cleaned + "}"]:
		brace = attempt.find("{")
		if brace == -1:
			continue
		try:
			data = json.loads(attempt[brace:])
			break
		except json.JSONDecodeError:
			continue

	# ── Bước 2: regex extract từng key — hoạt động ngay cả khi data={} ───────
	for key in _FLOAT_KEYS:
		if key not in data:
			# khớp cả "key":0.8 lẫn key: 0.8
			hit = re.search(
				rf'["\s]?{re.escape(key)}["\s]?\s*:\s*([0-9]+(?:\.[0-9]*)?)',
				cleaned,
			)
			if hit:
				try:
					data[key] = float(hit.group(1))
				except ValueError:
					pass

	# explanation
	if "explanation" not in data:
		hit = re.search(r'"explanation"\s*:\s*"([^"]*)"', cleaned)
		if hit:
			data["explanation"] = hit.group(1)

	# Nếu không extract được bất kỳ float nào → thực sự parse_error
	found_any = any(key in data for key in _FLOAT_KEYS)
	if not found_any:
		log.warning("VLM parse_error — no JSON found. Raw response: %r", raw[:200])
		return ReasoningScores(), 0.0, "parse_error"

	scores = ReasoningScores(
		object_match    = _clamp(data.get("object_match")),
		action_match    = _clamp(data.get("action_match")),
		scene_match     = _clamp(data.get("scene_match")),
		attribute_match = _clamp(data.get("attribute_match")),
		relation_match  = _clamp(data.get("relation_match")),
		lighting_match  = _clamp(data.get("lighting_match")),
		count_match     = _clamp(data.get("count_match")),
		text_match      = _clamp(data.get("text_match")),
		emotion_match   = _clamp(data.get("emotion_match")),
		style_match     = _clamp(data.get("style_match")),
	)
	confidence  = _clamp(data.get("confidence", 0.5))
	explanation = str(data.get("explanation", ""))[:300]

	return scores, confidence, explanation


def _score_to_decision(score: float) -> str:
	for threshold, label in _DECISION_THRESHOLDS:
		if score >= threshold:
			return label
	return "irrelevant"


# ─────────────────────────────────────────────────────────────────────────────
# Model loader — device map helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_device_map(model_path: str) -> str | dict:
	"""
	Single GPU → "cuda:0" (ép toàn bộ lên GPU 0, tránh OOM với int-8).
	Multi GPU  → layer-split map.
	No GPU     → "cpu".
	"""
	if not torch.cuda.is_available():
		return "cpu"

	n_gpus = torch.cuda.device_count()
	if n_gpus == 1:
		return "cuda:0"

	# Multi-GPU split (giống code gốc nhưng cho num_layers động)
	try:
		from transformers import AutoConfig
		cfg        = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
		num_layers = cfg.llm_config.num_hidden_layers
	except Exception:
		return "cuda:0"

	layers_per_gpu    = [math.ceil(num_layers / (n_gpus - 0.5))] * n_gpus
	layers_per_gpu[0] = math.ceil(layers_per_gpu[0] * 0.5)

	device_map: dict[str, int] = {}
	cnt = 0
	for gpu_id, n in enumerate(layers_per_gpu):
		for _ in range(n):
			if cnt < num_layers:
				device_map[f"language_model.model.layers.{cnt}"] = gpu_id
				cnt += 1

	for key in (
		"vision_model", "mlp1",
		"language_model.model.tok_embeddings",
		"language_model.model.embed_tokens",
		"language_model.output",
		"language_model.model.norm",
		"language_model.model.rotary_emb",
		"language_model.lm_head",
		f"language_model.model.layers.{num_layers - 1}",
	):
		device_map[key] = 0

	return device_map


# Các candidate stop-token thường gặp ở chat template của InternVL (ChatML / InternLM2).
# Nếu tokenizer.eos_token_id sẵn có và hợp lệ thì vẫn ưu tiên dùng, các token này chỉ
# bổ sung thêm để model có nhiều "cửa" để dừng sớm.
# Nếu tokenizer.eos_token_id sẵn có và hợp lệ thì vẫn ưu tiên dùng, các token này chỉ
_CHATML_STOP_CANDIDATES = ("<|im_end|>", "<|end|>", "</s>")


def _resolve_eos_token_ids(tokenizer: Any) -> list[int]:
	"""
	ROOT-CAUSE FIX cho log "Setting pad_token_id to eos_token_id:None" và cho việc
	phải set max_new_tokens=2048 mới "ổn".

	Khi eos_token_id không resolve được, HF generate() không có điều kiện dừng nào
	khác ngoài max_new_tokens → model luôn sinh đủ toàn bộ max_new_tokens, kể cả khi
	JSON đã đóng từ lâu. Đó là lý do mỗi sample tốn 30-48s (gần như y nhau bất kể nội
	dung ảnh) và lý do giảm max_new_tokens xuống thấp lại làm JSON/explanation bị cắt
	nửa chừng — model chưa "biết" nó nên dừng ngay sau dấu '}'.

	Hàm này tìm tất cả token id hợp lệ có thể đóng vai trò eos (tokenizer.eos_token_id
	gốc + các stop-token phổ biến của chat template ChatML/InternLM2 nếu tồn tại trong
	vocab), trả về list để truyền vào generation_config["eos_token_id"]. generate() hỗ
	trợ eos_token_id là list — dừng khi gặp BẤT KỲ token nào trong list.
	"""
	ids: set[int] = set()

	base_id = getattr(tokenizer, "eos_token_id", None)
	if isinstance(base_id, int) and base_id >= 0:
		ids.add(base_id)

	for tok in _CHATML_STOP_CANDIDATES:
		try:
			tid = tokenizer.convert_tokens_to_ids(tok)
		except Exception:
			continue
		unk_id = getattr(tokenizer, "unk_token_id", None)
		if isinstance(tid, int) and tid >= 0 and tid != unk_id:
			ids.add(tid)

	if not ids:
		log.warning(
			"Không resolve được eos_token_id nào — generation sẽ luôn chạy hết "
			"max_new_tokens (chậm). Kiểm tra lại tokenizer/model_id."
		)
	else:
		log.info("Resolved eos_token_id(s) cho early-stop: %s", sorted(ids))

	return sorted(ids)


# ─────────────────────────────────────────────────────────────────────────────
# Main Reranker class
# ─────────────────────────────────────────────────────────────────────────────

class InternVLReranker:
	"""
	Structured VLM Reranker — InternVL3-2B int-8.

	Parameters
	----------
	model_id   : HuggingFace model id (default InternVL3-2B)
	alpha      : pha trộn CLIP vs VLM.  final = alpha*clip + (1-alpha)*vlm_relevance
	batch_size : số sample xử lý song song trong một lần (tăng throughput nhưng tốn VRAM)
	max_tiles  : số tile tối đa khi dynamic preprocess (4 = cân bằng VRAM/accuracy)

	Output schema per sample
	──────────────────────────
	Xem dataclass RerankResult / RerankResult.to_dict()

	Downstream usage
	─────────────────
	Kết quả to_dict() là dict thuần Python — serialize JSON, forward trực tiếp
	vào bất kỳ pipeline nào mà không cần thay đổi gì.
	"""

	def __init__(
		self,
		model_id:         str   = "OpenGVLab/InternVL3-2B",
		device:           str   = "cuda",   # kept for API compat với retrieval_system — thực tế dùng device_map
		alpha:            float = 0.4,
		batch_size:       int   = 4,
		max_tiles:        int   = 4,
		max_new_tokens:   int   = 1024,   # an toàn cho JSON 10 field + explanation 20 từ;
										 # KHÔNG cần 2048 nữa vì eos đã resolve đúng (xem
										 # _resolve_eos_token_ids) — model tự dừng ngay sau
										 # khi đóng JSON, số này chỉ là ceiling an toàn.
		use_true_batch:   bool  = True,  # dùng model.batch_chat() để xử lý cả batch trong
										 # 1 forward pass thay vì loop tuần tự từng ảnh
	) -> None:
		self.alpha          = float(alpha)
		self.model_id       = model_id
		self.batch_size     = max(1, batch_size)
		self.max_tiles      = max_tiles
		self.use_true_batch = use_true_batch

		log.info("Loading %s | int-8 BitsAndBytes | batch_size=%d | max_tiles=%d",
				 model_id, self.batch_size, self.max_tiles)

		self.tokenizer = AutoTokenizer.from_pretrained(
			model_id, trust_remote_code=True, use_fast=False,
		)

		has_cuda = torch.cuda.is_available()
		bnb_cfg  = (
			BitsAndBytesConfig(
				load_in_8bit=True,
				llm_int8_threshold=6.0,
				llm_int8_has_fp16_weight=False,
			) if has_cuda else None
		)

		load_kwargs: dict = dict(
			torch_dtype        = torch.bfloat16 if has_cuda else torch.float32,
			low_cpu_mem_usage  = True,
			use_flash_attn     = False,
			trust_remote_code  = True,
			device_map         = _build_device_map(model_id),
		)
		if bnb_cfg is not None:
			load_kwargs["quantization_config"] = bnb_cfg

		self.model = AutoModel.from_pretrained(model_id, **load_kwargs).eval()
		self._device = "cuda:0" if has_cuda else "cpu"

		# ── ROOT-CAUSE FIX: resolve eos_token_id thật ───────────────────────────
		# Trước đây eos_token_id=None → generate() luôn chạy hết max_new_tokens
		# (38-48s/sample). Giờ model tự dừng ngay khi sinh xong token kết thúc câu,
		# nên max_new_tokens chỉ còn là ceiling an toàn, không phải target.
		eos_ids = _resolve_eos_token_ids(self.tokenizer)
		pad_id  = eos_ids[0] if eos_ids else self.tokenizer.eos_token_id

		self._gen_cfg = dict(
			max_new_tokens = max_new_tokens,
			do_sample      = False,
			num_beams      = 1,     # beam=1 + greedy → nhanh nhất, ít VRAM nhất
		)
		if eos_ids:
			self._gen_cfg["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]
		if pad_id is not None:
			self._gen_cfg["pad_token_id"] = pad_id   # tắt warning "Setting pad_token_id..."

		# generation_config riêng cho fallback YES/NO (vẫn dùng eos đã resolve)
		self._gen_cfg_fallback = dict(self._gen_cfg)
		self._gen_cfg_fallback["max_new_tokens"] = 4

		# Phát hiện xem model có hỗ trợ batch_chat thật (API gốc InternVL) không.
		self._has_batch_chat = hasattr(self.model, "batch_chat")
		if self.use_true_batch and not self._has_batch_chat:
			log.warning(
				"Model %s không có method batch_chat() — sẽ fallback xử lý tuần tự "
				"từng ảnh trong mỗi batch (vẫn đúng, chỉ chậm hơn).",
				model_id,
			)

		log.info(
			"InternVLReranker ready ✓ device=%s | max_new_tokens=%d | true_batch=%s",
			self._device, max_new_tokens, self._has_batch_chat and self.use_true_batch,
		)

	# ─────────────────────────────────────────────────────────────────────────
	# Helper: parse raw VLM string → RerankResult hoàn chỉnh (dùng chung cho cả
	# _score_one và _score_chunk/true-batch)
	# ─────────────────────────────────────────────────────────────────────────

	def _finalize_from_raw(
		self,
		raw: str,
		query: str,
		sample_id: str,
		clip_score: float,
		batch_id: int,
		t0: float,
		pv_for_fallback: torch.Tensor | None,
	) -> RerankResult:
		result = RerankResult(sample_id=sample_id, clip_score=clip_score, batch_id=batch_id)

		log.debug("[%s] raw VLM response: %r", sample_id, raw[:300])
		reasoning, confidence, explanation = _parse_vlm_response(raw)

		# Fallback: nếu JSON parse fail, thử lại với prompt YES/NO đơn giản hơn
		# trên cùng pixel_values đã load (không re-read ảnh)
		if explanation == "parse_error" and pv_for_fallback is not None:
			log.warning("[%s] JSON parse failed — retrying with YES/NO fallback prompt", sample_id)
			try:
				raw2: str = self.model.chat(
					self.tokenizer,
					pv_for_fallback,
					f"<image>\nDoes this image match: \"{query}\"? Reply YES or NO only.",
					self._gen_cfg_fallback,
					history=None,
					return_history=False,
				)
				log.debug("[%s] fallback raw: %r", sample_id, raw2[:60])
				is_yes = raw2.strip().lower().startswith("yes")
				# Map YES/NO → flat score dùng tạm cho relevance
				flat = 0.75 if is_yes else 0.10
				reasoning = ReasoningScores(
					object_match=flat, action_match=flat, scene_match=flat,
					attribute_match=flat, relation_match=flat,
				)
				confidence  = 0.5
				explanation = f"fallback:{'YES' if is_yes else 'NO'}"
			except Exception as exc2:
				log.error("[%s] fallback also failed: %s", sample_id, exc2)

		relevance_score = reasoning.weighted_mean
		decision        = _score_to_decision(relevance_score)

		result.reasoning        = reasoning
		result.relevance_score  = round(relevance_score, 4)
		result.confidence       = round(confidence, 4)
		result.decision         = decision
		result.explanation      = explanation
		result.vlm_score        = round(relevance_score, 4)
		result.final_score      = round(
			self.alpha * clip_score + (1 - self.alpha) * relevance_score, 4
		)
		result.sample_latency_ms = round((time.perf_counter() - t0) * 1000, 2)

		log.info(
			"[%s] batch=%d | decision=%-12s | relevance=%.3f | clip=%.3f | "
			"final=%.3f | latency=%.1fms | %s",
			sample_id, batch_id, decision,
			relevance_score, clip_score, result.final_score,
			result.sample_latency_ms, explanation[:120],
		)
		return result

	# ─────────────────────────────────────────────────────────────────────────
	# Core: score one image → RerankResult (partial, không có scores/perf)
	# ─────────────────────────────────────────────────────────────────────────

	@torch.inference_mode()
	def _score_one(
		self,
		image_path: str | Path,
		query: str,
		sample_id: str,
		clip_score: float,
		batch_id: int,
	) -> RerankResult:
		t0     = time.perf_counter()
		result = RerankResult(sample_id=sample_id, clip_score=clip_score, batch_id=batch_id)

		path = Path(image_path)
		if not path.exists():
			log.warning("[%s] Image not found: %s", sample_id, path)
			result.explanation = "image_not_found"
			result.sample_latency_ms = (time.perf_counter() - t0) * 1000
			return result

		pv = _load_pixel_values(path, max_num=self.max_tiles)
		if pv is None:
			result.explanation = "image_load_error"
			result.sample_latency_ms = (time.perf_counter() - t0) * 1000
			return result

		pv = pv.to(self._device)

		try:
			raw: str = self.model.chat(
				self.tokenizer,
				pv,
				_build_structured_prompt(query),
				self._gen_cfg,
				history=None,
				return_history=False,
			)
		except Exception as exc:
			log.error("[%s] model.chat failed: %s: %s", sample_id, type(exc).__name__, exc)
			result.sample_latency_ms = (time.perf_counter() - t0) * 1000
			return result

		return self._finalize_from_raw(
			raw, query, sample_id, clip_score, batch_id, t0, pv_for_fallback=pv,
		)

	# ─────────────────────────────────────────────────────────────────────────
	# True batch scoring — gửi NHIỀU ảnh trong CÙNG 1 forward pass qua
	# model.batch_chat(). Đây là fix thật cho "tối ưu xử lý theo batch": trước đây
	# batch_size chỉ dùng để group log, mọi ảnh vẫn được chat() tuần tự 1-by-1.
	# ─────────────────────────────────────────────────────────────────────────

	@torch.inference_mode()
	def _score_chunk(
		self,
		items_meta: list[tuple[str, str, str, float]],
		batch_id: int,
	) -> list[RerankResult]:
		"""
		items_meta: list các tuple (image_path, query, sample_id, clip_score).

		Nếu model không hỗ trợ batch_chat() (kiểm tra ở __init__) hoặc
		use_true_batch=False → fallback tuần tự _score_one (vẫn đúng, chỉ chậm hơn).
		Nếu batch_chat() raise exception ở runtime (OOM, version mismatch...) →
		cũng fallback tuần tự cho riêng chunk đó, KHÔNG làm crash toàn pipeline.
		"""
		if not self._has_batch_chat or not self.use_true_batch:
			return [
				self._score_one(image_path=p, query=q, sample_id=sid, clip_score=cs, batch_id=batch_id)
				for (p, q, sid, cs) in items_meta
			]

		t_batch0 = time.perf_counter()

		pv_list:     list[torch.Tensor] = []
		meta_valid:  list[tuple]        = []
		early_results: list[RerankResult] = []

		for (image_path, query, sample_id, clip_score) in items_meta:
			path = Path(image_path)
			if not path.exists():
				log.warning("[%s] Image not found: %s", sample_id, path)
				r = RerankResult(sample_id=sample_id, clip_score=clip_score, batch_id=batch_id)
				r.explanation = "image_not_found"
				early_results.append(r)
				continue
			pv = _load_pixel_values(path, max_num=self.max_tiles)
			if pv is None:
				r = RerankResult(sample_id=sample_id, clip_score=clip_score, batch_id=batch_id)
				r.explanation = "image_load_error"
				early_results.append(r)
				continue
			pv_list.append(pv)
			meta_valid.append((image_path, query, sample_id, clip_score, pv))

		if not meta_valid:
			return early_results

		pixel_values     = torch.cat(pv_list, dim=0).to(self._device)
		num_patches_list = [pv.size(0) for pv in pv_list]
		questions        = [_build_structured_prompt(q) for (_, q, _, _, _) in meta_valid]

		try:
			raw_list: list[str] = self.model.batch_chat(
				self.tokenizer,
				pixel_values,
				num_patches_list  = num_patches_list,
				questions         = questions,
				generation_config = self._gen_cfg,
			)
		except Exception as exc:
			log.warning(
				"batch_chat thất bại (%s: %s) — fallback tuần tự cho %d sample trong batch %d",
				type(exc).__name__, exc, len(meta_valid), batch_id,
			)
			seq_results = [
				self._score_one(image_path=p, query=q, sample_id=sid, clip_score=cs, batch_id=batch_id)
				for (p, q, sid, cs, _pv) in meta_valid
			]
			return early_results + seq_results

		t_batch_total = time.perf_counter() - t_batch0
		per_item_ms   = t_batch_total * 1000 / max(len(meta_valid), 1)

		log.info(
			"  ↳ true-batch forward: %d ảnh trong %.1fms (~%.1fms/ảnh)",
			len(meta_valid), t_batch_total * 1000, per_item_ms,
		)

		results: list[RerankResult] = []
		for (image_path, query, sample_id, clip_score, pv_single), raw in zip(meta_valid, raw_list):
			# Batch thật không có "latency riêng từng ảnh" — chia đều thời gian batch
			# để vẫn điền được sample_latency_ms cho schema downstream.
			fake_t0 = time.perf_counter() - (per_item_ms / 1000)
			res = self._finalize_from_raw(
				raw, query, sample_id, clip_score, batch_id, fake_t0,
				pv_for_fallback=pv_single.to(self._device),
			)
			results.append(res)

		return early_results + results

	# ─────────────────────────────────────────────────────────────────────────
	# Semantic rerank — single frames
	# ─────────────────────────────────────────────────────────────────────────

	def rerank_semantic_frames(
		self,
		candidate_records: list[dict],
		query: str,
	) -> BatchResult:
		"""
		Nhận list frame records từ CLIP, chấm VLM, trả BatchResult.

		Mỗi record cần có:
		  keyframe_path           : đường dẫn ảnh
		  retrieval_score /
			alignment_score /
			score               : điểm CLIP gốc
		  [sample_id]            : tuỳ chọn, tự sinh nếu thiếu

		Return
		──────
		BatchResult  — gọi .to_dict() để serialize / forward downstream.
		Trường results đã sắp xếp theo final_score giảm dần.
		Mỗi item trong results vẫn giữ toàn bộ fields gốc của record qua
		field `_original` để downstream có thể dùng mà không mất dữ liệu.
		"""
		if not candidate_records:
			return BatchResult(query=query)

		t_start = time.perf_counter()
		chunks  = [
			candidate_records[i: i + self.batch_size]
			for i in range(0, len(candidate_records), self.batch_size)
		]

		all_results: list[RerankResult] = []

		log.info("Rerank [semantic] query='%s' | %d samples | %d batches | batch_size=%d",
				 query, len(candidate_records), len(chunks), self.batch_size)

		for b_idx, chunk in enumerate(chunks, start=1):
			log.info("── Batch %d/%d (%d samples)", b_idx, len(chunks), len(chunk))

			items_meta: list[tuple[str, str, str, float]] = []
			for item in chunk:
				sid = str(item.get("sample_id", item.get("frame_id", f"s{len(all_results) + len(items_meta):04d}")))
				cs  = float(item.get("retrieval_score",
							item.get("alignment_score",
							item.get("score", 0.0))))
				items_meta.append((item.get("keyframe_path", ""), query, sid, cs))

			chunk_results = self._score_chunk(items_meta, batch_id=b_idx)

			# Map kết quả trở lại original record theo sample_id (true-batch có thể
			# trả về theo thứ tự khác nếu có item bị early-result trước)
			by_sid = {r.sample_id: r for r in chunk_results}
			for item, (_, _, sid, _cs) in zip(chunk, items_meta):
				res = by_sid.get(sid)
				if res is None:
					continue
				res._original = item   # type: ignore[attr-defined]
				all_results.append(res)

		all_results.sort(key=lambda r: r.final_score, reverse=True)
		t_total = time.perf_counter() - t_start

		batch_res = BatchResult(
			query                    = query,
			results                  = all_results,
			total_samples            = len(all_results),
			total_batches            = len(chunks),
			total_time_sec           = round(t_total, 3),
			avg_sample_ms            = round(t_total * 1000 / max(len(all_results), 1), 2),
			throughput_samples_per_sec = round(len(all_results) / max(t_total, 1e-9), 2),
		)

		log.info(
			"Done [semantic] | total=%.2fs | avg=%.1fms | throughput=%.1f sps",
			batch_res.total_time_sec,
			batch_res.avg_sample_ms,
			batch_res.throughput_samples_per_sec,
		)
		return batch_res

	# ─────────────────────────────────────────────────────────────────────────
	# Temporal rerank — video sequences
	# ─────────────────────────────────────────────────────────────────────────

	def rerank_temporal_sequences(
		self,
		candidate_records: list[dict],
		query: str = "",           # primary param — retrieval_system gọi không truyền query
		default_query: str = "",   # alias kept for backward compat — query takes precedence
	) -> BatchResult:
		"""
		Nhận list temporal sequence records, chấm VLM từng frame.

		Mỗi record cần có:
		  matched_sequence : list[dict] với keyframe_path + sub_query
		  video_score      : float — điểm CLIP tổng của sequence
		  [sample_id]      : tuỳ chọn

		Return  BatchResult  (xem rerank_semantic_frames).

		Compatibility
		─────────────
		retrieval_system.py gọi:  reranker.rerank_temporal_sequences(records)
		Nên query="" là mặc định an toàn — sub_query trong từng frame được dùng.
		"""
		effective_query = query or default_query
		if not candidate_records:
			return BatchResult(query=effective_query)

		t_start    = time.perf_counter()
		all_results: list[RerankResult] = []
		total_batches = 0

		log.info("Rerank [temporal] | %d sequences", len(candidate_records))

		for seq_idx, item in enumerate(candidate_records):
			seq: list[dict] = item.get("matched_sequence", [])
			if not seq:
				log.warning("Sequence %d has no frames — skipped", seq_idx)
				continue

			sid        = str(item.get("sample_id", item.get("video_id", f"seq{seq_idx:04d}")))
			clip_score = float(item.get("video_score", item.get("score", 0.0)))

			# Score each sub-frame
			frame_reasoning_list: list[ReasoningScores] = []
			frame_conf_list:      list[float]            = []
			updated_seq:          list[dict]             = []

			chunks = [seq[i: i + self.batch_size] for i in range(0, len(seq), self.batch_size)]
			total_batches += len(chunks)

			log.info("  Sequence %s | %d frames | %d sub-batches", sid, len(seq), len(chunks))

			for b_idx, chunk in enumerate(chunks, start=1):
				items_meta: list[tuple[str, str, str, float]] = [
					(
						frame.get("keyframe_path", ""),
						frame.get("sub_query", effective_query),
						f"{sid}_f{len(updated_seq) + i:03d}",
						float(frame.get("score", 0.0)),
					)
					for i, frame in enumerate(chunk)
				]
				chunk_results = self._score_chunk(items_meta, batch_id=b_idx)
				by_sid = {r.sample_id: r for r in chunk_results}

				for frame, (_, _, sub_sid, _cs) in zip(chunk, items_meta):
					sub_res = by_sid.get(sub_sid)
					if sub_res is None:
						continue
					frame_reasoning_list.append(sub_res.reasoning)
					frame_conf_list.append(sub_res.confidence)

					f_updated = frame.copy()
					f_updated["vlm_score"]       = sub_res.relevance_score
					f_updated["final_score"]     = sub_res.final_score
					f_updated["decision"]        = sub_res.decision
					f_updated["reasoning"]       = sub_res.reasoning.to_dict()
					f_updated["explanation"]     = sub_res.explanation
					updated_seq.append(f_updated)

			# Aggregate frame scores → sequence-level ReasoningScores (mean per dim)
			agg = ReasoningScores(
				object_match    = float(np.mean([r.object_match    for r in frame_reasoning_list])),
				action_match    = float(np.mean([r.action_match    for r in frame_reasoning_list])),
				scene_match     = float(np.mean([r.scene_match     for r in frame_reasoning_list])),
				attribute_match = float(np.mean([r.attribute_match for r in frame_reasoning_list])),
				relation_match  = float(np.mean([r.relation_match  for r in frame_reasoning_list])),
				lighting_match  = float(np.mean([r.lighting_match  for r in frame_reasoning_list])),
				count_match     = float(np.mean([r.count_match     for r in frame_reasoning_list])),
				text_match      = float(np.mean([r.text_match      for r in frame_reasoning_list])),
				emotion_match   = float(np.mean([r.emotion_match   for r in frame_reasoning_list])),
				style_match     = float(np.mean([r.style_match     for r in frame_reasoning_list])),
			)
			agg_relevance  = agg.weighted_mean
			agg_confidence = float(np.mean(frame_conf_list)) if frame_conf_list else 0.0
			final_score    = round(self.alpha * clip_score + (1 - self.alpha) * agg_relevance, 4)

			seq_result = RerankResult(
				sample_id        = sid,
				relevance_score  = round(agg_relevance, 4),
				confidence       = round(agg_confidence, 4),
				decision         = _score_to_decision(agg_relevance),
				reasoning        = agg,
				explanation      = f"{len(seq)} frames aggregated",
				clip_score       = clip_score,
				vlm_score        = round(agg_relevance, 4),
				final_score      = final_score,
				batch_id         = seq_idx + 1,
				sample_latency_ms= 0.0,  # perf is per-frame above
			)
			seq_result._original    = item          # type: ignore[attr-defined]
			seq_result._frames      = updated_seq   # type: ignore[attr-defined]
			all_results.append(seq_result)

			log.info(
				"[%s] seq decision=%-12s | agg_relevance=%.3f | final=%.3f",
				sid, seq_result.decision, agg_relevance, final_score,
			)

		all_results.sort(key=lambda r: r.final_score, reverse=True)
		t_total = time.perf_counter() - t_start

		batch_res = BatchResult(
			query                      = effective_query,
			results                    = all_results,
			total_samples              = len(all_results),
			total_batches              = total_batches,
			total_time_sec             = round(t_total, 3),
			avg_sample_ms              = round(t_total * 1000 / max(len(all_results), 1), 2),
			throughput_samples_per_sec = round(len(all_results) / max(t_total, 1e-9), 2),
		)

		log.info(
			"Done [temporal] | sequences=%d | total=%.2fs | throughput=%.1f sps",
			len(all_results), batch_res.total_time_sec, batch_res.throughput_samples_per_sec,
		)
		return batch_res

	# ─────────────────────────────────────────────────────────────────────────
	# Convenience: auto-detect pipeline type
	# ─────────────────────────────────────────────────────────────────────────

	def rerank(
		self,
		candidate_records: list[dict],
		query: str,
	) -> BatchResult:
		"""
		Auto-detect pipeline:
		  - nếu record có 'matched_sequence' → temporal
		  - ngược lại → semantic (single frame)
		"""
		if candidate_records and "matched_sequence" in candidate_records[0]:
			return self.rerank_temporal_sequences(candidate_records, query=query)
		return self.rerank_semantic_frames(candidate_records, query)


# ─────────────────────────────────────────────────────────────────────────────
# Downstream helper — merge rerank result back into original records
# ─────────────────────────────────────────────────────────────────────────────

def merge_rerank_into_records(batch_result: BatchResult) -> list[dict]:
	"""
	Trả về list dict hoàn chỉnh:
	  original record fields  +  rerank schema v2 fields
	Có thể dùng trực tiếp trong bất kỳ task downstream nào.

	Ví dụ:
		merged = merge_rerank_into_records(batch_result)
		# merged[0] có đủ cả 'keyframe_path', 'video_id', ... lẫn 'rerank', 'scores', 'perf'
	"""
	merged: list[dict] = []
	for res in batch_result.results:
		original = getattr(res, "_original", {})
		out = {**original, **res.to_dict()}
		# Frames trong temporal (nếu có)
		frames = getattr(res, "_frames", None)
		if frames is not None:
			out["matched_sequence"] = frames
		merged.append(out)
	return merged

