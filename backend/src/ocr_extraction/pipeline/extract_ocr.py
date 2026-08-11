from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm

from src.ocr_extraction.models.dataset import build_loader
from src.ocr_extraction.models.qwen_engine import extract_text_lines, load_qwen_model
from src.ocr_extraction.models.engine import load_ocr_model

class ExtractOCRPipeline:
    """Runs OCR extraction with one of two engines (extraction.engine):

        "qwen"           -> full-frame OCR via Qwen3-VL-4B-Instruct (VLM).
                             Default. Falls back to "paddle_vietocr" at
                             runtime if the Qwen model fails to load.
        "paddle_vietocr" -> text DETECTION with PaddleOCR, text RECOGNITION
                             with either PaddleOCR's own recognizer or
                             VietOCR (extraction.recognizer: "vietocr").

    Writes one JSON file per video in the shape consumed by
    src.retrieval.indexer.elasticsearch.ocr.indexing_pipeline.IndexPipeline:

        {
          "<keyframe_id>": [ [[x1,y1,x2,y2], ...], ["text1", "text2", ...] ],
          ...
        }

    (The "qwen" engine has no native bounding boxes, so it emits a
    [0.0, 0.0, 0.0, 0.0] placeholder box per detected text line -- only
    `texts` is indexed/searched downstream, so this keeps the output shape
    compatible without changing the indexer.)

    Input layout:
        <input_keyframes_root>/<dataset>/<video_id>/<keyframe>.jpg
    Output layout:
        <output_root>/<dataset>/<video_id>.json
    """

    def __init__(self, cfg: dict, logger: logging.Logger | None = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

        extraction_cfg = cfg["extraction"]

        self.input_root = Path(extraction_cfg["input_keyframes_root"])
        self.output_root = Path(extraction_cfg["output_root"])
        self.visualize_dir = extraction_cfg.get("visualize_dir")

        self.batch_size = int(extraction_cfg.get("batch_size", 16))
        self.num_workers = int(extraction_cfg.get("num_workers", 0))
        self.skip_existing = bool(extraction_cfg.get("skip_existing", True))
        self.visualized = bool(extraction_cfg.get("visualized", False))

        # "paddleocr" (default rec model, no Vietnamese support) or
        # "vietocr" (crops from PaddleOCR detection, recognized by VietOCR).
        # Only used when engine == "paddle_vietocr".
        self.recognizer = str(extraction_cfg.get("recognizer", "paddleocr")).lower()

        self.engine = str(extraction_cfg.get("engine", "qwen")).lower()

        self.qwen_model = None
        self.qwen_processor = None
        self.qwen_cfg: dict = {}

        if self.engine == "qwen":
            qwen_cfg = extraction_cfg.get("qwen", {}) or {}
            self.qwen_cfg = qwen_cfg

            model_cfg = extraction_cfg["model"]
            self.model = load_ocr_model(
                ocr_version=model_cfg["ocr_version"],
                text_detection_model_name=model_cfg["text_detection_model_name"],
                text_recognition_model_name=None,
                use_doc_orientation_classify=model_cfg.get("use_doc_orientation_classify", False),
                use_doc_unwarping=model_cfg.get("use_doc_unwarping", False),
                use_textline_orientation=model_cfg.get("use_textline_orientation", False),
                device=model_cfg.get("device", "cpu"),
                logger=self.logger,
            )

            try:
                self.qwen_model, self.qwen_processor = load_qwen_model(
                    model_id=qwen_cfg.get("model_id", "Qwen/Qwen3-VL-4B-Instruct"),
                    device=qwen_cfg.get("device", "cuda"),
                    dtype=qwen_cfg.get("dtype", "bfloat16"),
                    logger=self.logger,
                )
                
            except Exception as e:
                self.logger.warning(
                    "Qwen load failed (%s: %s) - falling back to paddle_vietocr", type(e).__name__, e
                )
                self.engine = "paddle_vietocr"

        self.model = None
        self.vietocr_predictor = None

        if self.engine == "paddle_vietocr":

            model_cfg = extraction_cfg["model"]
            self.model = load_ocr_model(
                ocr_version=model_cfg["ocr_version"],
                text_detection_model_name=model_cfg["text_detection_model_name"],
                text_recognition_model_name=model_cfg["text_recognition_model_name"],
                use_doc_orientation_classify=model_cfg.get("use_doc_orientation_classify", False),
                use_doc_unwarping=model_cfg.get("use_doc_unwarping", False),
                use_textline_orientation=model_cfg.get("use_textline_orientation", False),
                device=model_cfg.get("device", "cpu"),
                logger=self.logger,
            )

            if self.recognizer == "vietocr":
                from src.ocr_extraction.models.vietocr_engine import load_vietocr_model

                vietocr_cfg = extraction_cfg.get("vietocr", {})
                self.vietocr_predictor = load_vietocr_model(
                    base_config=vietocr_cfg.get("base_config", "vgg_transformer"),
                    device=vietocr_cfg.get("device", model_cfg.get("device", "cpu")),
                    weights_path=vietocr_cfg.get("weights_path"),
                    logger=self.logger,
                )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def scan_video_dirs(self) -> list[Path]:
        if not self.input_root.exists():
            raise FileNotFoundError(f"Keyframes root not found: {self.input_root}")

        video_dirs = []
        for dataset_dir in sorted(self.input_root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for video_dir in sorted(dataset_dir.iterdir()):
                if video_dir.is_dir():
                    video_dirs.append(video_dir)
        return video_dirs

    def output_json_for(self, video_dir: Path) -> Path:
        dataset = video_dir.parent.name
        video_id = video_dir.name
        return self.output_root / dataset / f"{video_id}.json"

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def _recognize(self, img, boxes: list[list[float]], paddle_texts: list[str]) -> list[str]:
        """Return the final texts list for one image's detected boxes,
        according to self.recognizer."""
        if self.recognizer != "vietocr":
            return paddle_texts

        from src.ocr_extraction.models.vietocr_engine import crop_box, recognize_crops

        crops = []
        keep_idx = []
        for i, box in enumerate(boxes):
            crop = crop_box(img, box)
            if crop is not None:
                crops.append(crop)
                keep_idx.append(i)

        if not crops:
            return ["" for _ in boxes]

        recognized = recognize_crops(self.vietocr_predictor, crops, logger=self.logger)

        texts = ["" for _ in boxes]
        for idx, text in zip(keep_idx, recognized):
            texts[idx] = text
        return texts

    def process_video_dir(self, video_dir: Path) -> int:
        output_file = self.output_json_for(video_dir)

        if output_file.exists() and self.skip_existing:
            self.logger.info("Skip existing: %s", output_file)
            return 0

        loader, total_images = build_loader(
            image_dir=video_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        if total_images == 0:
            self.logger.warning("No keyframes found: %s", video_dir)
            return 0

        data: dict[str, list] = {}

        for batch in tqdm(loader, desc=f"OCR {video_dir.name}", leave=False):
            if not batch:
                continue

            paths, imgs = zip(*batch)

            results = self.model.predict(list(imgs))
            
            if results is None:
                continue


            for path, img, res in zip(paths, imgs, results):
                keyframe_id = Path(path).stem
                boxes: list[list[float]] = []
                paddle_texts: list[str] = []

                if res:
                    raw_boxes = res.get("rec_boxes", [])
                    boxes = [[float(v) for v in box] for box in raw_boxes]
                    paddle_texts = list(res.get("rec_texts", []))

                    if self.visualized and self.visualize_dir:
                        vis_path = Path(self.visualize_dir) / video_dir.parent.name / video_dir.name / f"{keyframe_id}.jpg"
                        vis_path.parent.mkdir(parents=True, exist_ok=True)
                        res.save_to_img(str(vis_path))

                if self.engine == "paddle_vietocr":
                    texts = self._recognize(img, boxes, paddle_texts)

                elif self.engine == "qwen": 
                    if len(boxes) > 0: 
                        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        texts = extract_text_lines(
                            self.qwen_model,
                            self.qwen_processor,
                            pil_img,
                            max_new_tokens=int(self.qwen_cfg.get("max_new_tokens", 512)),
                            prompt=self.qwen_cfg.get("prompt"),
                            logger=self.logger,
                        )
                        boxes = [[0.0, 0.0, 0.0, 0.0] for _ in texts]
                else:
                    self.logger.info("Skip frame id %s", keyframe_id)
                    continue

                data[keyframe_id] = [boxes, texts]

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.logger.info("Saved %d keyframes -> %s", len(data), output_file)
        return len(data)

    def run(self) -> None:
        video_dirs = self.scan_video_dirs()
        self.logger.info("Found %d video keyframe folders", len(video_dirs))

        total_saved = 0
        for video_dir in tqdm(video_dirs, desc="Videos"):
            try:
                total_saved += self.process_video_dir(video_dir)
            except Exception as e:
                self.logger.error("Failed %s: %s", video_dir, e)

        self.logger.info("Total keyframes OCR'd: %d", total_saved)
