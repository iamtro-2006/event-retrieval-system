from __future__ import annotations

import logging

from paddleocr import PaddleOCR


def load_ocr_model(
    ocr_version: str,
    text_detection_model_name: str,
    text_recognition_model_name: str,
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_textline_orientation: bool = False,
    device: str = "cpu",
    logger: logging.Logger | None = None,
) -> PaddleOCR:
    """Initialize a PaddleOCR (PP-OCR) pipeline instance.

    Mirrors the model construction previously done in PaddleOCR/main.py,
    but reads all parameters from configs/ocr_extraction.yaml instead of config.py.

    NOTE: when extraction.recognizer == "vietocr" (see extract_ocr.py), this
    model is used ONLY as a text-region detector -- res["rec_boxes"] is kept,
    res["rec_texts"] (PaddleOCR's own recognizer, which does not support
    Vietnamese) is discarded and replaced by VietOCR output.
    """
    if logger:
        logger.info(
            "Loading PaddleOCR model: version=%s det=%s rec=%s device=%s",
            ocr_version,
            text_detection_model_name,
            text_recognition_model_name,
            device,
        )

    model = PaddleOCR(
        ocr_version=ocr_version,
        text_detection_model_name=text_detection_model_name,
        text_recognition_model_name=text_recognition_model_name,
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
        device=device,
    )

    return model
