"""Flagging and telemetry logger for Gradio interfaces using Evidently AI and local storage."""

import csv
from datetime import datetime
import io
import math
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import gradio as gr
from gradio.components import Component
import numpy as np
from PIL import Image

from app_gradio import s3_util
from text_recognizer.util import read_b64_string


def compute_entropy(text: str) -> float:
    """Calculates character-level Shannon entropy (replaces Gantry text_stats.basics.entropy)."""
    if not text or not isinstance(text, str):
        return 0.0
    prob = [float(text.count(c)) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in prob)


def compute_image_mean(image: Image.Image) -> float:
    """Calculates mean grayscale intensity (replaces Gantry image.greyscale_image_mean)."""
    try:
        return float(np.mean(image.convert("L")))
    except Exception:
        return 128.0


class GantryImageToTextLogger(gr.FlaggingCallback if hasattr(gr, "FlaggingCallback") else object):
    """FlaggingCallback that stores failure cases and calculates telemetry features for Evidently AI.

    Retains the original GantryImageToTextLogger name and constructor parameters so
    app_gradio/app.py continues to function without modifications.
    """

    def __init__(
        self,
        application: str = "fsdl-text-recognizer",
        version: Union[int, str, None] = None,
        api_key: Optional[str] = None,
    ):
        self.application = application
        self.version = version
        self.api_key = api_key or get_api_key()
        self._counter = 0

    def setup(self, components: List[Component], flagging_dir: str = "flagged"):
        """Initializes storage directories, CSV log files, and optional S3 connection."""
        self._counter = 0
        self.flag_dir = Path(flagging_dir)
        self.images_dir = self.flag_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.flag_dir / "log.csv"

        # Optional S3 initialization with safe fallback to local storage
        self.bucket = None
        try:
            self.bucket = s3_util.get_or_create_bucket(flagging_dir)
            s3_util.enable_bucket_versioning(self.bucket)
            s3_util.add_access_policy(self.bucket)
        except Exception:
            self.bucket = None

        self.image_component_idx, self.text_component_idx = self._find_image_and_text_components(components)

    def flag(
        self,
        flag_data: List[Any],
        flag_option: Optional[str] = None,
        flag_index: Optional[int] = None,
        username: Optional[str] = None,
    ) -> int:
        """Processes flagged user feedback, saves the image, and writes projections for Evidently."""
        raw_image = flag_data[self.image_component_idx]
        output_text = str(flag_data[self.text_component_idx]) if flag_data[self.text_component_idx] else ""

        pil_image, image_bytes, filetype = self._parse_image(raw_image)

        # 1. Save image to local disk
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        local_filename = f"flagged_{timestamp_str}.{filetype}"
        local_image_path = self.images_dir / local_filename
        pil_image.save(local_image_path)

        # 2. Upload to S3 if bucket is configured; otherwise fallback to local path
        image_uri = str(local_image_path)
        if self.bucket is not None:
            try:
                image_uri = self._to_s3(image_bytes, filetype=filetype)
            except Exception:
                pass

        # 3. Calculate behavioral projection metrics
        brightness = compute_image_mean(pil_image)
        entropy = compute_entropy(output_text)

        # 4. Write record containing both standard names and legacy Gantry column keys
        self._log_record(
            image_uri=image_uri,
            output_text=output_text,
            flag_option=flag_option or "flagged",
            username=username,
            brightness=brightness,
            entropy=entropy,
        )

        self._counter += 1
        return self._counter

    def _parse_image(self, raw_image: Any) -> Tuple[Image.Image, bytes, str]:
        """Normalizes Gradio image payload (base64 string, PIL Image, or numpy array) into PIL and bytes."""
        if isinstance(raw_image, str) and raw_image.startswith("data:"):
            data_type, image_buffer = read_b64_string(raw_image, return_data_type=True)
            image_bytes = image_buffer.read()
            pil_image = Image.open(io.BytesIO(image_bytes))
            return pil_image, image_bytes, data_type or "png"

        if isinstance(raw_image, Image.Image):
            buf = io.BytesIO()
            raw_image.save(buf, format="PNG")
            return raw_image, buf.getvalue(), "png"

        if isinstance(raw_image, np.ndarray):
            pil_image = Image.fromarray(raw_image)
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            return pil_image, buf.getvalue(), "png"

        # Fallback empty canvas if unrecognized
        dummy = Image.new("L", (64, 64), color=128)
        buf = io.BytesIO()
        dummy.save(buf, format="PNG")
        return dummy, buf.getvalue(), "png"

    def _log_record(
        self,
        image_uri: str,
        output_text: str,
        flag_option: str,
        username: Optional[str],
        brightness: float,
        entropy: float,
    ):
        """Appends tabular data to log.csv for pandas and Evidently drift reporting."""
        file_exists = self.csv_path.exists()
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "timestamp",
                        "inputs.image",
                        "outputs.output_text",
                        "feedback.flag",
                        "feedback.user",
                        "image.greyscale_image_mean(inputs.image)",
                        "text_stats.basics.entropy(outputs.output_text)",
                        "image_mean_brightness",
                        "text_entropy",
                    ]
                )
            writer.writerow(
                [
                    datetime.utcnow().isoformat(),
                    image_uri,
                    output_text,
                    flag_option,
                    username or "anonymous",
                    brightness,
                    entropy,
                    brightness,
                    entropy,
                ]
            )

    def _to_s3(self, image_bytes: bytes, key: Optional[str] = None, filetype: Optional[str] = None) -> str:
        """Stores binary image data into Amazon S3."""
        if key is None:
            key = s3_util.make_key(image_bytes, filetype=filetype)

        s3_uri = s3_util.get_uri_of(self.bucket, key)
        with open(s3_uri, "wb") as s3_object:
            s3_object.write(image_bytes)

        return s3_uri

    def _find_image_and_text_components(self, components: List[Component]) -> Tuple[int, int]:
        """Finds indices for image and textbox components across different Gradio versions."""
        image_component_idx, text_component_idx = None, None

        for idx, component in enumerate(components):
            comp_type = type(component).__name__.lower()
            if "image" in comp_type:
                image_component_idx = idx
            elif "text" in comp_type:
                text_component_idx = idx

        if image_component_idx is None:
            image_component_idx = 0
        if text_component_idx is None:
            text_component_idx = 1

        return image_component_idx, text_component_idx


# Alias for explicit imports
EvidentlyImageToTextLogger = GantryImageToTextLogger


def get_api_key() -> Optional[str]:
    """Stubbed key provider avoiding connection errors to gantry.io."""
    return os.environ.get("GANTRY_API_KEY", "local-offline-key")
