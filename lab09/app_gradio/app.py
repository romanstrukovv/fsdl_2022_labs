"""Provide an image of handwritten text and get back out a string!"""

import argparse
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Callable
import warnings

import gradio as gr
from PIL import ImageStat
from PIL.Image import Image
import requests

from app_gradio.flagging import GantryImageToTextLogger, get_api_key
from app_gradio.s3_util import make_unique_bucket_name
from text_recognizer.paragraph_text_recognizer import ParagraphTextRecognizer
import text_recognizer.util as util

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # do not use GPU

logging.basicConfig(level=logging.INFO)
DEFAULT_APPLICATION_NAME = "fsdl-text-recognizer"

APP_DIR = Path(__file__).resolve().parent
FAVICON = APP_DIR / "1f95e.png"
README = APP_DIR / "README.md"

DEFAULT_PORT = 11700


def main(args):
    predictor = PredictorBackend(url=args.model_url)
    frontend = make_frontend(
        predictor.run,
        flagging=args.flagging,
        gantry=args.gantry,
        app_name=args.application,
    )
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": args.port,
        "share": True,
    }
    if FAVICON.exists():
        launch_kwargs["favicon_path"] = str(FAVICON)

    frontend.launch(**launch_kwargs)


def make_frontend(
    fn: Callable[[Image], str],
    flagging: bool = False,
    gantry: bool = False,
    app_name: str = "fsdl-text-recognizer",
):
    """Creates a gradio.Interface frontend for an image-to-text function."""
    # 1. Safely resolve example paths
    examples = []
    examples_dir = Path("text_recognizer") / "tests" / "support" / "paragraphs"
    if examples_dir.exists():
        example_fnames = [elem for elem in os.listdir(examples_dir) if elem.endswith(".png")]
        examples = [[str(examples_dir / fname)] for fname in sorted(example_fnames)]

    # 2. Configure flagging callbacks and output directory
    allow_flagging = "never"
    flagging_callback = None
    flagging_dir = "flagged"

    if flagging:
        allow_flagging = "manual"
        api_key = get_api_key()

        if gantry:
            flagging_callback = GantryImageToTextLogger(application=app_name, api_key=api_key)
            try:
                flagging_dir = make_unique_bucket_name(prefix=app_name, seed=api_key)
            except Exception:
                flagging_dir = "flagged"
        else:
            # Fallback to Gradio's standard CSV logger for local exploration in cells 10-14
            if hasattr(gr, "CSVLogger"):
                flagging_callback = gr.CSVLogger()
            elif hasattr(gr, "SimpleCSVLogger"):
                flagging_callback = gr.SimpleCSVLogger()
            else:
                flagging_callback = GantryImageToTextLogger(application=app_name, api_key=api_key)
            flagging_dir = "flagged"

    readme = _load_readme(with_logging=allow_flagging == "manual")

    # 3. Build Interface with cross-version Gradio compatibility
    interface_kwargs = {
        "fn": fn,
        "inputs": gr.Image(type="pil", label="Handwritten Text"),
        "outputs": gr.Textbox(label="Model Output"),
        "title": "📝 Text Recognizer",
        "description": __doc__,
        "article": readme if readme else None,
        "examples": examples if examples else None,
        "cache_examples": False,
        "flagging_options": ["incorrect", "offensive", "other"],
        "flagging_callback": flagging_callback,
        "flagging_dir": flagging_dir,
    }

    # Handle Gradio 3 (allow_flagging) vs. Gradio 4+ (flagging_mode)
    sig = inspect.signature(gr.Interface.__init__)
    if "flagging_mode" in sig.parameters:
        interface_kwargs["flagging_mode"] = allow_flagging
    else:
        interface_kwargs["allow_flagging"] = allow_flagging

    if FAVICON.exists() and "thumbnail" in sig.parameters:
        interface_kwargs["thumbnail"] = str(FAVICON)

    return gr.Interface(**interface_kwargs)


class PredictorBackend:
    """Interface to a backend that serves predictions locally or over HTTP."""

    def __init__(self, url=None):
        if url is not None:
            self.url = url
            self._predict = self._predict_from_endpoint
        else:
            model = ParagraphTextRecognizer()
            self._predict = model.predict

    def run(self, image):
        pred, metrics = self._predict_with_metrics(image)
        self._log_inference(pred, metrics)
        return pred

    def _predict_with_metrics(self, image):
        pred = self._predict(image)
        stats = ImageStat.Stat(image)
        metrics = {
            "image_mean_intensity": stats.mean,
            "image_median": stats.median,
            "image_extrema": stats.extrema,
            "image_area": image.size[0] * image.size[1],
            "pred_length": len(pred),
        }
        return pred, metrics

    def _predict_from_endpoint(self, image):
        """Encodes image as base64 JSON payload and retrieves prediction from remote URL."""
        encoded_image = util.encode_b64_image(image)
        headers = {"Content-Type": "application/json"}
        payload = json.dumps({"image": "data:image/png;base64," + encoded_image})

        response = requests.post(self.url, data=payload, headers=headers)
        if response.status_code != 200:
            logging.error(f"Endpoint returned HTTP {response.status_code}: {response.text}")
            response.raise_for_status()

        data = response.json()
        if "pred" in data:
            return data["pred"]
        if "prediction" in data:
            return data["prediction"]
        return str(data)

    def _log_inference(self, pred, metrics):
        for key, value in metrics.items():
            logging.info(f"METRIC {key} {value}")
        logging.info(f"PRED >begin\n{pred}\nPRED >end")


def _load_readme(with_logging=False):
    """Safely loads README markdown without raising ValueError on missing split markers."""
    if not README.exists():
        return ""

    with open(README, encoding="utf-8") as f:
        content = f.read()

    split_marker = "<!-- logging content below -->"
    if not with_logging and split_marker in content:
        content = content.split(split_marker)[0]

    return content


def _make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_url",
        default=None,
        type=str,
        help="Endpoint URL to POST base64 JSON images to. Default is None (runs local model).",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        type=int,
        help=f"Port on which to expose this server. Default is {DEFAULT_PORT}.",
    )
    parser.add_argument(
        "--flagging",
        action="store_true",
        help="Allow users to flag outputs in the interface.",
    )
    parser.add_argument(
        "--gantry",
        action="store_true",
        help="Log flagged user feedback using GantryImageToTextLogger.",
    )
    parser.add_argument(
        "--application",
        default=DEFAULT_APPLICATION_NAME,
        type=str,
        help=f"Telemetry application name. Default is {DEFAULT_APPLICATION_NAME}.",
    )
    return parser


if __name__ == "__main__":
    parser = _make_parser()
    args = parser.parse_args()
    main(args)
