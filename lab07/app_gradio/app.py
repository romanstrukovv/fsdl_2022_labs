"""Provide an image of handwritten text and get back out a string!"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Callable

import gradio as gr
from PIL import ImageStat
from PIL.Image import Image
import requests

from text_recognizer.paragraph_text_recognizer import ParagraphTextRecognizer
import text_recognizer.util as util

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # do not use GPU

logging.basicConfig(level=logging.INFO)

APP_DIR = Path(__file__).resolve().parent  # what is the directory for this application?
FAVICON = APP_DIR / "1f95e.png"  # path to a small image for display in browser tab and social media
README = APP_DIR / "README.md"  # path to an app readme file in HTML/markdown

DEFAULT_PORT = 11700


def main(args):
    predictor = PredictorBackend(url=args.model_url)
    frontend = make_frontend(
        predictor.run,
    )
    frontend.launch(
        server_name="0.0.0.0",  # make server accessible, binding all interfaces  # noqa: S104
        server_port=args.port,  # set a port to bind to, failing if unavailable
        share=True,  # should we create a (temporary) public link on https://gradio.app?
        favicon_path=str(FAVICON) if FAVICON.exists() else None,
    )


def make_frontend(
    fn: Callable[[Image], str],
):
    """Creates a gradio.Interface frontend for an image to text function."""
    examples_dir = Path("text_recognizer") / "tests" / "support" / "paragraphs"
    example_fnames = [elem for elem in os.listdir(examples_dir) if elem.endswith(".png")]
    example_paths = [examples_dir / fname for fname in example_fnames]

    examples = [[str(path)] for path in example_paths]

    # Handle flagging parameter compatibility across Gradio versions
    flagging_kwargs = {}
    if "flagging_mode" in gr.Interface.__init__.__code__.co_varnames:
        flagging_kwargs["flagging_mode"] = "never"
    else:
        flagging_kwargs["allow_flagging"] = "never"

    readme = _load_readme(with_logging=False)

    # Build interface without deprecated thumbnail argument
    frontend = gr.Interface(
        fn=fn,
        outputs=gr.Textbox(),
        inputs=gr.Image(type="pil", label="Handwritten Text"),
        title="📝 Text Recognizer",
        description=__doc__,
        article=readme,
        examples=examples,
        cache_examples=False,
        **flagging_kwargs,
    )

    return frontend


class PredictorBackend:
    """Interface to a backend that serves predictions.

    To communicate with a backend accessible via a URL, provide the url kwarg.

    Otherwise, runs a predictor locally.
    """

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
        """Send an image to an endpoint that accepts JSON and return the predicted text.

        The endpoint should expect a base64 representation of the image, encoded as a string,
        under the key "image". It should return the predicted text under the key "pred".
        """
        encoded_image = util.encode_b64_image(image)

        headers = {"Content-type": "application/json"}
        payload = json.dumps({"image": "data:image/png;base64," + encoded_image})

        response = requests.post(self.url, data=payload, headers=headers)
        pred = response.json()["pred"]

        return pred

    def _log_inference(self, pred, metrics):
        for key, value in metrics.items():
            logging.info(f"METRIC {key} {value}")
        logging.info(f"PRED >begin\n{pred}\nPRED >end")


def _load_readme(with_logging=False):
    if not README.exists():
        return ""
    with open(README) as f:
        lines = f.readlines()
        if not with_logging:
            split_token = "<!-- logging content below -->"
            matching = [idx for idx, line in enumerate(lines) if split_token in line]
            if matching:
                lines = lines[: matching[0]]

        readme = "".join(lines)
    return readme


def _make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_url",
        default=None,
        type=str,
        help="Identifies a URL to which to send image data. Default is None (runs locally).",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        type=int,
        help=f"Port on which to expose this server. Default is {DEFAULT_PORT}.",
    )

    return parser


if __name__ == "__main__":
    parser = _make_parser()
    args = parser.parse_args()
    main(args)
