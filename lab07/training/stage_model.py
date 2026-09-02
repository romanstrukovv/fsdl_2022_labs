"""Stages a model for use in production.

If based on a checkpoint, the model is converted to torchscript, saved locally,
and uploaded to W&B.

If based on a model that is already converted and uploaded, the model file is downloaded locally.

For details on how the W&B artifacts backing the checkpoints and models are handled,
see the documentation for stage_model.find_artifact.
"""

import argparse
from pathlib import Path
import tempfile
import urllib.request

import torch
import wandb

from text_recognizer.lit_models import TransformerLitModel
from training.util import setup_data_and_model_from_args

# These names are all set by the pl.loggers.WandbLogger
MODEL_CHECKPOINT_TYPE = "model"
BEST_CHECKPOINT_ALIAS = "best"
MODEL_CHECKPOINT_PATH = "model.ckpt"
LOG_DIR = Path("training") / "logs"

STAGED_MODEL_TYPE = "prod-ready"
STAGED_MODEL_FILENAME = "model.pt"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LITMODEL_CLASS = TransformerLitModel

# Resilient API initialization
try:
    api = wandb.Api()
    DEFAULT_ENTITY = api.default_entity
except Exception:
    api = None
    DEFAULT_ENTITY = None

DEFAULT_FROM_PROJECT = "fsdl-text-recognizer-2022-training"
DEFAULT_TO_PROJECT = "fsdl-text-recognizer-2022-training"
DEFAULT_STAGED_MODEL_NAME = "paragraph-text-recognizer"

PROD_STAGING_ROOT = PROJECT_ROOT / "text_recognizer" / "artifacts"
PUBLIC_S3_BUCKET = "fsdl-public-assets"
PUBLIC_S3_KEY = "models/paragraph-text-recognizer/model.pt"


def main(args):
    prod_staging_directory = PROD_STAGING_ROOT / args.staged_model_name
    prod_staging_directory.mkdir(exist_ok=True, parents=True)
    target_model_file = prod_staging_directory / STAGED_MODEL_FILENAME

    # 1. Fetching an existing compiled model
    if args.fetch:
        if target_model_file.exists() and not args.force:
            print(f"Model already present at {target_model_file}. Use --force to re-download.")
            return

        # Direct URL download bypass (e.g. from W&B Files tab or direct CDN)
        if args.url:
            print(f"Downloading model directly from URL:\n{args.url}")
            download_from_url(args.url, target_model_file)
            print(f"Successfully saved to {target_model_file}")
            return

        # Explicit S3 unsigned bucket fetch
        if args.s3:
            print(f"Fetching from public S3 bucket: s3://{PUBLIC_S3_BUCKET}/{PUBLIC_S3_KEY}")
            if download_from_s3(PUBLIC_S3_BUCKET, PUBLIC_S3_KEY, target_model_file):
                print(f"Successfully saved to {target_model_file}")
                return
            print("S3 download failed. Trying W&B API...")

        # W&B Artifact download
        entity = _get_entity_from(args)
        version = args.staged_model_version
        staged_model = f"{entity}/{args.from_project}/{args.staged_model_name}:{version}"
        print(f"Fetching artifact from W&B: {staged_model}")

        try:
            artifact = download_artifact(staged_model, prod_staging_directory)
            print_info(artifact)
            return
        except Exception as e:
            print(f"\n[Warning] W&B API access failed ({e}).")
            print("Attempting automatic fallback to unsigned public S3 storage...")
            if download_from_s3(PUBLIC_S3_BUCKET, PUBLIC_S3_KEY, target_model_file):
                print(f"Recovered successfully! Model saved to {target_model_file}")
                return

            print("\n[Error] Could not fetch model automatically.")
            print("Fallback Options:")
            print(" 1. Copy the download link from the W&B UI 'Files' tab and run:")
            print('    --fetch --url="<COPIED_URL>"')
            print(" 2. Use the public S3 flag:")
            print("    --fetch --s3")
            raise

    # 2. Compiling and staging from an active training run checkpoint
    with wandb.init(job_type="stage", project=args.to_project, dir=LOG_DIR):
        entity = _get_entity_from(args)
        ckpt_at, ckpt_api = find_artifact(
            entity, args.from_project, type=MODEL_CHECKPOINT_TYPE, alias=args.ckpt_alias, run=args.run
        )

        logging_run = get_logging_run(ckpt_api)
        print_info(ckpt_api, logging_run)
        metadata = get_checkpoint_metadata(logging_run, ckpt_api)

        staged_at = wandb.Artifact(args.staged_model_name, type=STAGED_MODEL_TYPE, metadata=metadata)
        with tempfile.TemporaryDirectory() as tmp_dir:
            download_artifact(ckpt_at, tmp_dir)
            model = load_model_from_checkpoint(metadata, directory=tmp_dir)
            save_model_to_torchscript(model, directory=prod_staging_directory)

        upload_staged_model(staged_at, from_directory=prod_staging_directory)


def download_from_url(url: str, target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(target_path))


def download_from_s3(bucket_name: str, key: str, target_path: Path) -> bool:
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket_name, key, str(target_path))
        return True
    except Exception as e:
        print(f"S3 fetch failed: {e}")
        return False


def find_artifact(entity: str, project: str, type: str, alias: str, run=None):
    if run is not None:
        path = _find_artifact_run(entity, project, type=type, run=run, alias=alias)
    else:
        path = _find_artifact_project(entity, project, type=type, alias=alias)
    return path, api.artifact(path)


def get_logging_run(artifact):
    return artifact.logged_by()


def print_info(artifact, run=None):
    if run is None:
        run = get_logging_run(artifact)

    full_artifact_name = f"{artifact.entity}/{artifact.project}/{artifact.name}"
    print(f"Using artifact {full_artifact_name}")
    artifact_url_prefix = f"https://wandb.ai/{artifact.entity}/{artifact.project}/artifacts/{artifact.type}"
    artifact_url_suffix = f"{artifact.name.replace(':', '/')}"
    print(f"View at URL: {artifact_url_prefix}/{artifact_url_suffix}")

    if run:
        print(f"Logged by {run.name} -- {run.project}/{run.entity}/{run.id}")
        print(f"View at URL: {run.url}")


def get_checkpoint_metadata(run, checkpoint):
    config = run.config
    out = {"config": config}
    try:
        ckpt_filename = checkpoint.metadata["original_filename"]
        out["original_filename"] = ckpt_filename
        metric_key = checkpoint.metadata["ModelCheckpoint"]["monitor"]
        metric_score = checkpoint.metadata["score"]
        out[metric_key] = metric_score
    except KeyError:
        pass
    return out


def download_artifact(artifact_path, target_directory):
    if wandb.run is not None:
        artifact = wandb.use_artifact(artifact_path)
    else:
        if api is None:
            raise RuntimeError("W&B API is unauthenticated. Run `wandb login` first.")
        artifact = api.artifact(artifact_path)
    artifact.download(root=str(target_directory))
    return artifact


def load_model_from_checkpoint(ckpt_metadata, directory):
    config = ckpt_metadata["config"]
    args = argparse.Namespace(**config)

    _, model = setup_data_and_model_from_args(args)

    pth = Path(directory) / MODEL_CHECKPOINT_PATH
    lit_model = LITMODEL_CLASS.load_from_checkpoint(checkpoint_path=pth, args=args, model=model, strict=False)
    lit_model.eval()
    return lit_model


def save_model_to_torchscript(model, directory):
    scripted_model = model.to_torchscript(method="script", file_path=None)
    path = Path(directory) / STAGED_MODEL_FILENAME
    torch.jit.save(scripted_model, str(path))


def upload_staged_model(staged_at, from_directory):
    staged_at.add_file(str(Path(from_directory) / STAGED_MODEL_FILENAME))
    wandb.log_artifact(staged_at)


def _find_artifact_run(entity, project, type, run, alias):
    run_name = f"{entity}/{project}/{run}"
    api_run = api.run(run_name)
    artifacts = api_run.logged_artifacts()
    match = [art for art in artifacts if alias in art.aliases and art.type == type]
    if not match:
        raise ValueError(f"No artifact with alias {alias} found at {run_name} of type {type}")
    if len(match) > 1:
        raise ValueError(f"Multiple artifacts ({len(match)}) with alias {alias} found at {run_name} of type {type}")
    return f"{entity}/{project}/{match[0].name}"


def _find_artifact_project(entity, project, type, alias):
    project_name = f"{entity}/{project}"
    api_project = api.project(project, entity=entity)
    api_artifact_types = api_project.artifacts_types()
    for artifact_type in api_artifact_types:
        if artifact_type.name != type:
            continue
        collections = artifact_type.collections()
        for collection in collections:
            versions = collection.versions()
            for version in versions:
                if alias in version.aliases:
                    return f"{project_name}/{version.name}"
        raise ValueError(f"Artifact with alias {alias} not found in type {type} in {project_name}")
    raise ValueError(f"Artifact type {type} not found. {project_name} could be private or not exist.")


def _get_entity_from(args):
    entity = args.entity
    if entity is None or entity == "DEFAULT":
        if DEFAULT_ENTITY:
            return DEFAULT_ENTITY
        raise RuntimeError("No entity provided and no default entity found. Set --entity=<entity>.")
    return entity


def _setup_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help=f"Download the staged model to {PROD_STAGING_ROOT}/<STAGED_MODEL_NAME>.",
    )
    parser.add_argument(
        "--staged_model_version",
        type=str,
        default="latest",
        help="Version alias or hash of the artifact to fetch (e.g. 'latest' or '3e07efa34aec61999c5a').",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Direct URL to download model.pt, bypassing W&B API permissions.",
    )
    parser.add_argument(
        "--s3",
        action="store_true",
        help="Download model.pt from the public fsdl-public-assets S3 bucket.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model.pt is already present.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help=f"Entity to download the checkpoint/model from. Default: {DEFAULT_ENTITY}.",
    )
    parser.add_argument(
        "--from_project",
        type=str,
        default=DEFAULT_FROM_PROJECT,
        help=f"Project from which to download. Default: {DEFAULT_FROM_PROJECT}.",
    )
    parser.add_argument(
        "--to_project",
        type=str,
        default=DEFAULT_TO_PROJECT,
        help=f"Project to upload the compiled model to. Default: {DEFAULT_TO_PROJECT}.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help=f"Run name containing the {MODEL_CHECKPOINT_TYPE} artifact.",
    )
    parser.add_argument(
        "--ckpt_alias",
        type=str,
        default=BEST_CHECKPOINT_ALIAS,
        help=f"Alias identifying the checkpoint to stage. Default: {BEST_CHECKPOINT_ALIAS!r}.",
    )
    parser.add_argument(
        "--staged_model_name",
        type=str,
        default=DEFAULT_STAGED_MODEL_NAME,
        help=f"Name of the staged model artifact. Default: {DEFAULT_STAGED_MODEL_NAME!r}.",
    )
    return parser


if __name__ == "__main__":
    parser = _setup_parser()
    args = parser.parse_args()
    main(args)
