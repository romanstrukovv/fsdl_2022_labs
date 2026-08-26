"""Experiment-running framework."""

import argparse
import inspect
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only
import torch

from text_recognizer import lit_models
from training.util import DATA_CLASS_MODULE, import_class, MODEL_CLASS_MODULE, setup_data_and_model_from_args

# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)


def _str_to_bool(v):
    """Helper to perfectly mimic PyTorch Lightning 1.x argparse boolean flags."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Replace deprecated pl.Trainer.add_argparse_args dynamically for PL 2.0+
    trainer_group = parser.add_argument_group("Trainer Args")
    sig = inspect.signature(pl.Trainer.__init__)
    for name, param in sig.parameters.items():
        if name in ["self", "logger", "callbacks", "profiler"]:
            continue

        default_val = param.default
        if default_val is not inspect.Parameter.empty and type(default_val) in (int, float, str, bool):
            if type(default_val) is bool:
                # Allows both `--fast_dev_run` and `--fast_dev_run False`
                trainer_group.add_argument(
                    f"--{name}", type=_str_to_bool, default=default_val, const=not default_val, nargs="?"
                )
            else:
                trainer_group.add_argument(f"--{name}", type=type(default_val), default=default_val)
        else:
            trainer_group.add_argument(f"--{name}", default=default_val)

    # Legacy FSDL flags missing in PL 2.0+
    trainer_group.add_argument("--gpus", type=str, default=None, help="Legacy FSDL argument")
    trainer_group.add_argument("--auto_lr_find", action="store_true", default=False)

    parser.set_defaults(max_epochs=1)

    # Basic arguments
    parser.add_argument(
        "--data_class",
        type=str,
        default="MNIST",
        help=f"String identifier for the data class, relative to {DATA_CLASS_MODULE}.",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="MLP",
        help=f"String identifier for the model class, relative to {MODEL_CLASS_MODULE}.",
    )
    parser.add_argument(
        "--load_checkpoint", type=str, default=None, help="If passed, loads a model from the provided path."
    )
    parser.add_argument(
        "--stop_early",
        type=int,
        default=0,
        help="If non-zero, applies early stopping, with the provided value as the 'patience' argument."
        + " Default is 0.",
    )

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = import_class(f"{DATA_CLASS_MODULE}.{temp_args.data_class}")
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


@rank_zero_only
def _ensure_logging_dir(experiment_dir):
    """Create the logging directory via the rank-zero process, if necessary."""
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    data, model = setup_data_and_model_from_args(args)

    lit_model_class = lit_models.BaseLitModel

    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(args.load_checkpoint, args=args, model=model)
    else:
        lit_model = lit_model_class(args=args, model=model)

    log_dir = Path("training") / "logs"
    _ensure_logging_dir(log_dir)
    logger = pl.loggers.TensorBoardLogger(log_dir)
    experiment_dir = logger.log_dir

    goldstar_metric = "validation/cer" if args.loss in ("transformer",) else "validation/loss"
    filename_format = "epoch={epoch:04d}-validation.loss={validation/loss:.3f}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=5,
        filename=filename_format,
        monitor=goldstar_metric,
        mode="min",
        auto_insert_metric_name=False,
        dirpath=experiment_dir,
        every_n_epochs=args.check_val_every_n_epoch,
    )

    summary_callback = pl.callbacks.ModelSummary(max_depth=2)

    callbacks = [summary_callback, checkpoint_callback]
    if args.stop_early:
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor="validation/loss", mode="min", patience=args.stop_early
        )
        callbacks.append(early_stopping_callback)

    # --- PL 2.0 Fix: Pluck valid Trainer arguments manually ---
    valid_kwargs = inspect.signature(pl.Trainer.__init__).parameters.keys()
    trainer_kwargs = {k: v for k, v in vars(args).items() if k in valid_kwargs}

    # Handle legacy FSDL --gpus string converting to accelerator + devices
    if hasattr(args, "gpus") and args.gpus is not None:
        trainer_kwargs["accelerator"] = "gpu"
        if isinstance(args.gpus, str) and "," in args.gpus:
            trainer_kwargs["devices"] = [int(x) for x in args.gpus.split(",") if x.strip()]
        else:
            trainer_kwargs["devices"] = int(args.gpus)

    if trainer_kwargs.get("max_epochs") is not None:
        trainer_kwargs["max_epochs"] = int(trainer_kwargs["max_epochs"])

    trainer = pl.Trainer(**trainer_kwargs, callbacks=callbacks, logger=logger)

    # --- PL 2.0 Fix: tune() migrated to Tuner object ---
    if hasattr(args, "auto_lr_find") and args.auto_lr_find:
        try:
            from pytorch_lightning.tuner import Tuner

            Tuner(trainer).lr_find(lit_model, datamodule=data)
        except ImportError:
            trainer.tune(lit_model, datamodule=data)  # Fallback

    trainer.fit(lit_model, datamodule=data)

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        rank_zero_info(f"Best model saved at: {best_model_path}")
        trainer.test(datamodule=data, ckpt_path=best_model_path)
    else:
        trainer.test(lit_model, datamodule=data)


if __name__ == "__main__":
    main()
