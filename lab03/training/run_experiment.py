"""Experiment-running framework."""

import argparse
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


def _parse_fraction_or_int(val):
    """Safely parse float vs int. PyTorch Lightning treats 1.0 (float) as 100% and 1 (int) as 1 batch."""
    if isinstance(val, str):
        if "." in val:
            return float(val)
        return int(val)
    return val


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Replaces pl.Trainer.add_argparse_args to support PyTorch Lightning 2.0+
    trainer_group = parser.add_argument_group("Trainer Args")
    trainer_group.add_argument("--max_epochs", type=int, default=1)
    trainer_group.add_argument("--accelerator", type=str, default="auto")
    trainer_group.add_argument("--devices", type=str, default="auto")
    trainer_group.add_argument("--gpus", type=str, default=None, help="Legacy arg: mapped to accelerator='gpu'")
    trainer_group.add_argument("--precision", type=str, default="32")
    trainer_group.add_argument("--fast_dev_run", action="store_true", default=False)
    trainer_group.add_argument("--overfit_batches", type=_parse_fraction_or_int, default=0.0)
    trainer_group.add_argument("--check_val_every_n_epoch", type=int, default=1)
    trainer_group.add_argument("--limit_train_batches", type=_parse_fraction_or_int, default=1.0)
    trainer_group.add_argument("--limit_val_batches", type=_parse_fraction_or_int, default=1.0)
    trainer_group.add_argument("--limit_test_batches", type=_parse_fraction_or_int, default=1.0)
    trainer_group.add_argument("--accumulate_grad_batches", type=int, default=1)
    trainer_group.add_argument("--gradient_clip_val", type=float, default=None)
    trainer_group.add_argument("--log_every_n_steps", type=int, default=50)
    trainer_group.add_argument("--auto_lr_find", action="store_true", default=False)
    trainer_group.add_argument("--strategy", type=str, default="auto")

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

    if args.loss == "transformer":
        lit_model_class = lit_models.TransformerLitModel

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
    if goldstar_metric == "validation/cer":
        filename_format += "-validation.cer={validation/cer:.3f}"

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

    # Map the parsed args to trainer kwargs manually
    trainer_kwargs = {
        "max_epochs": args.max_epochs,
        "accelerator": args.accelerator,
        "precision": args.precision,
        "fast_dev_run": args.fast_dev_run,
        "overfit_batches": args.overfit_batches,
        "check_val_every_n_epoch": args.check_val_every_n_epoch,
        "limit_train_batches": args.limit_train_batches,
        "limit_val_batches": args.limit_val_batches,
        "limit_test_batches": args.limit_test_batches,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "log_every_n_steps": args.log_every_n_steps,
        "strategy": args.strategy,
        "callbacks": callbacks,
        "logger": logger,
    }

    if args.gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = args.gradient_clip_val

    # Backward compatibility with PyTorch Lightning 1.x --gpus format (e.g., '0,')
    if args.gpus is not None:
        trainer_kwargs["accelerator"] = "gpu"
        if isinstance(args.gpus, str) and args.gpus.endswith(","):
            trainer_kwargs["devices"] = [int(x) for x in args.gpus.split(",") if x]
        else:
            try:
                trainer_kwargs["devices"] = int(args.gpus)
            except ValueError:
                trainer_kwargs["devices"] = args.gpus
    else:
        try:
            trainer_kwargs["devices"] = int(args.devices)
        except ValueError:
            trainer_kwargs["devices"] = args.devices

    # Instantiate Trainer with mapped kwargs (replacing from_argparse_args)
    trainer = pl.Trainer(**trainer_kwargs)

    # Handle LR finder changes in PL 2.0+
    if args.auto_lr_find:
        try:
            from pytorch_lightning.tuner import Tuner

            tuner = Tuner(trainer)
            tuner.lr_find(lit_model, datamodule=data)
        except ImportError:
            # Fallback in case a legacy version somehow hits this block
            trainer.tune(lit_model, datamodule=data)

    trainer.fit(lit_model, datamodule=data)

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        rank_zero_info(f"Best model saved at: {best_model_path}")
        # Explicitly pass model to test() which is safer in PL 2.0+
        trainer.test(lit_model, datamodule=data, ckpt_path=best_model_path)
    else:
        trainer.test(lit_model, datamodule=data)


if __name__ == "__main__":
    main()
