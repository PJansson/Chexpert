import configargparse
from pathlib import Path


def create_configargparser():
    p = configargparse.ArgParser()
    p.add("-c", "--config", required=True, is_config_file=True, help="Config file path")

    # Dataset
    p.add("--data_dir", type=Path)
    p.add("--dataset", type=Path)
    p.add("--image_size", type=int)
    p.add("--classes", nargs="*")

    # Model
    p.add("--model_name", type=str)
    p.add("--multi_view_model", action="store_true")
    p.add("--max_before_pool", action="store_true")

    p.add("--ema_decay", type=float)

    # Optimizer
    p.add("--learning_rate", type=float)
    p.add("--weight_decay", type=float)

    # Scheduler
    p.add("--scheduler_factor", type=float)
    p.add("--scheduler_patience", type=int)

    # Training
    p.add("--epochs", type=int)
    p.add("--validations_per_epoch", type=int)

    p.add("--batch_size", type=int)
    p.add("--accumulation_steps", type=int)

    p.add("--num_workers", type=int)
    p.add("--use_amp", action="store_true")
    p.add("--pin_memory", action="store_true")

    p.add("--model_save_path", type=Path)

    # AUROC
    p.add("--auroc_class_scores", action="store_true")
    p.add("--per_study_auroc", action="store_true")

    # Data Augmentation
    p.add("--coarse_dropout_max", type=int)
    p.add("--coarse_dropout_min", type=int)
    p.add("--coarse_dropout_h", type=int)
    p.add("--coarse_dropout_w", type=int)

    # Model file, for test.py
    p.add("--model_path", type=str)

    return p
