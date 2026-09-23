"""Finding and downloading the pretrained TotalSegmentator weights.

TotalSegmentator is a set of nnU-Net models. Which network we need depends on the
resolution in the config:
- 'fast' (3 mm): one network that predicts all 117 structures (task 297).
- 'full' (1.5 mm): five networks, each predicting one group of structures (tasks 291-295).
  We only need the one whose group contains our organ (e.g. task 291 'organs' for the spleen).

This module turns (resolution, organ) into the right task id, trainer name and the
organ's label number in that network's output, and downloads the weights into models/.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# TotalSegmentator calls its two resolutions 'fast' and 'default'. We call them 'fast' and 'full'.
RESOLUTION_TO_SUBMODE = {"fast": "fast", "full": "default"}


@dataclass
class ModelSpec:
    """Everything needed to locate and use one pretrained network."""
    task_id: int
    trainer: str
    organ_label: int        # the organ's channel index in the network output
    spacing_mm: float       # voxel size the network works at


def _use_models_dir(cfg: dict) -> None:
    """Make TotalSegmentator store and look for weights under paths.models_dir.

    TotalSegmentator reads the environment variable TOTALSEG_HOME_DIR (default: ~/.totalsegmentator).
    Setting it here keeps the weights inside the project folder, where the config says they are.
    """
    os.environ["TOTALSEG_HOME_DIR"] = str(cfg["paths"]["models_dir"] / "totalsegmentator")


def model_spec(cfg: dict) -> ModelSpec:
    """Work out which network to use for the configured task, resolution and organ.

    Input: the loaded config (model.task, model.resolution, model.organ).
    Output: a ModelSpec.
    """
    from totalsegmentator.map_tasks_config import TASK_CONFIGS
    from totalsegmentator.map_to_binary import class_map, class_map_5_parts, map_taskid_to_partname_ct

    task, organ = cfg["model"]["task"], cfg["model"]["organ"]
    sub = TASK_CONFIGS[task]["sub_modes"][RESOLUTION_TO_SUBMODE[cfg["model"]["resolution"]]]
    task_ids = sub["task_id"] if isinstance(sub["task_id"], list) else [sub["task_id"]]

    for task_id in task_ids:
        # A single network uses the full class map; a 'part' network uses its own smaller map.
        labels = class_map[task] if len(task_ids) == 1 else class_map_5_parts[map_taskid_to_partname_ct[task_id]]
        name_to_label = {name: label for label, name in labels.items()}
        if organ in name_to_label:
            return ModelSpec(task_id, sub["trainer"], name_to_label[organ], sub["resample"])
    raise ValueError(f"Organ '{organ}' is not predicted by task '{task}'.")


def model_folder(cfg: dict, spec: ModelSpec) -> Path:
    """Return the nnU-Net results folder that holds the network's plans and fold_* checkpoints."""
    _use_models_dir(cfg)
    from totalsegmentator.config import get_weights_dir

    matches = sorted(get_weights_dir().glob(f"Dataset{spec.task_id:03d}_*"))
    if not matches:
        raise FileNotFoundError(f"Weights for task {spec.task_id} not found. Run scripts/download_model.py first.")
    return matches[-1] / f"{spec.trainer}__nnUNetPlans__3d_fullres"


def available_folds(folder: Path) -> list[int]:
    """List the folds that actually have a trained checkpoint in the weights folder.

    nnU-Net normally trains 5 folds (5-fold cross-validation). Disagreement between folds
    is only possible as an uncertainty measure if more than one fold is released.
    """
    return sorted(int(p.parent.name.split("_")[1]) for p in folder.glob("fold_*/checkpoint_final.pth"))


def download_weights(cfg: dict) -> Path:
    """Download the weights for the configured network (skipped if already present).

    Output: the model folder.
    """
    _use_models_dir(cfg)
    from totalsegmentator.libs import download_pretrained_weights

    spec = model_spec(cfg)
    download_pretrained_weights(spec.task_id)
    return model_folder(cfg, spec)
