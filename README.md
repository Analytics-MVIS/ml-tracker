# mltracker

mltracker is an internal SDK that standardizes experiment tracking for all teams.

The goal is simple: teams import mltracker and never call MLflow directly. This keeps configuration, metadata, artifact layout, and run status handling consistent across projects.

## Quick start (recommended path for most teams)

### 1. Install in your training repo

```bash
pip install git+https://github.com/Analytics-MVIS/ml-tracker.git
```

### 2. Set tracking URI once

```bash
export MLTRACK_TRACKING_URI="https://your-mlflow-server:5000"
```

### 3. Wrap your train loop with tracker

```python
from mltracker.configs import ClassificationConfig
from mltracker.trackers import ClassificationTracker

cfg = ClassificationConfig(
	project="axle-bolt-classification",
	run_name="resnet101-v1",
	model_name="resnet101",
	dataset_name="dataset4",
	epochs=50,
	learning_rate=0.001,
	batch_size=16,
	num_classes=2,
)

with ClassificationTracker(
	experiment_name="axle-bolt-classification",
	config=cfg,
) as tracker:
	# inside your training loop
	tracker.log_metric("train_loss", 0.42, step=1)
	tracker.log_metric("val_acc", 91.8, step=1)

	# after training completes
	tracker.log_model(
		model_path="outputs/best.pt",
		model_name="resnet101-v1",
		register_name="axle-bolt-classifier",
	)
```

If your training raises an exception, mltracker marks the run as `FAILED` automatically.

## Security and privacy first

Before using mltracker in any project, follow these rules:

- Never commit private data, credentials, tokens, customer identifiers, or internal host details to this repository.
- Never log sensitive values in MLflow params, tags, artifact filenames, or model names.
- Use placeholders in shared examples and docs, for example `https://your-mlflow-server:5000`.
- Keep secrets in environment variables or a secret manager, not in code.
- Review artifacts before logging (images, reports, CSV exports) to ensure they do not contain sensitive information.

## What you get

- Enforced schemas for YOLO and classification runs
- Automatic logging of config fields as MLflow params
- Automatic system tags (git, host, python, gpu/cuda when available)
- Consistent artifact helpers for model files and evaluation outputs
- Context manager and decorator patterns for reliable run lifecycle handling
- Optional model registration in MLflow Model Registry

## Installation modes

Choose one of these modes depending on whether you are using the SDK or contributing to it.

### 1. Use mltracker in your training project (recommended)

Install from GitHub default branch (currently `main`):

```bash
pip install git+https://github.com/Analytics-MVIS/ml-tracker.git
```

Equivalent explicit ref:

```bash
pip install git+https://github.com/Analytics-MVIS/ml-tracker.git@main
```

Using uv:

```bash
uv add git+https://github.com/Analytics-MVIS/ml-tracker.git
```

For strict reproducibility, pin to a commit SHA instead of a tag:

```bash
pip install git+https://github.com/Analytics-MVIS/ml-tracker.git@<commit-sha>
```

Use this path for all model teams that only need to log experiments.

### 2. Contribute to mltracker SDK (maintainers)

Clone and install in editable mode:

```bash
git clone https://github.com/Analytics-MVIS/ml-tracker.git
cd ml-tracker
pip install -e .[dev]
```

Run tests locally:

```bash
pytest -q
```

Use this path only if you are changing mltracker code.

## Tracking URI behavior

Tracker URI resolution priority:

1. Explicit tracking_uri argument on tracker
2. MLTRACK_TRACKING_URI
3. MLFLOW_TRACKING_URI
4. no fallback (raises configuration error)

You must provide a server URL either by environment variable or explicitly in tracker initialization.

Example with environment variable:

```bash
export MLTRACK_TRACKING_URI="https://your-mlflow-server:5000"
```

Example passing explicitly in code:

```python
tracker = YoloTracker(
	experiment_name="YOLO",
	config=cfg,
	tracking_uri="https://your-mlflow-server:5000",
)
```

## Use with existing team scripts (minimal rewrite)

Many team scripts already use keys like `num_epochs`, `learning_rate`, `batch_size`, `dataset`, `experiment_name`, or `name`.

You can pass those dictionaries directly using factories and keep your script mostly unchanged.

### Option A: Build config from a dict

```python
from mltracker.configs import classification_config_from_dict
from mltracker.trackers import ClassificationTracker

train_params = {
	"experiment": "axle-bolt-classification",
	"run": "resnet101-05022026",
	"architecture": "resnet101",
	"dataset": "dataset4",
	"num_epochs": 50,
	"lr": 0.001,
	"batch": 16,
	"class_count": 2,
	"weight_decay": 1e-4,
}

cfg = classification_config_from_dict(train_params)

with ClassificationTracker(experiment_name="axle-bolt-classification", config=cfg) as tracker:
	tracker.log_metric("val_acc", 92.3, step=50)
```

### Option B: Build tracker directly from a dict

```python
from mltracker import build_tracker

yolo_params = {
	"experiment_name": "door-defects",
	"name": "vertical_retrain",
	"model": "yolov8m.pt",
	"data": "datasets/door/data.yaml",
	"num_epochs": 200,
	"input_size": 640,
	"batch_size": 16,
	"learning_rate": 0.01,
	"exist_ok": True,
}

tracker = build_tracker(
	"yolo",
	experiment_name="door-defect-detection",
	params=yolo_params,
	project="door-defect-detection",
)

with tracker:
	tracker.log_metric("mAP50", 0.74)
```

Any fields that are not part of the strict config schema are preserved in `extra_params` and logged under `extra.*`.

## Two-step defect identification flow

Many teams follow a two-step flow:

1. Defect detection model finds candidate defects in component images, for example spring defects.
2. Defect classification model is retrained or fine-tuned to identify the exact defect class.

The examples below show this flow using safe placeholders only.

## Step 1: Detection tracking (spring defect candidate detection)

```python
from mltracker.configs import YoloConfig
from mltracker.trackers import YoloTracker

cfg = YoloConfig(
	project="defect-detection",
	run_name="spring-detector-v1",
	model="yolov8n.pt",
	data="datasets/spring_detection.yaml",
	epochs=80,
	imgsz=640,
	batch=16,
	lr0=0.01,
	optimizer="SGD",
	extra_params={
		"pipeline.step": "detection",
		"component": "spring",
		"dataset.version": "safe-placeholder-v1",
	},
)

with YoloTracker(
    experiment_name="defect-detection",
    config=cfg,
    tracking_uri="https://your-mlflow-server:5000",
) as tracker:
    tracker.log_metric("mAP50", 0.74)
    tracker.log_metric("recall", 0.81)

    # Logs best.pt and last.pt with a clear artifact name.
    tracker.log_best_model(
        weights_path="runs/train/spring_detector/weights",
        model_name="spring-defect-detector-v1",
        register_name="defect-detector",
    )

    tracker.log_confusion_matrix("runs/train/spring_detector/confusion_matrix.png")
    tracker.log_validation_images("runs/train/spring_detector/val_images")
```

## Step 2: Classification tracking (defect class identification)

```python
from mltracker.configs import ClassificationConfig
from mltracker.trackers import ClassificationTracker

cfg = ClassificationConfig(
	project="defect-classification",
	run_name="spring-defect-classifier-v1",
	model_name="resnet50",
	dataset_name="spring_defect_patches",
	epochs=30,
	learning_rate=0.0005,
	batch_size=32,
	num_classes=6,
	extra_params={
		"pipeline.step": "classification",
		"source.detector": "spring-defect-detector-v1",
		"retrain.reason": "new defect classes",
	},
)

with ClassificationTracker(
	experiment_name="defect-classification",
	config=cfg,
	tracking_uri="https://your-mlflow-server:5000",
) as tracker:
	tracker.log_metrics({"val_acc": 0.92, "val_loss": 0.24}, step=30)

	tracker.log_model(
		model_path="runs/classifier/best.onnx",
		model_name="spring-defect-classifier-v1",
		artifact_path="model",
		register_name="defect-classifier",
	)
```

## Suggested naming convention for this flow

Use consistent and non-sensitive names so runs are easy to search in MLflow:

- Experiments: `defect-detection`, `defect-classification`
- Run names: `<component>-<task>-v<version>`
- Model names: `<component>-defect-detector-v<version>`, `<component>-defect-classifier-v<version>`

Example:

- `spring-defect-detector-v1`
- `spring-defect-classifier-v1`

## Model logging with proper names

Use one of the two supported paths depending on your training output structure.

### Option A: weights directory with best.pt and last.pt

Use log_best_model when your trainer writes best.pt and last.pt under a single directory.

```python
tracker.log_best_model(
	weights_path="runs/train/exp42/weights",
	model_name="detector-v1",           # stored as detector-v1.pt
	register_name="vision-detector-prod" # optional registry step
)
```

Behavior:

- best.pt is uploaded as model/detector-v1.pt
- last.pt is uploaded as model/last.pt (if present)
- Model registration happens only if register_name is provided
- Useful model tags are added automatically:
  - model.name
  - model.best_path
  - model.last_path
  - model.registry_name (if registered)
  - model.version (if registered)

### Option B: direct single model file

Use log_model when you have a specific file and want explicit naming.

```python
tracker.log_model(
	model_path="artifacts/checkpoints/final.onnx",
	model_name="classifier-main",
	artifact_path="model",
	register_name="vision-classifier-prod",  # optional
)
```

Behavior:

- Uploads the file under model/classifier-main.onnx or model/classifier-main.pt (suffix preserved or .pt default)
- Registers model only when register_name is provided
- Adds model.path and model.name tags

## Reliability patterns

### Context manager

```python
with YoloTracker(experiment_name="YOLO", config=cfg) as tracker:
	tracker.log_metric("loss", 0.12)
```

If an exception is raised, run status is marked FAILED automatically.

### Decorator

```python
tracker = YoloTracker(experiment_name="YOLO", config=cfg)

@tracker.run
def train_one_run() -> None:
	# training logic
	pass

train_one_run()
```

## Enforced configs and extension model

The SDK includes two base task configs:

- YoloConfig
- ClassificationConfig

Teams can extend these to add required fields while keeping platform-level consistency.

```python
from pydantic import Field
from mltracker.configs import YoloConfig

class TeamYoloConfig(YoloConfig):
	dataset_version: str = Field(..., min_length=1)
	training_recipe: str = Field(..., min_length=1)
```

At run start, all config fields are logged as params:

- Base fields (project, run_name, task-specific fields)
- Team subclass fields
- extra_params prefixed as extra.<key>

## Artifact location and storage ownership

mltracker logs artifacts through MLflow APIs only. It does not decide filesystem destinations.

Storage destination is controlled by MLflow server configuration, typically with default-artifact-root pointing to NAS/object storage.

Typical MLflow organization:

artifact-root / experiment_id / run_id / artifact_path

Teams should navigate artifacts from MLflow UI and run metadata, not by hardcoding backend paths.

## Common API summary

- start()
- end(status="FINISHED")
- fail()
- log_metric(key, value, step=None)
- log_metrics(dict, step=None)
- log_param(key, value)
- log_artifact(path, artifact_path=None)
- log_model(model_path, model_name, artifact_path="model", register_name=None)
- log_best_model(weights_path, model_name=None, register_name=None)
- log_confusion_matrix(image_path)
- log_validation_images(dir_path)

## Integration checklist for teams

- Use `with ... as tracker:` around a full train/eval run.
- Keep one run per training attempt.
- Log epoch-level metrics with `step=epoch`.
- Log your best model artifact (`log_model` or `log_best_model`).
- Use `register_name` only for models intended for shared registry usage.
- Put script-specific knobs in `extra_params` (or pass them via dict factories).
- Never log secrets or internal confidential values.

## Troubleshooting

### No run visible in MLflow UI

- Confirm URL resolution by explicitly setting tracking_uri in tracker init
- Confirm network access to your MLflow server URL
- Check experiment name used in tracker

### Model not in registry

- Ensure register_name was provided
- Ensure a run is active when log_model or log_best_model is called
- Ensure MLflow registry permissions allow create/update

### Missing GPU tags

- GPU tags are best-effort and depend on runtime tools such as nvidia-smi or nvcc
- Runs continue even when GPU tag collection is unavailable

## Maintainer notes

- `tests/sample/` contains real team reference scripts and is intentionally not collected by pytest.
- CI/unit tests validate SDK behavior and compatibility adapters without requiring heavyweight runtime dependencies such as YOLO or Torch.
