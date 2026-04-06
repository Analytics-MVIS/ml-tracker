# mltracker

mltracker is an internal SDK that standardizes experiment tracking for all teams.

The goal is simple: teams import mltracker and never call MLflow directly. This keeps configuration, metadata, artifact layout, and run status handling consistent across projects.

## What you get

- Enforced schemas for YOLO and classification runs
- Automatic logging of config fields as MLflow params
- Automatic system tags (git, host, python, gpu/cuda when available)
- Consistent artifact helpers for model files and evaluation outputs
- Context manager and decorator patterns for reliable run lifecycle handling
- Optional model registration in MLflow Model Registry

## Install

Internal GitLab install:

```bash
pip install git+ssh://git@gitlab.internal/ml-team/mltrack.git@v0.2.0
```

## Tracking URI behavior

Tracker URI resolution priority:

1. Explicit tracking_uri argument on tracker
2. MLTRACK_TRACKING_URI
3. MLFLOW_TRACKING_URI
4. http://lab.l2m.internal:5000

This means teams can run with no extra setup in most internal environments.

## Quick start (YOLO)

```python
from mltracker.configs import YoloConfig
from mltracker.trackers import YoloTracker

cfg = YoloConfig(
	project="vision-detection",
	run_name="yolo-v8n-baseline",
	model="yolov8n.pt",
	data="datasets/coco128.yaml",
	epochs=50,
	imgsz=640,
	batch=16,
	lr0=0.01,
	optimizer="SGD",
	extra_params={"team": "vision", "ticket": "ML-341"},
)

with YoloTracker(experiment_name="YOLO", config=cfg) as tracker:
	tracker.log_metric("mAP50", 0.72)
	tracker.log_metric("mAP50-95", 0.46)

	# Logs best.pt and last.pt from weights directory.
	# model_name controls the artifact filename used for best weights in MLflow.
	tracker.log_best_model(
		weights_path="runs/train/exp42/weights",
		model_name="yolo-v8n-coco128",
		register_name="vision-yolo-detector",
	)

	tracker.log_confusion_matrix("runs/train/exp42/confusion_matrix.png")
	tracker.log_validation_images("runs/train/exp42/val_batch")
```

## Quick start (Classification)

```python
from mltracker.configs import ClassificationConfig
from mltracker.trackers import ClassificationTracker

cfg = ClassificationConfig(
	project="vision-classification",
	run_name="resnet50-v1",
	model_name="resnet50",
	dataset_name="imagenet-subset",
	epochs=20,
	learning_rate=0.001,
	batch_size=32,
	num_classes=10,
)

with ClassificationTracker(experiment_name="Classification", config=cfg) as tracker:
	tracker.log_metrics({"val_acc": 0.91, "val_loss": 0.28}, step=20)
```

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

## Troubleshooting

### No run visible in MLflow UI

- Confirm URL resolution by explicitly setting tracking_uri in tracker init
- Confirm network access to http://lab.l2m.internal:5000
- Check experiment name used in tracker

### Model not in registry

- Ensure register_name was provided
- Ensure a run is active when log_model or log_best_model is called
- Ensure MLflow registry permissions allow create/update

### Missing GPU tags

- GPU tags are best-effort and depend on runtime tools such as nvidia-smi or nvcc
- Runs continue even when GPU tag collection is unavailable
