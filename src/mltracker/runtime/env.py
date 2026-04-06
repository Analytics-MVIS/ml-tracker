from __future__ import annotations

import os


def resolve_tracking_uri(explicit_uri: str | None = None) -> str:
    if explicit_uri:
        return explicit_uri

    mltrack_uri = os.getenv("MLTRACK_TRACKING_URI")
    if mltrack_uri:
        return mltrack_uri

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        return mlflow_uri

    return "http://lab.l2m.internal:5000"
