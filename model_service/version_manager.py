import os
import json
from datetime import datetime

VERSION_REGISTRY = "models/version_registry.json"

def register_model(model_path: str, metrics: dict) -> str:
    if os.path.exists(VERSION_REGISTRY):
        with open(VERSION_REGISTRY, "r") as f:
            registry = json.load(f)
    else:
        registry = {"versions": [], "active": None}

    version_id = f"v{len(registry['versions']) + 1}"
    entry = {
        "version":    version_id,
        "path":       model_path,
        "trained_at": datetime.utcnow().isoformat(),
        "metrics":    metrics
    }
    registry["versions"].append(entry)
    registry["active"] = version_id

    with open(VERSION_REGISTRY, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"Model {version_id} registered as active.")
    return version_id

def get_active_version() -> dict:
    with open(VERSION_REGISTRY, "r") as f:
        registry = json.load(f)
    active_id = registry["active"]
    return next(v for v in registry["versions"] if v["version"] == active_id)

def rollback(version_id: str):
    with open(VERSION_REGISTRY, "r") as f:
        registry = json.load(f)
    registry["active"] = version_id
    with open(VERSION_REGISTRY, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"Rolled back to {version_id}")
