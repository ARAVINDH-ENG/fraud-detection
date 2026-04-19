import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_service.version_manager import rollback, get_active_version

current = get_active_version()
print(f"Current active: {current['version']} (trained: {current['trained_at']})")

version_id = input("Enter version to rollback to (e.g. v1): ").strip()
rollback(version_id)
print(f"Done. Restart the API server to load {version_id}.")
