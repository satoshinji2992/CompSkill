"""Portable paths; importing the package does not create or load anything."""
import os
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = Path(os.environ.get('COMPSKILL_WORK', 'work'))

