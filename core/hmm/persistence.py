"""Save and load HMMEngine state to/from disk."""

import pickle
from typing import Any, Dict


def save(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)
