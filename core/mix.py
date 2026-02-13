import json
from pathlib import Path
from typing import Dict

PROFILES_PATH = Path(__file__).parent.parent / "data" / "modulator_profiles.json"

_profiles: Dict[str, Dict[str, float]] = {}


def _load_profiles() -> Dict[str, Dict[str, float]]:
    global _profiles
    if not _profiles:
        with open(PROFILES_PATH, "r", encoding="utf-8") as f:
            _profiles = json.load(f)
    return _profiles


def compute_latent_state(emotion: Dict[str, float]) -> Dict[str, float]:
    """Compute weighted sum of modulator profiles based on emotion weights."""
    profiles = _load_profiles()
    keys = ["dopamine_like", "serotonin_like", "adrenaline_like", "cortisol_like", "novelty_drive", "threat_bias"]
    result = {k: 0.0 for k in keys}

    for emo_label, weight in emotion.items():
        if emo_label in profiles:
            for k in keys:
                result[k] += profiles[emo_label].get(k, 0.0) * weight

    return result
