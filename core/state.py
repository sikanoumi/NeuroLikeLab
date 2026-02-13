from typing import Dict

DEFAULT_RHO = 0.93


def update_state(
    state_before: Dict[str, float],
    latent_state: Dict[str, float],
    rho: float = DEFAULT_RHO
) -> Dict[str, float]:
    """Update state with exponential smoothing and clip to [-1, 1]."""
    keys = set(state_before.keys()) | set(latent_state.keys())
    result = {}
    for k in keys:
        before = state_before.get(k, 0.0)
        latent = latent_state.get(k, 0.0)
        new_val = rho * before + (1 - rho) * latent
        result[k] = max(-1.0, min(1.0, new_val))
    return result


def get_initial_state() -> Dict[str, float]:
    """Return default initial state (all zeros)."""
    return {
        "dopamine_like": 0.0,
        "serotonin_like": 0.0,
        "adrenaline_like": 0.0,
        "cortisol_like": 0.0,
        "novelty_drive": 0.0,
        "threat_bias": 0.0,
    }
