# v9 (round 3, exclusive claims): claim the uncontested middle band of rewards.
#
# Rules: exactly one accepter gets the reward; two or more accepters all score zero; every
# accepter is locked out for `lockout` turns. Rewards ~ Exponential(mean 10).
# Field observed on the server: many teams use the round-1 optimal cut (~20) and collide on
# large rewards; near-greedy teams take small rewards. The middle band is the least crowded.
# Band chosen from a plateau (8-12 lower, 14-20 upper) in simulations against the observed field.

BAND_LOW, BAND_HIGH = 9.0, 16.0


def play(reward: float, lockout: int, t: int, T: int, your_score: float,
         other_scores: list[tuple[float, bool]], is_locked: bool) -> bool:
    try:
        if is_locked:
            return False
        r = float(reward)
        return BAND_LOW <= r <= BAND_HIGH
    except Exception:
        return False
