"""Utilities for canonicalizing observations and serializing actions."""

from typing import List, Optional, Dict, Any
from ..uno.cards import Card, card_to_string


def canonicalize_observation(
    H_1: List[Card], opponent_size: int, deck_size: int, P_t: Optional[Card], G_o: str
) -> str:
    """
    Convert observation to canonical string key.
    Format: "hand:R1,R2,Y5|opponent:7|deck:50|top:R5|game_over:false"
    """
    # Sort hand cards for canonical representation
    sorted_hand = sorted([card_to_string(c) for c in H_1])
    hand_str = ",".join(sorted_hand)

    # Top card string
    top_str = card_to_string(P_t) if P_t else "None"

    # Game over boolean
    game_over_str = "true" if G_o == "GameOver" else "false"

    return f"hand:{hand_str}|opponent:{opponent_size}|deck:{deck_size}|top:{top_str}|game_over:{game_over_str}"


def serialize_action(action) -> Dict[str, Any]:
    """
    Convert Action object to JSON-serializable dict.
    Returns: {"type": "play", "card": [color, value]} or {"type": "draw", "n": 1}
    """
    if action.is_play():
        card = action.X_1
        color, value = card
        result = {"type": "play", "card": [color, value]}
        if action.wild_color:
            result["wild_color"] = action.wild_color
        return result
    else:
        return {"type": "draw", "n": action.n}
