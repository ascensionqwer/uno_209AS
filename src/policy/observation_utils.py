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


def canonicalize_game_state(
    H_1: List[Card],
    opponent_size: int,
    deck_size: int,
    P: List[Card],
    P_t: Optional[Card],
    G_o: str,
    action_history: Optional[List[Any]] = None,
) -> str:
    """
    Convert full game state to canonical string key for caching.
    Includes action history for disambiguation.
    Format: "hand:R1,R2|opponent:7|deck:50|pile:R5,Y3|top:R5|game_over:false|history:..."
    """
    # Sort hand cards for canonical representation
    sorted_hand = sorted([card_to_string(c) for c in H_1])
    hand_str = ",".join(sorted_hand)

    # Sort pile cards (excluding top card) for canonical representation
    pile_without_top = [c for c in P if c != P_t]
    sorted_pile = sorted([card_to_string(c) for c in pile_without_top])
    pile_str = ",".join(sorted_pile) if sorted_pile else ""

    # Top card string
    top_str = card_to_string(P_t) if P_t else "None"

    # Game over boolean
    game_over_str = "true" if G_o == "GameOver" else "false"

    # Action history (serialized)
    history_str = ""
    if action_history:
        history_parts = []
        for action in action_history:
            if hasattr(action, "is_play") and action.is_play():
                card_str = card_to_string(action.X_1)
                if action.wild_color:
                    history_parts.append(f"play:{card_str}:{action.wild_color}")
                else:
                    history_parts.append(f"play:{card_str}")
            elif hasattr(action, "is_draw") and action.is_draw():
                history_parts.append(f"draw:{action.n}")
            else:
                # Fallback: string representation
                history_parts.append(str(action))
        history_str = "|".join(history_parts)

    return f"hand:{hand_str}|opponent:{opponent_size}|deck:{deck_size}|pile:{pile_str}|top:{top_str}|game_over:{game_over_str}|history:{history_str}"


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
