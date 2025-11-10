from typing import List, Set, Tuple, Optional
from cards import RED, YELLOW, GREEN, BLUE, Card

# State type: S = (H_1, H_2, D_g, P, P_t, G_o)
State = Tuple[List[Card], List[Card], List[Card], List[Card], Optional[Card], str]