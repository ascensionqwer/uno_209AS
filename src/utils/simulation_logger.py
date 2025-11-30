import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class SimulationResult:
    timestamp: str
    config: Dict[str, Any]
    matchup: str
    total_games: int
    player_wins: Dict[str, int]
    win_rates: Dict[str, float]
    avg_decision_times: Dict[str, float]
    cache_stats: Optional[Dict[str, Any]] = None


class SimulationLogger:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def get_timestamp_filename(self) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{self.results_dir}/simulation_{timestamp}.json"
    
    def log_results(self, result: SimulationResult) -> str:
        filename = self.get_timestamp_filename()
        with open(filename, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        return filename
    
    def log_batch_results(self, results: List[SimulationResult]) -> str:
        filename = self.get_timestamp_filename()
        batch_data = {
            "batch_timestamp": datetime.now().isoformat(),
            "total_simulations": len(results),
            "results": [asdict(result) for result in results]
        }
        with open(filename, 'w') as f:
            json.dump(batch_data, f, indent=2)
        return filename