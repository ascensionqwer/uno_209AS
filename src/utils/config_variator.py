import json
import copy
import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


@dataclass
class ParameterVariant:
    parameter_path: str
    original_value: Any
    variant_value: Any
    variation_type: str  # 'increase' or 'decrease'
    percentage_change: float


class ConfigVariator:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            content = f.read()
            # Remove JSONC comments
            content = re.sub(r"//.*?\n", "\n", content)
            content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
            self.base_config = json.loads(content)
        self.config_path = config_path

    def generate_variants(
        self, percentage: float = 0.25
    ) -> List[Tuple[Dict, List[ParameterVariant]]]:
        variants = []

        # Get all numeric parameters from particle_policy section
        numeric_params = self._get_numeric_parameters(
            self.base_config["particle_policy"]
        )

        for param_path, value in numeric_params.items():
            # Generate increased variant
            increased_config = copy.deepcopy(self.base_config)
            increased_value = self._apply_variation(value, percentage, "increase")
            self._set_nested_value(
                increased_config, f"particle_policy.{param_path}", increased_value
            )

            increased_variant = ParameterVariant(
                parameter_path=f"particle_policy.{param_path}",
                original_value=value,
                variant_value=increased_value,
                variation_type="increase",
                percentage_change=percentage,
            )
            variants.append((increased_config, [increased_variant]))

            # Generate decreased variant
            decreased_config = copy.deepcopy(self.base_config)
            decreased_value = self._apply_variation(value, percentage, "decrease")
            self._set_nested_value(
                decreased_config, f"particle_policy.{param_path}", decreased_value
            )

            decreased_variant = ParameterVariant(
                parameter_path=f"particle_policy.{param_path}",
                original_value=value,
                variant_value=decreased_value,
                variation_type="decrease",
                percentage_change=percentage,
            )
            variants.append((decreased_config, [decreased_variant]))

        return variants

    def generate_baseline(self) -> Tuple[Dict, List[ParameterVariant]]:
        """Generate baseline variant with 0% change (original config).

        Returns:
            Tuple of (baseline_config, [baseline_variant])
        """
        baseline_config = copy.deepcopy(self.base_config)

        # Get all numeric parameters from particle_policy section
        numeric_params = self._get_numeric_parameters(
            self.base_config["particle_policy"]
        )

        # Create a baseline variant that represents all parameters at original values
        baseline_variants = []
        for param_path, value in numeric_params.items():
            baseline_variant = ParameterVariant(
                parameter_path=f"particle_policy.{param_path}",
                original_value=value,
                variant_value=value,
                variation_type="baseline",
                percentage_change=0.0,
            )
            baseline_variants.append(baseline_variant)

        return (baseline_config, baseline_variants)

    def _get_numeric_parameters(
        self, config_section: Dict, prefix: str = ""
    ) -> Dict[str, float]:
        numeric_params = {}

        for key, value in config_section.items():
            current_path = f"{prefix}.{key}" if prefix else key

            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_params[current_path] = float(value)
            elif isinstance(value, dict):
                nested_params = self._get_numeric_parameters(value, current_path)
                numeric_params.update(nested_params)

        return numeric_params

    def _apply_variation(
        self, value: float, percentage: float, variation_type: str
    ) -> float:
        if variation_type == "increase":
            return value * (1 + percentage)
        elif variation_type == "decrease":
            return value * (1 - percentage)
        else:
            raise ValueError(f"Unknown variation type: {variation_type}")

    def _set_nested_value(self, config: Dict, path: str, value: Any):
        keys = path.split(".")
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def save_variant(self, config: Dict, filename: str):
        with open(filename, "w") as f:
            json.dump(config, f, indent=2)
