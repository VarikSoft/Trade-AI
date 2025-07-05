import os
import importlib
from typing import List, Type
from strategy_base import StrategyBase

def discover_strategies(path: str) -> List[Type[StrategyBase]]:
    """
    Scans the directory at `path`, finds Python modules with classes
    inheriting from StrategyBase, and returns a list of those classes.
    """
    strategies = []
    for fname in os.listdir(path):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        module_name = fname[:-3]
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(path, fname))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, StrategyBase) and obj is not StrategyBase:
                strategies.append(obj)
    return strategies

def instantiate_strategies(strat_classes, param_grid: dict):
    """
    For each strategy class, creates instances with different parameters
    (param_grid is a dict: parameter_name -> list of values).
    Returns a generator yielding each strategy instance.
    """
    from itertools import product

    for cls in strat_classes:
        # Filter only the parameters that appear in the class's __init__ signature
        keys = [k for k in param_grid if k in cls.__init__.__code__.co_varnames]
        values = [param_grid[k] for k in keys]
        for combo in product(*values):
            params = dict(zip(keys, combo))
            instance = cls(**params)
            yield instance

# Example usage:
if __name__ == "__main__":
    strat_dir = os.path.join(os.path.dirname(__file__), "..", "strategies")
    strat_classes = discover_strategies(strat_dir)
    print("Found strategies:", [c.__name__ for c in strat_classes])

    # Example parameter grid for a genetic algorithm
    param_grid = {
        "fast": [5, 10, 20],
        "slow": [30, 50, 100],
    }

    for instance in instantiate_strategies(strat_classes, param_grid):
        print(instance.get_name(), instance.get_params())