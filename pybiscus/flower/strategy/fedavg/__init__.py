
from typing import Dict, List, Tuple
from pydantic import BaseModel
from flwr.server.strategy import Strategy

from pybiscus.flower.strategy.fedavg.fedavgstrategy import FabricFedAvgStrategy, ConfigFabricFedAvgStrategy

def get_modules_and_configs() -> Tuple[Dict[str, Strategy], List[BaseModel]]:

    registry = {"fedavg": FabricFedAvgStrategy,}
    configs  = [ConfigFabricFedAvgStrategy,]

    return registry, configs
