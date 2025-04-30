from typing import Literal, ClassVar
from pydantic import BaseModel, ConfigDict

class ConfigHDFS(BaseModel):
    """Pydantic Model used to validate the LightningDataModule config

    Attributes
    ----------
    dir_train:   str, optional = the training data directory path (required for clients)
    dir_val:     str, optional = the validating data directory path (required for clients)
    dir_test:    str, optional = the testing data directory path (required for server)
    batch_size:  int, optional = the batch size (default to 32)
    num_workers: int, optional = the number of workers for the DataLoaders (default to 0)
    """

    PYBISCUS_CONFIG: ClassVar[str] = "config"

    num_samples: int = 100
    feature_dim: int = 1
    seed:        int = 42
    batch_size:  int = 32

# --- Pybiscus HDFS configuration definition 

class ConfigData_RandomVector(BaseModel):

    PYBISCUS_ALIAS: ClassVar[str] = "Random vector"

    name:   Literal["randomvector"]
    config: ConfigHDFS

    model_config = ConfigDict(extra="forbid")
