import hydra
import asds
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np



@hydra.main(config_name="configs/config.yaml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
