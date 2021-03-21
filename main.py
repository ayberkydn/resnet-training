import hydra
from src.train import train
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="cfg", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    train(cfg)


if __name__ == "__main__":
    main()
