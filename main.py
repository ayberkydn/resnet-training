import hydra
from src.train import train


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    train(cfg)

if __name__ == "__main__":
    main()
