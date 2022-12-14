import hydra
from src.datasets import ImageDataset
from src.algorithms import ExplainableDetector


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    dataset = ImageDataset(cfg.dataset.path)
    detector = ExplainableDetector(cfg)
    detector.detect(dataset)
    
    
if __name__ == "__main__":
    main()