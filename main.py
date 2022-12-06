import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import hydra
from src.datasets import ImageDataset
from src.algorithms import HierarchicalSliderAlghoritm


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    
    dataset = ImageDataset(cfg.dataset.path)
    hierch_slider_alg = HierarchicalSliderAlghoritm(cfg)
    hierch_slider_alg.run(dataset)
    
    
if __name__ == "__main__":
    main()