import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import hydra
from src.datasets import ImageDataset
from src.algorithms import *



class SlidingAnalyzer():

    def __init__(self):
        print("we")

    def analyze_dataset(self, dataset):
        for img in dataset:
            print(img.shape)
            self.analyze_image(img)

    def analyze_image(self, img):
        image_width = len(img)
        image_height = len(img[0])
        min_dim = min(image_width, image_height)
        print(image_height)

        for grid in range(MIN_GRID_BLOCK, MAX_GRID_BLOCK):
            analyze_image_by_grid(img, grid)





@hydra.main(config_path="config", config_name="config")
def main(cfg):
    
    dataset = ImageDataset(cfg.dataset.path)
    hierch_slider = HierarchicalSlider(cfg)
    hierch_slider.run(dataset)
    
    
    

    

    
if __name__ == "__main__":
    main()