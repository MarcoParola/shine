import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import hydra
from src.datasets import ImageDataset
from src.algorithms import HierarchicalSliderAlghoritm
import cv2
import csv


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    
    dataset = ImageDataset(cfg.dataset.path)
    for idx in range(dataset.len()):
        print('-----', dataset.get_file_name_by_id(idx),'-----')
        file_name = os.path.join(cfg.dataset.path, dataset.get_file_name_by_id(idx))
        im = np.array(Image.open(file_name))
        
        try:
            with open(os.path.join( '/home/marco/Documents/shine/outputs', 'res', dataset.get_file_name_by_id(idx) +'.csv'), "r") as f:
                reader = csv.reader(f, delimiter=",")
                for i, line in enumerate(reader):
                    y1 = int(line[1])
                    x1 = int(line[2])
                    y2 = y1 + int(line[3])
                    x2 = x1 + int(line[4])
                    cv2.rectangle(im, (x1, y1), (x2, y2), (255,0,0), 2)
        except FileNotFoundError:
            print('\n\tSorry, not found.\n')
                 
        
        #plt.imshow(np.asarray(im))
        im = Image.fromarray(im)
        im.save("/home/marco/Documents/shine/outputs/out/" + dataset.get_file_name_by_id(idx).split('.')[0] + ".jpeg")
        #plt.show()
    
    
if __name__ == "__main__":
    main()