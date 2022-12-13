import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import hydra
from src.datasets import ImageDataset
from src.algorithms import HierarchicalSliderAlghoritm
from src.utils import create_dir
import cv2
import csv


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    
    dataset = ImageDataset(cfg.dataset.path)
    for idx in range(dataset.len()):
        print('-----', dataset.get_file_name_by_id(idx),'-----')
        
        file_name = dataset.get_file_name_by_id(idx)
        img_file = os.path.join(cfg.dataset.path, file_name)
        im = np.array(Image.open(img_file))
        
        csv_boxes_path = os.path.join(cfg.output.path, cfg.output.predicted_boxes_folder)
        create_dir(csv_boxes_path)
        
        try:
            with open(os.path.join( csv_boxes_path, file_name +'.csv'), "r") as f:
                reader = csv.reader(f, delimiter=",")
                for i, line in enumerate(reader):
                    y1 = int(line[1])
                    x1 = int(line[2])
                    y2 = y1 + int(line[3])
                    x2 = x1 + int(line[4])
                    cv2.rectangle(im, (x1, y1), (x2, y2), (255,0,0), 2)
        except FileNotFoundError:
            print('\n\tSorry, not found.\n')
                 
        
        bounded_imgs_folder = os.path.join(cfg.output.path, cfg.output.bounded_imgs_folder)
        create_dir(bounded_imgs_folder)
        
        #im = Image.fromarray(im)
        #bounded_img = os.path.join(bounded_imgs_folder, file_name)
        #im.save(bounded_img)

        plt.imshow(np.asarray(im))
        plt.show()
        
    
if __name__ == "__main__":
    main()