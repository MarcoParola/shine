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
        file_name = os.path.join(cfg.dataset.path, dataset.get_file_name_by_id(idx))
        im = np.array(Image.open(file_name))
        
        if cfg.dataset.view == 'both' or cfg.dataset.view == 'actual':
            try:
                with open(os.path.join( '/home/marco/Documents/shine/data/', 'yolo_dataset', dataset.get_file_name_by_id(idx) +'.txt'), "r") as f:
                #with open('/home/marco/Documents/shine/data/yolo_dataset/img_005.jpg.txt', "r") as f:
                    reader = csv.reader(f, delimiter=" ")
                    for i, line in enumerate(reader):
                        x1 = int( float(line[1]) * len(im[0])) 
                        y1 = int( float(line[2]) * len(im))
                        x2 = x1 + int( float(line[3]) * len(im[0]))
                        y2 = y1 + int( float(line[4]) * len(im))
                        x_shift = int( (x2 - x1) / 2)
                        y_shift = int( (y2 - y1) / 2)
                        x1 -= x_shift
                        x2 -= x_shift
                        y1 -= y_shift
                        y2 -= y_shift
                        cv2.rectangle(im, (x1, y1), (x2, y2), (255,0,0), 2)

            except FileNotFoundError:
                print('\n\tSorry, not found.\n')
                         
        #im = Image.fromarray(im)
        #im.save("/home/marco/Documents/shine/outputs/out/" + dataset.get_file_name_by_id(idx).split('.')[0] + ".jpeg")
        




#######################################################################################
        #file_name = dataset.get_file_name_by_id(idx)
        #img_file = os.path.join(cfg.dataset.path, file_name)
        #im = np.array(Image.open(img_file))

        if cfg.dataset.view == 'both' or cfg.dataset.view == 'predicted':
            file_name = dataset.get_file_name_by_id(idx)
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
                        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 1)
            except FileNotFoundError:
                print('\n\tSorry, not found!!!\n')
                    
            
            bounded_imgs_folder = os.path.join(cfg.output.path, cfg.output.bounded_imgs_folder)
            #create_dir(bounded_imgs_folder)




        plt.imshow(np.asarray(im))
        plt.show()
    
    
if __name__ == "__main__":
    main()