import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import hydra
from src.datasets import ImageDataset
from src.utils import create_dir, get_predicted_bboxes
import cv2
import csv



@hydra.main(config_path="config", config_name="config")
def main(cfg):
    
    dataset = ImageDataset(cfg.dataset.path)
    
    for idx in range(dataset.len()):
        file_name = os.path.join(cfg.dataset.path, dataset.get_file_name_by_id(idx))
        im = np.array(Image.open(file_name))
        file_name = dataset.get_file_name_by_id(idx)
        birads = 0

        # actual bboxes
        if cfg.dataset.view == 'all' or cfg.dataset.view == 'actual':
            try:
                with open(os.path.join( cfg.dataset.path, cfg.dataset.yolo_dataset_folder, file_name +'.txt'), "r") as f:
                    reader = csv.reader(f, delimiter=" ")
                    for i, line in enumerate(reader):
                        birads = int(line[0]) +1
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
                        cv2.rectangle(im, (x1, y1), (x2, y2), (255,0,0), 3)

            except FileNotFoundError:
                print('\n\tSorry, not found.\n')
                         

        
        # predicted bboxes
        if cfg.dataset.view == 'all' or cfg.dataset.view == 'predicted':
            predicted_boxes_folder = os.path.join(cfg.output.path, cfg.output.predicted_boxes_folder)
            predicted_boxes = get_predicted_bboxes(dataset, predicted_boxes_folder)
            for bbox in predicted_boxes[file_name]:
                x,y,widht,height = bbox[0],bbox[1],bbox[2],bbox[3]
                cv2.rectangle(im, (x, y), (x+widht, y+height), (255, 255, 255), 1)

        # merged bboxes
        if cfg.dataset.view == 'all' or cfg.dataset.view == 'merged':
            predicted_merged_boxes_folder = os.path.join(cfg.output.path, cfg.output.predicted_merged_boxes_folder)
            predicted_boxes = get_predicted_bboxes(dataset, predicted_merged_boxes_folder)
            for bbox in predicted_boxes[file_name]:
                x,y,widht,height = bbox[0],bbox[1],bbox[2],bbox[3]
                cv2.rectangle(im, (x, y), (x+widht, y+height), (255, 0, 255), 2)
     
            
        if cfg.dataset.save:
            bounded_imgs_folder = os.path.join(cfg.output.path, cfg.output.bounded_imgs_folder)
            create_dir(bounded_imgs_folder)
            im = Image.fromarray(im)
            #draw = ImageDraw.Draw(im)
            #font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 55)
            #draw.text((0, 0),"Birads"+str(birads),(255,255,0),font=font) 
            img_file = os.path.join(bounded_imgs_folder, file_name )
            im.save(img_file)
        else:
            plt.imshow(np.asarray(im))
            plt.show()
    
    
if __name__ == "__main__":
    main()