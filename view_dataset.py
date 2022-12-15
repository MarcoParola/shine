import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import hydra
from src.datasets import ImageDataset
from src.utils import create_dir, get_predicted_bboxes, get_actual_bboxes
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
            actual_boxes_folder = os.path.join(cfg.dataset.path, cfg.dataset.yolo_dataset_folder)
            actual_boxes = get_actual_bboxes(dataset, actual_boxes_folder, cfg.dataset.path)
            print(actual_boxes[file_name])
            for bbox in actual_boxes[file_name]:
                x,y,widht,height = bbox[0],bbox[1],bbox[2],bbox[3]
                print(x,y,widht,height)
                cv2.rectangle(im, (x1, y1), (x2, y2), (255,0,0), 3)

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