from csv import writer, reader
import numpy as np
from PIL import Image
import os



def write_bbox(file_name, x, y, width, height):
    '''function to append the bbox information (coco format) in 
    'file_name' csv file'''
    coco_row = [file_name, x, y, width, height]
    with open( file_name, 'a+') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(coco_row)
        f_object.close()



def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def get_predicted_bboxes(dataset, csv_bboxes_folder):
    '''function to load predicted bboxes
    dataset: ImageDataset  
    csv_bboxes_folder: path string where to get bboxes'''
    predicted_bboxes_dict = {}
    for idx in range(dataset.len()):
        file_name = dataset.get_file_name_by_id(idx)
        bboxes_list = []
        try:
            with open(os.path.join( csv_bboxes_folder, file_name +'.csv'), "r") as f:
                reader_object = reader(f, delimiter=",")
                for i, line in enumerate(reader_object):
                    x = int(float(line[1]))
                    y = int(float(line[2]))
                    width = int(float(line[3]))
                    height = int(float(line[4]))
                    area = width*height # additional information to sort bboxes by area
                    bboxes_list.append([x, y, width, height, area])
        except FileNotFoundError:
            print('\nSorry, ' + os.path.join( csv_bboxes_folder, file_name +'.csv') + ' not found.\n')

        predicted_bboxes_dict[file_name] = bboxes_list
    return predicted_bboxes_dict




def get_actual_bboxes(dataset, csv_bboxes_folder, img_folder):
    '''function to load actual bboxes (YOLO format)
    dataset: ImageDataset  
    csv_bboxes_folder: path string where bboxes are stored
    img_folder: path string where images are stored (needed to retrieve the info by yolo format)'''
    actual_bboxes_dict = {}
    for idx in range(dataset.len()):
        file_name = dataset.get_file_name_by_id(idx)
        img = os.path.join(img_folder, file_name)
        img = np.array(Image.open(img))
        bboxes_list = []
        try:
            with open(os.path.join( csv_bboxes_folder, file_name +'.txt'), "r") as f:
                reader_object = reader(f, delimiter=" ")
                for i, line in enumerate(reader_object):
                    birads = int(line[0]) +1
                    width = int( float(line[3]) * len(img[0]))
                    height = int( float(line[4]) * len(img))
                    x = int( float(line[1]) * len(img[0]) - width/2)
                    y = int( float(line[2]) * len(img) - height/2)
                    bboxes_list.append([x, y, width, height])
        except FileNotFoundError:
            print('\nSorry, ' + os.path.join( csv_bboxes_folder, file_name +'.txt') + ' not found.\n')

        actual_bboxes_dict[file_name] = bboxes_list
    return actual_bboxes_dict


def from_xywh_to_x1y1x2y2_bbox(xywh_bbox):
    '''function to convert bbox encoding'''
    x1,x2 = xywh_bbox[0], xywh_bbox[0] + xywh_bbox[2]
    y1,y2 = xywh_bbox[1], xywh_bbox[1] + xywh_bbox[3]
    return [x1, y1, x2, y2]



def from_x1y1x2y2_to_xywh_bbox(x1y1x2y2_bbox):
    '''function to convert bbox encoding'''
    x,width = x1y1x2y2_bbox[0], x1y1x2y2_bbox[2] - x1y1x2y2_bbox[0]
    y,height = x1y1x2y2_bbox[1], x1y1x2y2_bbox[3] - x1y1x2y2_bbox[1]
    return [x, y, width, height]






        

