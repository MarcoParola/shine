from csv import writer, reader
import numpy as np
from PIL import Image
import os



def write_bbox(file_name, x, y, width, height):
    coco_row = [file_name, x, y, width, height]
    with open( file_name, 'a+') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(coco_row)
        f_object.close()


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_predicted_bboxes(dataset, csv_bboxes_folder):
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
                    bboxes_list.append([x, y, width, height, width*height])
        except FileNotFoundError:
            print('\nSorry, ' + os.path.join( csv_bboxes_folder, file_name +'.csv') + ' not found.\n')

        predicted_bboxes_dict[file_name] = bboxes_list
    return predicted_bboxes_dict


def get_actual_bboxes(dataset, csv_bboxes_folder, img_folder):
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
                    bboxes_list.append([x, y, width, height, width*height])
        except FileNotFoundError:
            print('\nSorry, ' + os.path.join( csv_bboxes_folder, file_name +'.txt') + ' not found.\n')

        actual_bboxes_dict[file_name] = bboxes_list
    return actual_bboxes_dict






        

