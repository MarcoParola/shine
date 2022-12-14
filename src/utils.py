from csv import writer, reader
import numpy as np
from PIL import Image
import os



def write_predicted_bbox(i,j, file_name, block_size, sliding_window_blocks):
    write_bbox(file_name, (i+1)*block_size, (j+1)*block_size, block_size*2, block_size*2)


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
                    y1 = int(line[1])
                    x1 = int(line[2])
                    y2 = y1 + int(line[3])
                    x2 = x1 + int(line[4])
                    bboxes_list.append([x1, y1, x2, y2, (x2-x1)*(y2-y1)])
        except FileNotFoundError:
            print('\nSorry, ' + os.path.join( csv_bboxes_folder, file_name +'.csv') + ' not found.\n')

        predicted_bboxes_dict[file_name] = bboxes_list
    return predicted_bboxes_dict



        

