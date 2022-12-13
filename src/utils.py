from csv import writer
import os




def write_predicted_bbox(i,j, file_name, block_size, sliding_window_blocks):

    coco_row = [file_name, (i+1)*block_size, (j+1)*block_size, block_size*2, block_size*2]
    with open( file_name, 'a+') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(coco_row)
        f_object.close()

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

