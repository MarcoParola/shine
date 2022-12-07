from csv import writer
import os




def write_predicted_bbox(i,j, file_name, block_size, sliding_window_blocks):

    coco_row = [file_name, (i+1)*block_size, (j+1)*block_size, block_size*2, block_size*2]
    with open( os.path.join( '/home/marco/Documents/shine/outputs', 'res', file_name +'.csv'), 'a+') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(coco_row)
        f_object.close()
