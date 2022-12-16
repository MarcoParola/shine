import hydra
from src.datasets import ImageDataset
import os
from src.datasets import ImageDataset
from src.utils import get_predicted_bboxes, get_actual_bboxes
from src.metrics import compute_iou



@hydra.main(config_path="config", config_name="config")
def main(cfg):

    dataset = ImageDataset(cfg.dataset.path)
    actual_boxes_folder = os.path.join(cfg.dataset.path, cfg.dataset.yolo_dataset_folder)
    actual_boxes = get_actual_bboxes(dataset, actual_boxes_folder, cfg.dataset.path)
    predicted_merged_boxes_folder = os.path.join(cfg.output.path, cfg.output.predicted_merged_boxes_folder)
    predicted_boxes = get_predicted_bboxes(dataset, predicted_merged_boxes_folder)

    print(actual_boxes)
    print(predicted_boxes)
    
    
    for idx in range(dataset.len()):
        file_name = dataset.get_file_name_by_id(idx)
        print( '---', file_name, '---')
        true_positive = 0
        for actual_box in actual_boxes[file_name]:
            for predicted_box in predicted_boxes[file_name]:
                iou = compute_iou(actual_box, predicted_box[:-1])
                
                if iou > cfg.metrics.iou.threshold:
                    true_positive += 1
                    break

        false_positive = len(predicted_boxes[file_name]) - true_positive
        false_negative = len(actual_boxes[file_name]) - true_positive
        print('TP: ', true_positive, '\nFP: ', false_positive, '\nFN: ', false_negative, '\n' )
        #print(actual_box, predicted_box[:-1], iou[0,0])
    
    
if __name__ == "__main__":
    main()