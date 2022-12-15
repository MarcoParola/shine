import hydra
from src.datasets import ImageDataset
import os
from src.datasets import ImageDataset
from src.utils import get_predicted_bboxes, get_actual_bboxes



@hydra.main(config_path="config", config_name="config")
def main(cfg):

    dataset = ImageDataset(cfg.dataset.path)
    actual_boxes_folder = os.path.join(cfg.dataset.path, cfg.dataset.yolo_dataset_folder)
    actual_boxes = get_actual_bboxes(dataset, actual_boxes_folder, cfg.dataset.path)
    predicted_merged_boxes_folder = os.path.join(cfg.output.path, cfg.output.predicted_merged_boxes_folder)
    predicted_boxes = get_predicted_bboxes(dataset, predicted_merged_boxes_folder)

    for idx in range(dataset.len()):
        file_name = dataset.get_file_name_by_id(idx)
        for bbox in actual_boxes[file_name]:
            print('\n', bbox, '\n', predicted_boxes[file_name] , '\n')
    
if __name__ == "__main__":
    main()