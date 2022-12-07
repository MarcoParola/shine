import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import hydra

class ImageDataset():
    def __init__(self, data_path):
        self.data_path = data_path
        self.img_names = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    
    def get_item(self, idx):
        file_name = os.path.join(self.data_path, self.img_names[idx])
        im = Image.open(file_name)
        return np.asarray(im)

    def get_file_name_by_id(self, idx):
        return self.img_names[idx]

    def len(self):
        return len(self.img_names)


# testing main
@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    
    dataset = ImageDataset(cfg.dataset.path)
    print(type(dataset))
    plt.imshow((dataset.get_item(0)))
    plt.show()
    
if __name__ == "__main__":
    main()