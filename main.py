import numpy as np
import matplotlib.pyplot as plt


MIN_GRID_BLOCK = 3
MAX_GRID_BLOCK = 20


class SlidingAnalyzer():

    def __init__(self):
        print("we")

    def analyze_dataset(self, dataset):
        for img in dataset:
            print(img.shape)
            self.analyze_image(img)

    def analyze_image(self, img):
        image_width = len(img)
        image_height = len(img[0])
        min_dim = min(image_width, image_height)
        print(image_height)

        for grid in range(MIN_GRID_BLOCK, MAX_GRID_BLOCK):
            analyze_image_by_grid(img, grid)
            

def analyze_image_by_grid(img, grid):
    
    image_width = len(img)
    image_heigth = len(img[0])
    min_dim = min(image_width, image_heigth)
    block_size = int(min_dim / grid)
    grid_blocks_on_width = int(image_width / block_size)
    grid_blocks_on_heigth = int(image_heigth / block_size)
    sliding_window_blocks = 3
    print(grid_blocks_on_heigth, grid_blocks_on_width, sliding_window_blocks)
    for i in range(grid_blocks_on_width - sliding_window_blocks + 1):
        for j in range(grid_blocks_on_heigth - sliding_window_blocks + 1):
            #subwindow = img[i*block_size : (i+sliding_window_blocks) * block_size, j*block_size : (j+sliding_window_blocks) * block_size]
            #analyze_window(subwindow, )
            img_tmp = img.copy()
            img_tmp[i*block_size : (i+sliding_window_blocks) * block_size, j*block_size : (j+sliding_window_blocks) * block_size] = 0.90
            img_tmp[::block_size] = 0
            img_tmp[:,::block_size] = 0
            plt.imshow(img_tmp)
            plt.show()


def analyze_image(img):
    for grid in range(MIN_GRID_BLOCK, MAX_GRID_BLOCK):
        analyze_image_by_grid(img, grid)

def main():

    dataset = np.ones((1, 300, 300, 3))

    for img in dataset:
        analyze_image(img)

    print("hello")

    
if __name__ == "__main__":
    main()