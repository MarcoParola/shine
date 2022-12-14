import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.utils import write_predicted_bbox, create_dir, get_predicted_bboxes, write_bbox
from src.formal_methods import verify_property


import torch
import torchvision.ops.boxes as bops

class SlidingHierarchicalImageTraversalAlghoritm():

    def __init__(self, cfg):
        self.min_grid_block = cfg.algorithm.min_grid_block
        self.max_grid_block = cfg.algorithm.max_grid_block
        self.sliding_window_blocks = cfg.algorithm.sliding_window_blocks
        self.predicted_boxes_folder = os.path.join(cfg.output.path, cfg.output.predicted_boxes_folder)
        create_dir(self.predicted_boxes_folder)


    def analyze_image_by_grid(self, img, file_name, grid):
    
        image_width = len(img)
        image_heigth = len(img[0])
        min_dim = min(image_width, image_heigth)
        block_size = int(min_dim / grid)
        grid_blocks_on_width = int(image_width / block_size)    
        grid_blocks_on_heigth = int(image_heigth / block_size)
        for i in range(grid_blocks_on_width - self.sliding_window_blocks + 1):
            for j in range(grid_blocks_on_heigth - self.sliding_window_blocks + 1):
                img_cropped = img[i*block_size : (i+self.sliding_window_blocks) * block_size, j*block_size : (j+self.sliding_window_blocks) * block_size].copy() 
                detect = verify_property(img_cropped, block_size)
                '''
                plt.xticks([], [])
                plt.yticks([], [])
                img_tmp1 = img.copy()
                img_tmp2 = img.copy()
                img_tmp1[i*block_size : (i+self.sliding_window_blocks) * block_size, j*block_size : (j+self.sliding_window_blocks) * block_size] = img_tmp1[i*block_size : (i+self.sliding_window_blocks) * block_size, j*block_size : (j+self.sliding_window_blocks) * block_size] / 1.7
                img_tmp1[::block_size] = 0
                img_tmp1[:,::block_size] = 0
                img_tmp2[::block_size] = 255
                img_tmp2[:,::block_size] = 255
                img_tmp2[1::block_size] = 255
                img_tmp2[:,1::block_size] = 255
                img_tmp2[2::block_size] = 255
                img_tmp2[:,2::block_size] = 255
                plt.imshow(img_tmp2)
                plt.show()
                break
                '''
                if detect:
                    file = os.path.join(self.predicted_boxes_folder, file_name + '.csv')
                    write_predicted_bbox(i,j, file, block_size, self.sliding_window_blocks)
                    #self.plot_grid_and_distribution(img_tmp1, img_tmp2[i*block_size : (i+self.sliding_window_blocks) * block_size, j*block_size : (j+self.sliding_window_blocks) * block_size], block_size, grid, i, j)
            

    
            

    def analyze_image(self, img, file_name):
        for grid in range(self.min_grid_block, self.max_grid_block):
            print(grid)
            self.analyze_image_by_grid(img, file_name, grid)


    def visit(self, dataset):
        for idx in range(dataset.len()):
            print('-----', dataset.get_file_name_by_id(idx), '-----')
            self.analyze_image(dataset.get_item(idx), dataset.get_file_name_by_id(idx))



    def plot_grid_and_distribution(self, img, img_cropped, block_size, grid, idx, idy):
        
        fig = plt.figure(figsize=(12, 9))
        outer = gridspec.GridSpec(2, 1, wspace=0.12, hspace=0.12, height_ratios=[5,7])

        inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0], wspace=0.1, hspace=0.1)
        ax = plt.Subplot(fig, inner[0])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)

        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.1, hspace=0.0)
        subinner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=inner[0], wspace=0.1, hspace=0.1)
        ax = plt.Subplot(fig, subinner[0])
        ax.imshow(img_cropped)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)

        subinner = gridspec.GridSpecFromSubplotSpec(self.sliding_window_blocks, self.sliding_window_blocks, subplot_spec=inner[1], wspace=0.3, hspace=0.3)
        for j in range(self.sliding_window_blocks ** 2):
            ax = plt.Subplot(fig, subinner[j])
            custom_xlim = (0, 255)
            custom_ylim = (0, block_size**2/1.6)
            plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
            if j%self.sliding_window_blocks != 0:
                plt.setp(ax, yticks=[])
            if j/self.sliding_window_blocks < 2:
                plt.setp(ax, xticks=[])
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            cell = img_cropped[ int(j/self.sliding_window_blocks)*block_size : (int(j/self.sliding_window_blocks)+1)*block_size , (j%self.sliding_window_blocks)*block_size : ((j%self.sliding_window_blocks)+1)*block_size ].flatten()
            ax.hist(cell, bins=40, alpha=0.75)
            quantiles = np.quantile(cell, [0.25,0.5,0.75])
            for quantile in quantiles:
                ax.axvline(quantile, color='k', linestyle='dashed', linewidth=1, alpha=.5)
            ax.text(138, block_size**2/2.5, 'q1='+ str(int(np.mean([quantiles[1], quantiles[0]]))) + '\nq2='+ str(int( np.mean([quantiles[1], quantiles[2]]))) )
            fig.add_subplot(ax)

        plt.show()
        #plt.savefig(os.path.join('data', 'tmp','img' + str(grid) + str(idx) + str(idy) + '.png'))


class ExplainableDetector:

    def __init__(self, cfg):
        self.min_grid_block = cfg.algorithm.min_grid_block
        self.max_grid_block = cfg.algorithm.max_grid_block
        self.sliding_window_blocks = cfg.algorithm.sliding_window_blocks
        self.shita = SlidingHierarchicalImageTraversalAlghoritm(cfg)
        self.predicted_boxes_folder = os.path.join(cfg.output.path, cfg.output.predicted_boxes_folder)
        self.predicted_merged_boxes_folder = os.path.join(cfg.output.path, cfg.output.predicted_merged_boxes_folder)
        create_dir(self.predicted_merged_boxes_folder)

    def run(self, dataset):
        self.shita.visit(dataset)
        predicted_bboxes = get_predicted_bboxes(dataset, self.predicted_boxes_folder)
        self.merge_bboxes_by_iou(predicted_bboxes, self.predicted_merged_boxes_folder)


    
    def merge_bboxes_by_iou(self, predicted_bboxes_dict, predicted_merged_boxes_folder):
        for key in predicted_bboxes_dict.keys():
            file_name = os.path.join(predicted_merged_boxes_folder, key + '.csv')
            self.merge_bbox(file_name, predicted_bboxes_dict[key])
            


    def merge_bbox(self, file_name, bboxes_list):

        merged_bboxes = []
        for bbox in bboxes_list:
            if len(merged_bboxes) == 0:
                merged_bboxes.append([bbox[:-1], 1])

            iou_check = True
            for i in range(len(merged_bboxes)):
                box1 = torch.tensor([merged_bboxes[i][0]], dtype=torch.float)
                box2 = torch.tensor([bbox[:-1]], dtype=torch.float)
                iou = bops.box_iou(box1, box2)
                
                if iou > 0:
                    iou_check = False
                    merged_bboxes[i][0] = ((box1*merged_bboxes[i][1] + box2) / (merged_bboxes[i][1]+1) ).tolist()[0]
                    merged_bboxes[i][1] += 1
            if iou_check:
                merged_bboxes.append([bbox[:-1], 1])

        merged_bboxes = [bbox[0] for bbox in merged_bboxes]

        for bbox in merged_bboxes:
            x = bbox[0]
            y = bbox[1]
            width = bbox[2]
            height = bbox[3]
            write_bbox(file_name, x, y, width, height)

        

