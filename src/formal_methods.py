import numpy as np

def verify_property(img, block_size):
        
    # block1
    mean1 = np.mean(img[ : block_size, : block_size])
    mean2 = np.mean(img[ : block_size, block_size : 2*block_size])
    mean3 = np.mean(img[block_size : 2*block_size, : block_size])
    mean4 = np.mean(img[block_size : 2*block_size, block_size : 2*block_size])
    block1 = mean4+4 < mean1 and mean4+4 < mean2 and mean4+4 < mean3

    # block2
    mean1 = np.mean(img[ : block_size, 2*block_size : 3*block_size])
    mean2 = np.mean(img[ : block_size, 3*block_size : 4*block_size])
    mean3 = np.mean(img[block_size : 2*block_size, 2*block_size : 3*block_size])
    mean4 = np.mean(img[block_size : 2*block_size, 3*block_size : 4*block_size])
    block2 = mean3+4 < mean1 and mean3+4 < mean2 and mean3+4 < mean4

    # block3
    mean1 = np.mean(img[ 2*block_size : 3*block_size, : block_size])
    mean2 = np.mean(img[ 2*block_size : 3*block_size, block_size : 2*block_size])
    mean3 = np.mean(img[ 3*block_size : 4*block_size, : block_size])
    mean4 = np.mean(img[ 3*block_size : 4*block_size, block_size : 2*block_size])
    block3 = mean2+4 < mean1 and mean2+4 < mean3 and mean2+4 < mean4

    # block4
    mean1 = np.mean(img[ 2*block_size : 3*block_size, 2*block_size : 3*block_size])
    mean2 = np.mean(img[ 2*block_size : 3*block_size, 3*block_size : 4*block_size])
    mean3 = np.mean(img[ 3*block_size : 4*block_size, 2*block_size : 3*block_size])
    mean4 = np.mean(img[ 3*block_size : 4*block_size, 3*block_size : 4*block_size])
    block4 = mean1+4 < mean4 and mean1+4 < mean2 and mean1+4 < mean3

    return block1 and block2 and block3 and block4