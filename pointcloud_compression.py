from unittest.mock import patch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from os.path import exists
import cv2
from Common import *
import os

quantization_data = {}

def stitch_images(grid):
    num_imgs = grid.num_images
    grid_size = int(np.sqrt(num_imgs)) + 1

    output_occ = np.zeros((grid_size*16, grid_size*16))
    output_height = np.zeros((grid_size*16, grid_size*16))
    output_col = np.zeros((grid_size*16, grid_size*16, 3))
    orientation_data = np.zeros((num_imgs, 6))
    patch_sizes = np.zeros((num_imgs, 4))

    for y in range(grid_size):
        for x in range(grid_size):
            idx = x + grid_size * y
            if idx >= num_imgs:
                return output_occ, output_height, output_col, patch_sizes, orientation_data

            vxl = grid.data[grid.filled_voxels[idx]]
            occ_img = vxl.occupancy
            output_occ[y * 16 : y*16 + 16, x * 16 : x*16 + 16] = occ_img

            height_img = vxl.height
            output_height[y * 16 : y*16 + 16, x * 16 : x*16 + 16] = height_img

            color_img = vxl.colormap
            output_col[y * 16 : y*16 + 16, x * 16 : x*16 + 16] = color_img

            orientation_data[idx] = np.array([*vxl.position, *vxl.global2local[-1]])
            patch_sizes[idx] = vxl.patch_size

    return output_occ, output_height, output_col, patch_sizes, orientation_data

def quantisize(img, name):
    amin = np.amin(img)
    amax = np.amax(img)

    quantization_data[name] = (amin, amax)

    return (img - amin) / (amax - amin)
    
def compress(path = "sample.xyz", grid_size = 128, encoding=".png"):
    for path in os.listdir("output"):
        os.remove("output/" + path)

    pc = load_pointcloud(path)
    pc.storeBinary()
    #show_cloud(pc)

    grid = VoxelGrid(grid_size)
    if exists(path + ".bin"):
        with open(path + ".bin", "rb") as file:
            grid = pickle.load(file)
    else:
        grid.fill(pc)
        #with open(path + ".bin", "wb") as file:
        #    pickle.dump(grid, file)


    grid.processAll(pc)

    occupancy, height, color, patch_sizes, orientation = stitch_images(grid)
    occupancy = (occupancy*255.0).astype(np.uint8)
    height = (quantisize(height, "height")*255).astype(np.uint8)
    color = (color).astype(np.uint8)

    plt.imshow(occupancy)
    plt.show()
    cv2.imwrite("output/occupancy.png", occupancy)

    plt.imshow(height)
    plt.show()
    cv2.imwrite("output/height" + encoding, height)
    plt.imshow(color)
    plt.show()
    cv2.imwrite("output/color" + encoding, color)

    quantization_data['voxel_size'] = grid.voxel_length
    quantization_data['orientation'] = orientation
    quantization_data['patch_size'] = patch_sizes
    with open("output/quantization.bin", "wb") as file:
        pickle.dump(quantization_data, file)

if __name__ == "__main__":
    compress("sample.xyz", 128, ".png")
    