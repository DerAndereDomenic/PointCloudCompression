from turtle import position
from unittest.mock import patch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from os.path import exists
import cv2
from Common import *
import os
from itertools import product

quantization_data = {}

def stitch_images(grid):
    num_imgs = grid.num_images
    grid_size = int(np.sqrt(num_imgs)) + 1

    output_occ = np.zeros((grid_size*16, grid_size*16))
    output_height = np.zeros((grid_size*16, grid_size*16))
    output_col = np.zeros((grid_size*16, grid_size*16, 3))
    position_data = np.zeros((num_imgs, 3))
    orientation_data = np.zeros((num_imgs, 3))
    patch_sizes = np.zeros((num_imgs, 4))

    #Sort maps according to hue...
    voxels = [grid.data[grid.filled_voxels[x + grid_size * y]] for y, x in list(product(range(grid_size), range(grid_size))) if x + grid_size * y < num_imgs]
    colormaps = [vxl.colormap for vxl in voxels]
    hue_maps = np.array([cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)[...,0].flatten() for img in colormaps])
    hues_sorted = np.argsort(np.mean(hue_maps, axis=-1)).flatten()

    for y, x in tqdm(list(product(range(grid_size), range(grid_size)))):
        idx = x + grid_size * y
        if idx >= num_imgs:
            break

        vxl = voxels[hues_sorted[idx]]#grid.data[grid.filled_voxels[idx]]
        occ_img = vxl.occupancy
        output_occ[y * 16 : y*16 + 16, x * 16 : x*16 + 16] = occ_img

        height_img = vxl.height
        output_height[y * 16 : y*16 + 16, x * 16 : x*16 + 16] = height_img

        color_img = vxl.colormap
        output_col[y * 16 : y*16 + 16, x * 16 : x*16 + 16] = color_img

        position_data[idx] = vxl.position
        orientation_data[idx] = vxl.global2local[-1]
        patch_sizes[idx] = vxl.patch_size

    return output_occ, output_height, output_col, patch_sizes, position_data, orientation_data

def quantisize(data, name):
    amin = np.amin(data)
    amax = np.amax(data)

    quantization_data[name] = (amin, amax)

    return (data - amin) / (amax - amin)

def blurr(img, occupancy):
    img_copy = img.copy()
    for _ in range(3):
        img_copy = cv2.boxFilter(img_copy, -1, (5,5))

    valid = np.where(occupancy == 255)
    img_copy[valid] = img[valid]
    return img_copy
    
def compress(cloud_path = "sample.xyz", grid_size = 128, encoding=".png"):
    for path in os.listdir("output"):
        os.remove("output/" + path)

    pc = load_pointcloud(cloud_path)
    pc.storeBinary()
    #show_cloud(pc)

    grid = VoxelGrid(grid_size)
    if exists(cloud_path + ".bin"):
        with open(cloud_path + ".bin", "rb") as file:
            grid = pickle.load(file)
    else:
        grid.fill(pc)
        #with open(cloud_path + ".bin", "wb") as file:
        #    pickle.dump(grid, file)


    grid.processAll(pc)

    occupancy, height, color, patch_sizes, position, orientation = stitch_images(grid)
    occupancy = (occupancy*255.0).astype(np.uint8)
    height = (quantisize(height, "height")*255).astype(np.uint8)
    color = (color).astype(np.uint8)

    #Blurring actually increases file size?
    #height = blurr(height, occupancy)
    #color = blurr(color, occupancy)

    #plt.imshow(occupancy)
    #plt.show()
    cv2.imwrite("output/occupancy.png", occupancy)

    #plt.imshow(height)
    #plt.show()
    cv2.imwrite("output/height" + encoding, height)

    #plt.imshow(color)
    #plt.show()
    cv2.imwrite("output/color" + encoding, color)

    position = quantisize(position, "position")
    orientation = quantisize(orientation, "orientation")
    patch_sizes = quantisize(patch_sizes, "patch_size")
    
    position = ((position*65535).astype(np.uint16)).view(np.uint8)
    orientation = (orientation*255).astype(np.uint8)
    patch_sizes = (patch_sizes*255).astype(np.uint8)

    quantisized_data = np.hstack([position, orientation, patch_sizes])
    quantisized_data.tofile("output/patch_information.bin")
    print(quantisized_data.shape)

    with open("output/quantization.bin", "wb") as file:
        pickle.dump(quantization_data, file)

if __name__ == "__main__":
    compress("sample.xyz", 128, ".png")
    