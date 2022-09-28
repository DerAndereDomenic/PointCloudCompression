import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from os.path import exists
import cv2
from Common import *

quantization_data = {}

def stitch_images(grid):
    num_imgs = grid.num_images
    grid_size = int(np.sqrt(num_imgs)) + 1

    output_occ = np.zeros((grid_size*16, grid_size*16))
    output_height = np.zeros((grid_size*16, grid_size*16))
    output_col = np.zeros((grid_size*16, grid_size*16, 3))
    orientation_data = np.zeros((num_imgs, 6))

    for y in range(grid_size):
        for x in range(grid_size):
            idx = x + grid_size * y
            if idx >= num_imgs:
                return output_occ, output_height, output_col, orientation_data

            vxl = grid.data[grid.filled_voxels[idx]]
            occ_img = vxl.occupancy
            output_occ[y * 16 : y*16 + 16, x * 16 : x*16 + 16] = occ_img

            height_img = vxl.height
            output_height[y * 16 : y*16 + 16, x * 16 : x*16 + 16] = height_img

            color_img = vxl.colormap
            output_col[y * 16 : y*16 + 16, x * 16 : x*16 + 16] = color_img

            orientation_data[idx] = np.array([*vxl.position, *vxl.local2global[:,-1]])

    return output_occ, output_height, output_col, orientation_data

def quantisize(img, name):
    amin = np.amin(img)
    amax = np.amax(img)

    quantization_data[name] = (amin, amax)

    return (img - amin) / (amax - amin)
    

if __name__ == "__main__":
    path = "sample.xyz"
    pc = load_pointcloud(path)
    #show_cloud(pc)

    grid = VoxelGrid(300)
    if exists(path + ".bin"):
        with open(path + ".bin", "rb") as file:
            grid = pickle.load(file)
    else:
        grid.fill(pc)
        with open(path + ".bin", "wb") as file:
            pickle.dump(grid, file)

    #points = pc.xyz[grid.data[22156].indices]
    #col = pc.rgb[grid.data[22156].indices]

    #show_cloud(points, col, point_size=1)

    grid.processAll(pc)
    #xyz = points @ grid.data[22156].local2global.T
    #show_cloud_projection(xyz, col, point_size=10)

    occupancy, height, color, orientation = stitch_images(grid)
    occupancy = (occupancy*255.0).astype(np.uint8)
    height = (quantisize(height, "height")*255).astype(np.uint8)
    color = (color).astype(np.uint8)

    #export as bitmap for now to get loss free data for now
    plt.imshow(occupancy)
    plt.show()
    cv2.imwrite("occupancy.png", occupancy)

    plt.imshow(height)
    plt.show()
    cv2.imwrite("height.png", height)
    plt.imshow(color)
    plt.show()
    cv2.imwrite("color.png", color)

    quantization_data['voxel_size'] = grid.voxel_length
    quantization_data['orientation'] = orientation
    with open("quantization.bin", "wb") as file:
        pickle.dump(quantization_data, file)