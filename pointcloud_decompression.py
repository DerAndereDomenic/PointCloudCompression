import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Common import *
import cv2
import pickle

def split_maps(occupancy_, height_, color_, quantization):
    occupancy_maps = []
    height_maps = []
    color_maps = []

    height = (height_.astype(float)/255.0)*(quantization[1] - quantization[0]) + quantization[0]
    occupancy = occupancy_/255
    color = color_.astype(float)

    grid_size = len(occupancy)//16

    for y in range(grid_size):
        for x in range(grid_size):
            occupancy_maps.append(occupancy[y * 16 : y*16 + 16, x * 16 : x*16 + 16])
            height_maps.append(height[y * 16 : y*16 + 16, x * 16 : x*16 + 16])
            color_maps.append(color[y * 16 : y*16 + 16, x * 16 : x*16 + 16])

    return occupancy_maps, height_maps, color_maps

def reconstruct_patch(occ, height, color, orientation, patch_size):
    indices = np.where(occ == 1)
    points = np.hstack([indices[0].reshape(-1,1), indices[1].reshape(-1,1)]).astype(float)
    points[:,0] = points[:,0]/16.0 * (patch_size[1] - patch_size[0]) + patch_size[0]
    points[:,1] = points[:,1]/16.0 * (patch_size[3] - patch_size[2]) + patch_size[2]
    heights = height[indices].reshape(-1,1)
    points = np.hstack([points, heights])
    n = orientation[3:]
    p = orientation[:3]

    if abs(n[0]-1) < 1e-10:
        e1 = normalize(np.cross(n, np.array([0,1,0])))
    else:
        e1 = normalize(np.cross(n, np.array([1,0,0])))
    e2 = normalize(np.cross(n, e1))
    e3 = n

    global2local = np.array([e1, e2, e3])

    points = points @ global2local + p

    return points, color[indices]

if __name__ == "__main__":
    occupancy_map = cv2.imread("output/occupancy.png", 0)
    height_map = cv2.imread("output/height.png", 0)
    color_map = cv2.imread("output/color.png")

    with open("output/quantization.bin", "rb") as file:
        quantization_data = pickle.load(file)

    occupancy_maps, height_maps, color_maps = split_maps(occupancy_map, height_map, color_map, quantization_data['height'])
    
    xyz = []
    rgb = []
    for id in range(len(quantization_data['orientation'])):
        pos, col = reconstruct_patch(occupancy_maps[id], height_maps[id], color_maps[id], quantization_data['orientation'][id], quantization_data['patch_size'][id])
        xyz.append(pos)
        rgb.append(col)

    xyz = np.vstack(xyz)
    rgb = np.vstack(rgb)

    pointcloud_data = np.hstack([xyz, rgb])

    np.savetxt("decompressed.xyz", pointcloud_data, header="X Y Z R G B", fmt="%1.4e %.14e %1.4e %d %d %d")

    