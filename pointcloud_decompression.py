import numpy as np
from tqdm import tqdm
from Common import *
import cv2
import pickle
from itertools import product

def split_maps(occupancy_, height_, color_, quantization):
    occupancy_maps = []
    height_maps = []
    color_maps = []

    height = (height_.astype(float)/255.0)*(quantization[1] - quantization[0]) + quantization[0]
    occupancy = occupancy_/255
    color = color_.astype(float)

    grid_size = len(occupancy)//16

    for y, x in tqdm(list(product(range(grid_size), range(grid_size)))):
        occupancy_maps.append(occupancy[y * 16 : y*16 + 16, x * 16 : x*16 + 16])
        height_maps.append(height[y * 16 : y*16 + 16, x * 16 : x*16 + 16])
        color_maps.append(color[y * 16 : y*16 + 16, x * 16 : x*16 + 16])

    return occupancy_maps, height_maps, color_maps

def reconstruct_patch(occ, height, color, position, orientation, patch_size):
    indices = np.where(occ == 1)
    points = np.hstack([indices[0].reshape(-1,1), indices[1].reshape(-1,1)]).astype(float)
    points[:,0] = dequantasize(points[:,0]/16, patch_size[0], patch_size[1])
    points[:,1] = dequantasize(points[:,1]/16, patch_size[2], patch_size[3])
    heights = height[indices].reshape(-1,1)
    points = np.hstack([points, heights])
    n = orientation
    p = position

    if abs(n[0]-1) < 1e-10:
        e1 = normalize(np.cross(n, np.array([0,1,0])))
    else:
        e1 = normalize(np.cross(n, np.array([1,0,0])))
    e2 = normalize(np.cross(n, e1))
    e3 = n

    global2local = np.array([e1, e2, e3])

    points = points @ global2local + p

    return points, color[indices]

def dequantasize(data, min, max):
    return data * (max - min) + min

def decompress(encoding = ".png"):
    occupancy_map = cv2.imread("output/occupancy.png", 0)
    height_map = cv2.imread("output/height" + encoding, 0)
    color_map = cv2.imread("output/color" + encoding)

    with open("output/quantization.bin", "rb") as file:
        quantization_data = pickle.load(file)

    patch_information = np.fromfile("output/patch_information.bin", dtype=np.float64).reshape(-1, 10)
    position = patch_information[:,:3]
    orientation = patch_information[:,3:6]
    patch_size = patch_information[:,6:]

    #position = position.astype(float)/65535
    #orientation = orientation.astype(float)/255
    #patch_size = patch_size.astype(float)/255

    position = dequantasize(position, quantization_data['position'][0], quantization_data['position'][1])
    orientation = dequantasize(orientation, quantization_data['orientation'][0], quantization_data['orientation'][1])
    patch_size = dequantasize(patch_size, quantization_data['patch_size'][0], quantization_data['patch_size'][1])


    occupancy_maps, height_maps, color_maps = split_maps(occupancy_map, height_map, color_map, quantization_data['height'])
    
    xyz = []
    rgb = []
    for id in tqdm(range(len(patch_information))):
        pos, col = reconstruct_patch(occupancy_maps[id], height_maps[id], color_maps[id], position[id], orientation[id], patch_size[id])
        xyz.append(pos)
        rgb.append(col)

    xyz = np.vstack(xyz)
    rgb = np.vstack(rgb)

    pointcloud_data = np.hstack([xyz, rgb])

    np.savetxt("decompressed.xyz", pointcloud_data, header="X Y Z R G B", fmt="%1.4e %.14e %1.4e %d %d %d")

if __name__ == "__main__":
    decompress(".jp2")