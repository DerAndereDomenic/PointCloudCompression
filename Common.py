import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class Voxel:

    def __init__(self):
        self.indices = []
        self.occupancy = None
        self.height = None

    def push(self, idx):
        self.indices.append(idx)

    def compute_occupancy(self, points):
        patch_img,_,_ = np.histogram2d(points[:,0], points[:,1],bins=[16,16])
        patch_img[np.where(patch_img > 0)] = 1
        self.occupancy = patch_img

    def compute_heightmap(self, points):
        occur,_,_ = np.histogram2d(points[:,0], points[:,1],bins=[16,16])
        acc_height,_,_ = np.histogram2d(points[:,0], points[:,1],bins=[16,16],weights=points[:,2])
        valid_points = np.where(occur > 0)
        acc_height[valid_points] = acc_height[valid_points] / occur[valid_points]
        self.height = acc_height

    def compute_colormap(self, points, col):
        occur,_,_ = np.histogram2d(points[:,0], points[:,1],bins=[16,16])
        acc_r,_,_ = np.histogram2d(points[:,0], points[:,1],bins=[16,16],weights=col[:,0])
        acc_g,_,_ = np.histogram2d(points[:,0], points[:,1],bins=[16,16],weights=col[:,1])
        acc_b,_,_ = np.histogram2d(points[:,0], points[:,1],bins=[16,16],weights=col[:,2])
        valid_points = np.where(occur > 0)
        acc_r[valid_points] = acc_r[valid_points] / occur[valid_points]
        acc_g[valid_points] = acc_g[valid_points] / occur[valid_points]
        acc_b[valid_points] = acc_b[valid_points] / occur[valid_points]
        self.colormap = np.stack([acc_r, acc_g, acc_b], axis=-1)

    def process(self, cloud):
        points = cloud.xyz[self.indices]
        """plt.figure()
        ax = plt.subplot(projection="3d")
        ax.scatter(points[:,0], points[:,1], points[:,2])
        plt.show()"""
        self.position = np.mean(points,axis=0)
        points -= self.position
        n = fit_normal(points)

        if abs(n[0]-1) < 1e-10:
            e1 = normalize(np.cross(n, np.array([0,1,0])))
        else:
            e1 = normalize(np.cross(n, np.array([1,0,0])))
        e2 = normalize(np.cross(n, e1))
        e3 = n

        self.local2global = np.array([e1, e2, e3])
        points = points @ self.local2global.T
        self.compute_occupancy(points)
        self.compute_heightmap(points)
        self.compute_colormap(points, cloud.rgb[self.indices])


class VoxelGrid:

    def __init__(self, grid_size=32):
        self.num_voxels = grid_size
        self.data = [Voxel() for _ in range(self.num_voxels**3)]
        self.num_images = 0
        self.filled_voxels = []

    def fill(self, pc):
        bbox = pc.bbox()
        max_side_length = np.amax(np.abs(bbox[1] - bbox[0]))
        self.voxel_length = max_side_length / self.num_voxels

        for i,p in enumerate(tqdm(pc.xyz)):
            voxel = self.position2voxel(p)
            idx = self.voxel2index(voxel)
            self.data[idx].push(i)
        #22156

    def getVoxelByIndex(self, index):
        return self.data[index]

    def getVoxelByPosition(self, position):
        return self.data[self.voxel2index(self.position2voxel(position))]

    def index2voxel(self, index):
        voxel_half = self.num_voxels // 2
        x = index % self.num_voxels - voxel_half
        y = (index // self.num_voxels) % self.num_voxels - voxel_half
        z = index // (self.num_voxels * self.num_voxels) - voxel_half

        return np.array([x,y,z])

    def voxel2index(self, voxel):
        voxel_half = self.num_voxels // 2

        x = voxel[0] + voxel_half
        y = voxel[1] + voxel_half
        z = voxel[2] + voxel_half

        return x + y * self.num_voxels + z * self.num_voxels * self.num_voxels

    def voxel2position(self, voxel):
        center = np.array([voxel[0],voxel[1],voxel[2]]) * self.voxel_length
        return center

    def position2voxel(self, position):
        center = position

        x = int(np.floor(center[0] / self.voxel_length + 0.5))
        y = int(np.floor(center[1] / self.voxel_length + 0.5))
        z = int(np.floor(center[2] / self.voxel_length + 0.5))

        return np.array([x,y,z])

    def voxel_center(self, position):
        voxel = self.position2voxel(position)
        return np.array(
            [self.voxel_length * voxel[0],
            self.voxel_length * voxel[1],
            self.voxel_length * voxel[2]]
        )

    def processAll(self, cloud):
        for i,voxel in enumerate(tqdm(self.data)):
            if len(voxel.indices) > 0:
                voxel.process(cloud)
                self.num_images += 1
                self.filled_voxels.append(i)


class PointCloud:

    def __init__(self, data):
        self.data = data
        self.xyz = data[:,:3]

        self.xyz -= np.mean(self.xyz, axis=0)
        self.data[:,:3] = self.xyz

        self.rgb = data[:,3:6]

    def bbox(self):
        min_x = np.amin(self.xyz[:,0])
        min_y = np.amin(self.xyz[:,1])
        min_z = np.amin(self.xyz[:,2])

        max_x = np.amax(self.xyz[:,0])
        max_y = np.amax(self.xyz[:,1])
        max_z = np.amax(self.xyz[:,2])

        return np.array([min_x, min_y, min_z]), np.array([max_x, max_y, max_z])

def normalize(x):
    return x/np.linalg.norm(x)

def fit_normal(points):
    covariance = (points.T @ points)

    _, eigv = np.linalg.eigh(covariance)
    return normalize(eigv[:,0])

def load_pointcloud(path):
    point_cloud = np.loadtxt(path, skiprows=1, max_rows=1000000)

    return PointCloud(point_cloud)

def show_cloud(cloud, num_points=-1):
    show_cloud_points(cloud.xyz, cloud.rgb, num_points)

def show_cloud_points(xyz, rgb, num_points=-1, point_size=0.01):
    plt.figure()
    ax = plt.subplot(projection="3d")
    ax.scatter(xyz[:num_points,0], xyz[:num_points,1], xyz[:num_points,2], color=rgb[:num_points]/255, s=point_size)
    plt.show()

def show_cloud_projection(xyz, rgb, num_points=-1, point_size=0.01):
    plt.figure()
    plt.scatter(xyz[:num_points,0], xyz[:num_points,1], color=rgb[:num_points]/255, s=point_size)
    plt.show()