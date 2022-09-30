import numpy as np
from Common import *
import os
from scipy.spatial import cKDTree

def compression_ratio():
    file_size_uncompressed = os.path.getsize("model.bin")
    print(f"Uncompressed filesize: {np.round(file_size_uncompressed/(1024**2),2)} MB")

    file_size_compressed = 0
    for path in os.listdir("output"):
        file_size_compressed += os.path.getsize("output/" + path)
    
    print(f"Compressed filesize: {np.round(file_size_compressed/(1024**2),2)} MB")
    print(f"Compression ratio: {np.round(file_size_compressed/file_size_uncompressed * 100, 2)} %")

def distance_error(pc_uncompressed, pc_compressed):
    P = pc_uncompressed.xyz
    Q = pc_compressed.xyz

    tree_P = cKDTree(P)
    tree_Q = cKDTree(Q)

    query_P,_ = tree_Q.query(P, k=1)
    query_Q,_ = tree_P.query(Q, k=1)

    MSE_PQ = np.mean(query_P**2)
    MSE_QP = np.mean(query_Q**2)

    RMSE_SNN = np.sqrt(0.5 * MSE_PQ + 0.5 * MSE_QP)
    print(f"RMSE_NN: {RMSE_SNN}")

def color_error(pc_uncompressed, pc_compressed):
    P = pc_uncompressed.xyz
    Q = pc_compressed.xyz

    tree_P = cKDTree(P)
    tree_Q = cKDTree(Q)

    _,query_P = tree_Q.query(P, k=1)
    _,query_Q = tree_P.query(Q, k=1)

    color_PQ = pc_compressed.rgb[query_P]
    color_QP = pc_uncompressed.rgb[query_Q]

    MSE_PQ = np.mean((color_PQ - pc_uncompressed.rgb)**2)
    MSE_QP = np.mean((color_QP - pc_compressed.rgb)**2)

    RMSE_SNN = np.sqrt(0.5 * MSE_PQ + 0.5 * MSE_QP)
    print(f"RMSE_RGB,NN: {RMSE_SNN}")

def symmetric_rmse():
    pc_uncompressed = load_pointcloud("sample.xyz")
    pc_compressed = load_pointcloud("decompressed.xyz")

    distance_error(pc_uncompressed, pc_compressed)
    color_error(pc_uncompressed, pc_compressed)
    

if __name__ == "__main__":
    compression_ratio()
    symmetric_rmse()