import pointcloud_compression
import pointcloud_decompression
import evaluate

if __name__ == "__main__":
    encoding = ".png"
    cloud = "sample.xyz"
    grid_size = 128

    print("========COMPRESSION========")
    pointcloud_compression.compress(cloud, grid_size, encoding)
    print("=======DECOMPRESSION=======")
    pointcloud_decompression.decompress(encoding)
    print("========EVALUATION=========")
    evaluate.compression_ratio()
    evaluate.symmetric_rmse()