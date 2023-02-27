from engine.get_point_cloud import read_point_cloud
import open3d as o3d
import numpy as np

if __name__ == '__main__':
    pcd = read_point_cloud(filepath='../logs/point_cloud',filename='point_clouds.txt')
    print(pcd)
    pcd.paint_uniform_color([0.5,0.5,0.5])

    labels = np.array(pcd.cluster_dbscan(eps=0.25,min_points=16,print_progress=True))

    max_label = np.max(labels)
    print(max(labels))