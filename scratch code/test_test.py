import open3d as o3d
import numpy as np

ply_path = "../data/mannequin/man 11.ply"

# Load with Open3D
pcd = o3d.io.read_point_cloud(ply_path)
points = np.asarray(pcd.points)

print(f"[INFO] Loaded point cloud with {len(points)} points")
print(f"[INFO] Example point: {points[0] if len(points) > 0 else 'No points found'}")
