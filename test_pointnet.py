import os
import numpy as np
import torch
import open3d as o3d
from sklearn.cluster import DBSCAN
from tinypointnet2_model import TinyPointNetClassifier
from scipy.spatial import procrustes
import matplotlib.pyplot as plt  # <-- for easy distinct cluster colors

# ==============================
# SETTINGS (same as create_cluster.py)
# ==============================
PLY_FILE = "data/test data/test_5.ply"
MODEL_PATH = "pointnet_mannequin_classifier.pth"

SKELETON_FOLDER = "manual_skeletons"
REFERENCE_TEMPLATE = os.path.join("reference_templates", "cluster_1_skeleton.npy")

NUM_POINTS = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIDENCE_THRESHOLD = 0.5
SKELETON_MATCH_THRESHOLD = 0.5  # NOTE: used as threshold in matching function below

# --- Preprocessing  ---
VOXEL_SIZE = 0.01
PLANE_DIST_THRESH = 0.005
MAX_PLANES_TO_REMOVE = 1

USE_OUTLIER_REMOVAL = True
OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO = 2.0

DBSCAN_EPS = 0.033
DBSCAN_MIN_SAMPLES = 40
MIN_POINTS_IN_CLUSTER = 150

ENABLE_CLUSTER_MERGE = False
MERGE_DIST = 0.08

SHOW_GROUND = False
GROUND_COLOR = (0.6, 0.6, 0.6)

NOISE_GRAY = (0.5, 0.5, 0.5)     # optional: show DBSCAN noise points
MANNEQUIN_BLUE = (0.1, 0.4, 1.0)  # mannequin cluster highlight color

# ==============================
# MODEL
# ==============================
model = TinyPointNetClassifier(num_classes=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==============================
# CLUSTERING (from create_cluster.py style)
# ==============================
def segment_scene_with_dbscan(points, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, min_points=MIN_POINTS_IN_CLUSTER):
    """
    Returns:
      clusters_dict: {label: Nx3}
      noise_points: Mx3 (label == -1)
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    clusters_dict = {}
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            continue
        cluster_pts = points[labels == label]
        if cluster_pts.shape[0] < min_points:
            continue
        clusters_dict[label] = cluster_pts

    noise_points = points[labels == -1] if np.any(labels == -1) else np.empty((0, 3), dtype=np.float64)
    return clusters_dict, noise_points


def merge_close_clusters(clusters_dict, merge_dist=MERGE_DIST):
    """
    Merge clusters whose centroids are within merge_dist.
    Returns list[np.ndarray] of merged clusters.
    """
    labels = list(clusters_dict.keys())
    used = set()
    merged_clusters = []
    centroids = {lbl: clusters_dict[lbl].mean(axis=0) for lbl in labels}

    for i, lbl_i in enumerate(labels):
        if lbl_i in used:
            continue

        current_pts = clusters_dict[lbl_i]
        used.add(lbl_i)

        for lbl_j in labels[i + 1:]:
            if lbl_j in used:
                continue
            dist = np.linalg.norm(centroids[lbl_i] - centroids[lbl_j])
            if dist < merge_dist:
                current_pts = np.vstack((current_pts, clusters_dict[lbl_j]))
                used.add(lbl_j)

        merged_clusters.append(current_pts)

    return merged_clusters


def extract_dominant_planes(pcd, dist_thresh=PLANE_DIST_THRESH, max_planes=MAX_PLANES_TO_REMOVE):
    """
    Extract up to max_planes dominant planes (e.g., floor/wall).
    Returns:
      plane_pcd: extracted planes (combined)
      remaining_pcd: non-plane points
    """
    current = pcd
    plane_parts = []

    for k in range(max_planes):
        if len(current.points) < 2000:
            break

        _, inliers = current.segment_plane(
            distance_threshold=dist_thresh,
            ransac_n=3,
            num_iterations=3000
        )

        if len(inliers) < 0.25 * len(current.points):
            break

        plane_k = current.select_by_index(inliers)
        plane_parts.append(plane_k)
        current = current.select_by_index(inliers, invert=True)
        print(f"[INFO] Extracted plane {k+1}: plane_pts={len(plane_k.points)}, remaining={len(current.points)}")

    if len(plane_parts) > 0:
        plane_pcd = plane_parts[0]
        for p in plane_parts[1:]:
            plane_pcd += p
    else:
        plane_pcd = o3d.geometry.PointCloud()

    return plane_pcd, current

# ==============================
# CLASSIFIER HELPERS
# ==============================
def is_skeleton_match(candidate_skeleton_path, reference_skeleton_path, threshold=SKELETON_MATCH_THRESHOLD):
    """
    Procrustes disparity smaller => more similar.
    """
    cand = np.load(candidate_skeleton_path)
    ref = np.load(reference_skeleton_path)
    try:
        _, _, disparity = procrustes(ref, cand)
        return disparity < threshold
    except Exception as e:
        print(f"[ERROR] Skeleton matching failed: {e}")
        return False


def normalize_and_sample(cluster_pts, num_points=NUM_POINTS):
    """
    Center + scale normalization, then sample to fixed size.
    """
    centroid = np.mean(cluster_pts, axis=0)
    pts = cluster_pts - centroid
    furthest_dist = np.max(np.linalg.norm(pts, axis=1))
    if furthest_dist > 0:
        pts = pts / furthest_dist

    if pts.shape[0] >= num_points:
        idxs = np.random.choice(pts.shape[0], num_points, replace=False)
    else:
        idxs = np.random.choice(pts.shape[0], num_points, replace=True)

    return pts[idxs]


def classify_cluster(cluster_pts):
    """
    Returns: (P(background), P(mannequin))
    """
    pc = normalize_and_sample(cluster_pts, NUM_POINTS)
    x = torch.from_numpy(pc).unsqueeze(0).float().to(DEVICE)  # [1,N,3]
    x = x.permute(0, 2, 1)  # [1,3,N]
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    return float(probs[0]), float(probs[1])

# ==============================
# MAIN
# ==============================
pcd_raw = o3d.io.read_point_cloud(PLY_FILE)
if len(pcd_raw.points) == 0:
    raise RuntimeError("[ERROR] Point cloud is empty.")
print(f"[INFO] Loaded {len(pcd_raw.points)} points from {PLY_FILE}")

# Downsample + outlier removal
pcd = pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
print(f"[INFO] After voxel downsample: {len(pcd.points)} points")

if USE_OUTLIER_REMOVAL:
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=OUTLIER_NB_NEIGHBORS, std_ratio=OUTLIER_STD_RATIO)
    print(f"[INFO] After outlier removal: {len(pcd.points)} points")

# Plane extraction
plane_pcd, obj_pcd = extract_dominant_planes(pcd)
obj_points = np.asarray(obj_pcd.points)
print(f"[INFO] Object points for DBSCAN: {obj_points.shape[0]}")

if obj_points.shape[0] == 0:
    raise RuntimeError("[ERROR] No object points left after plane extraction.")

# DBSCAN + merge
clusters_dict, noise_points = segment_scene_with_dbscan(obj_points)
print(f"[INFO] DBSCAN found {len(clusters_dict)} valid clusters")
print(f"[INFO] DBSCAN noise points: {noise_points.shape[0]}")

if ENABLE_CLUSTER_MERGE and len(clusters_dict) > 0:
    clusters_list = merge_close_clusters(clusters_dict, merge_dist=MERGE_DIST)
    print(f"[INFO] After merging: {len(clusters_list)} clusters")
else:
    clusters_list = list(clusters_dict.values())

# ==============================
# DETECTION: pick BEST mannequin only (for bbox)
# ==============================
best = {"idx": None, "p_man": -1.0, "pts": None, "reason": ""}

for i, cluster_pts in enumerate(clusters_list):
    p_back, p_man = classify_cluster(cluster_pts)
    pred = 1 if p_man >= p_back else 0
    print(f"[DEBUG] Cluster {i}: pred={pred}, P(back)={p_back:.2f}, P(man)={p_man:.2f}, size={len(cluster_pts)}")

    # Only consider mannequin candidates above confidence threshold
    if pred != 1 or p_man < CONFIDENCE_THRESHOLD:
        continue

    # Optional skeleton verification (only if skeleton exists)
    skeleton_path = os.path.join(SKELETON_FOLDER, f"cluster_{i}_skeleton.npy")
    use_bbox = True
    reason = "NN"

    if os.path.exists(skeleton_path) and os.path.exists(REFERENCE_TEMPLATE):
        if is_skeleton_match(skeleton_path, REFERENCE_TEMPLATE, threshold=SKELETON_MATCH_THRESHOLD):
            reason = "NN+skeleton"
        else:
            use_bbox = False
            reason = "skeleton reject"

    if use_bbox and p_man > best["p_man"]:
        best = {"idx": i, "p_man": p_man, "pts": cluster_pts, "reason": reason}

# ==============================
# VISUALIZE: show ALL clusters, bbox ONLY for mannequin
# ==============================
geoms = []

# ground plane
if SHOW_GROUND and len(plane_pcd.points) > 0:
    plane_vis = o3d.geometry.PointCloud(plane_pcd)
    plane_vis.paint_uniform_color(GROUND_COLOR)
    geoms.append(plane_vis)

# optional: show DBSCAN noise points (gray)
if noise_points.shape[0] > 0:
    noise_pcd = o3d.geometry.PointCloud()
    noise_pcd.points = o3d.utility.Vector3dVector(noise_points)
    noise_pcd.paint_uniform_color(NOISE_GRAY)
    geoms.append(noise_pcd)

# show ALL clusters with different colors
cmap = plt.get_cmap("tab20")
cluster_pcds = []

for i, cluster_pts in enumerate(clusters_list):
    cl_pcd = o3d.geometry.PointCloud()
    cl_pcd.points = o3d.utility.Vector3dVector(cluster_pts)

    # unique-ish color per cluster
    color = cmap(i % 20)[:3]  # (r,g,b) in [0,1]
    cl_pcd.paint_uniform_color(color)
    cluster_pcds.append(cl_pcd)

geoms += cluster_pcds

# mannequin bbox only
if best["pts"] is not None and best["idx"] is not None:
    print(f"[RESULT] Mannequin cluster = {best['idx']}  P(man)={best['p_man']:.2f} ({best['p_man']*100:.1f}%) via {best['reason']}")

    # highlight mannequin cluster in blue (optional)
    if 0 <= best["idx"] < len(cluster_pcds):
        cluster_pcds[best["idx"]].paint_uniform_color(MANNEQUIN_BLUE)

    bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(best["pts"])
    )
    bbox.color = (0, 1, 0)
    geoms.append(bbox)
else:
    print("[RESULT] No mannequin detected above threshold.")

o3d.visualization.draw_geometries(geoms)
