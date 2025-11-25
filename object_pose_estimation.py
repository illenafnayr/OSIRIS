import cv2
import numpy as np
import time
from sklearn.cluster import DBSCAN

orb = cv2.ORB_create(2000)
VIDEO_SRC = 0  # 0 for webcam

MIN_FEATURES_PER_OBJECT = 10
DBSCAN_EPS = 50
DBSCAN_MIN_SAMPLES = 5
MAX_CENTROID_DIST = 100

SMOOTHING_ALPHA = 0.3 

def cluster_keypoints(pts):
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(pts)
    labels = clustering.labels_
    clusters = {}
    for lbl in set(labels):
        if lbl == -1:
            continue
        indices = np.where(labels == lbl)[0]
        if len(indices) >= MIN_FEATURES_PER_OBJECT:
            clusters[lbl] = indices
    return clusters

def compute_centroid(pts, indices):
    return np.mean(pts[indices], axis=0)

def match_cluster_to_tracked(cluster_centroid, tracked_objects):
    for obj_id, obj in tracked_objects.items():
        dist = np.linalg.norm(cluster_centroid - obj['centroid'])
        if dist < MAX_CENTROID_DIST:
            return obj_id
    return None

def rotation_from_flow(prev_pts, next_pts, dt):
    motion = next_pts - prev_pts
    if len(motion) < 3:
        return np.array([0,0,1]), 0
    motion_mean = motion.mean(axis=0)
    motion_centered = motion - motion_mean
    cov = np.cov(motion_centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    axis_2d = eigvecs[:, np.argmax(eigvals)]
    axis_3d = np.array([axis_2d[0], axis_2d[1], 0.0])
    angles = np.linalg.norm(motion_centered, axis=1) / 50.0  # scale factor
    omega = np.mean(angles) / dt
    return axis_3d, omega


cap = cv2.VideoCapture(VIDEO_SRC)
cap.set(3,1280)
cap.set(4,720)

prev_gray = None
prev_pts_all = None
tracked_objects = {}
object_id_counter = 0
prev_time = time.time()

print("Starting Hybrid ORB + Optical Flow + PCA rotation tracker. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    now = time.time()
    dt = now - prev_time
    prev_time = now

    if prev_gray is None:
        # First frame: detect ORB points
        kp = orb.detect(gray, None)
        if kp is None or len(kp) < MIN_FEATURES_PER_OBJECT:
            continue
        prev_pts_all = np.array([k.pt for k in kp], dtype=np.float32)
        prev_gray = gray.copy()
        continue

    # Track points using optical flow
    next_pts_all, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts_all, None)
    status = status.flatten()
    valid_prev_pts = prev_pts_all[status==1]
    valid_next_pts = next_pts_all[status==1]

    if len(valid_next_pts) < MIN_FEATURES_PER_OBJECT:
        prev_gray = gray.copy()
        prev_pts_all = np.array([k.pt for k in orb.detect(gray, None)], dtype=np.float32)
        continue

    # Cluster points to detect objects
    clusters = cluster_keypoints(valid_next_pts)
    current_objects = {}

    for cluster_lbl, indices in clusters.items():
        centroid = compute_centroid(valid_next_pts, indices)
        matched_id = match_cluster_to_tracked(centroid, tracked_objects)
        if matched_id is None:
            object_id_counter += 1
            obj_id = object_id_counter
        else:
            obj_id = matched_id

        cluster_prev_pts = valid_prev_pts[indices]
        cluster_next_pts = valid_next_pts[indices]
        if len(cluster_prev_pts) < 3:
            axis, omega = np.array([0,0,1]), 0
        else:
            axis, omega = rotation_from_flow(cluster_prev_pts, cluster_next_pts, dt)

        # Smooth omega
        prev_omega = tracked_objects[obj_id]['omega'] if matched_id else 0.0
        omega = SMOOTHING_ALPHA * omega + (1-SMOOTHING_ALPHA) * prev_omega

        current_objects[obj_id] = {
            'indices': indices,
            'centroid': centroid,
            'axis': axis,
            'omega': omega,
        }

        # Draw bounding box and info
        pts_obj = valid_next_pts[indices]
        x, y, w, h = cv2.boundingRect(pts_obj.astype(np.int32))
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f"ID {obj_id}", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        cv2.putText(frame, f"Omega: {omega*180/np.pi:.1f} deg/s", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
        cv2.putText(frame, f"Axis: [{axis[0]:.2f},{axis[1]:.2f},{axis[2]:.2f}]", (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,0), 1)

        # Draw points
        for pt in pts_obj:
            cv2.circle(frame, tuple(pt.astype(int)), 2, (0,0,255), -1)

    cv2.imshow("Hybrid Rotation Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # Prepare for next frame
    prev_gray = gray.copy()
    prev_pts_all = valid_next_pts
    tracked_objects = current_objects

cap.release()
cv2.destroyAllWindows()
