import cv2
import numpy as np
import time
from sklearn.cluster import DBSCAN
from collections import deque

# ---------------- Config ----------------
VIDEO_SRC = 0
FRAME_W, FRAME_H = 1280, 720
FPS_TARGET = 30

orb = cv2.ORB_create(5000)

# Clustering
MIN_FEATURES_PER_OBJECT = 15
DBSCAN_EPS = 35
DBSCAN_MIN_SAMPLES = 6
MAX_CENTROID_DIST = 200

# Motion filtering
MIN_MOTION_THRESHOLD = 0.3
MIN_ROTATION_THRESHOLD = 0.05
STATIC_FRAMES_THRESHOLD = 5

# Smoothing
ALPHA_V = 0.2  # velocity smoothing
ALPHA_OMEGA = 0.2  # angular velocity smoothing
HISTORY_LENGTH = 15

# Optical flow
LK_PARAMS = dict(winSize=(21,21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# ---------------- Helpers ----------------
def cluster_keypoints(pts):
    if len(pts) < DBSCAN_MIN_SAMPLES: return {}
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(pts)
    clusters = {}
    for lbl in set(clustering.labels_):
        if lbl == -1: continue
        indices = np.where(clustering.labels_ == lbl)[0]
        if len(indices) >= MIN_FEATURES_PER_OBJECT:
            clusters[lbl] = indices
    return clusters

def compute_motion_magnitude(prev_pts, next_pts):
    motion = next_pts - prev_pts
    return np.mean(np.linalg.norm(motion, axis=1))

def rotation_from_flow(prev_pts, next_pts, centroid, dt):
    if len(prev_pts)<5 or dt<1e-3: return np.array([0,0,1]),0,0
    prev_centered = prev_pts - centroid
    next_centered = next_pts - centroid
    distances = np.linalg.norm(prev_centered, axis=1)
    valid_mask = distances>5
    if np.sum(valid_mask)<5: return np.array([0,0,1]),0,0
    prev_centered = prev_centered[valid_mask]
    next_centered = next_centered[valid_mask]
    motion = next_centered - prev_centered
    radial = prev_centered / (np.linalg.norm(prev_centered, axis=1, keepdims=True)+1e-6)
    tangential = motion - (motion*radial).sum(axis=1, keepdims=True)*radial
    omega = np.median(np.linalg.norm(tangential, axis=1)/distances[valid_mask])/dt
    axis_estimate = np.mean(np.cross(prev_centered, tangential), axis=0)
    axis_norm = np.linalg.norm(axis_estimate)
    axis = axis_estimate/axis_norm if axis_norm>1e-6 else np.array([0,0,1])
    consistency = 1.0 - np.std(np.linalg.norm(tangential, axis=1)/(distances[valid_mask]+1e-6))/(np.mean(np.linalg.norm(tangential, axis=1)/(distances[valid_mask]+1e-6))+1e-6)
    consistency = np.clip(consistency,0,1)
    return axis, omega, consistency

def detect_new_features(gray, mask=None):
    kp = orb.detect(gray, mask)
    if kp and len(kp)>0:
        return np.array([k.pt for k in kp], dtype=np.float32)
    return np.array([])

# ---------------- Tracker State ----------------
class TrackedObject:
    def __init__(self):
        self.centroid = None
        self.velocity = np.array([0.0,0.0])
        self.axis = np.array([0,0,1])
        self.omega = 0.0
        self.motion_history = deque(maxlen=HISTORY_LENGTH)
        self.omega_history = deque(maxlen=HISTORY_LENGTH)
        self.static_counter = 0
        self.centroid_history = deque(maxlen=HISTORY_LENGTH)
        self.last_update = time.time()

tracker = TrackedObject()

# ---------------- Main Loop ----------------
cap = cv2.VideoCapture(VIDEO_SRC)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

prev_gray = None
prev_pts = None
last_feature_refresh = 0
frame_count = 0
prev_time = time.time()

print("Smooth Single-Object Tracker")

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)
    now = time.time()
    dt = max(now-prev_time, 1e-3)
    prev_time = now
    frame_count +=1

    # initialize features
    if prev_gray is None or prev_pts is None or len(prev_pts)<50:
        prev_pts = detect_new_features(gray)
        prev_gray = gray.copy()
        continue

    # optical flow
    next_pts, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **LK_PARAMS)
    if next_pts is None: 
        prev_pts = detect_new_features(gray)
        prev_gray = gray.copy()
        continue
    good_mask = (status.flatten()==1) & (error.flatten()<50)
    valid_prev = prev_pts[good_mask]
    valid_next = next_pts[good_mask]

    # clustering
    clusters = cluster_keypoints(valid_next)
    if not clusters:
        prev_pts = detect_new_features(gray)
        prev_gray = gray.copy()
        continue

    # choose best cluster: closest to previous centroid or center
    best_cluster = None
    min_dist = float('inf')
    for lbl, indices in clusters.items():
        c = np.mean(valid_next[indices], axis=0)
        if tracker.centroid is not None:
            dist = np.linalg.norm(c - tracker.centroid)
        else:
            dist = np.linalg.norm(c - np.array([FRAME_W/2, FRAME_H/2]))
        if dist<min_dist:
            min_dist = dist
            best_cluster = (lbl, indices)
    cluster_lbl, indices = best_cluster
    cluster_prev_pts = valid_prev[indices]
    cluster_next_pts = valid_next[indices]
    centroid_prev = np.mean(cluster_prev_pts, axis=0)
    centroid_next = np.mean(cluster_next_pts, axis=0)
    velocity = (centroid_next - centroid_prev)/dt

    # rotation
    axis, omega, quality = rotation_from_flow(cluster_prev_pts, cluster_next_pts, centroid_prev, dt)

    # low-pass filter velocities
    tracker.velocity = ALPHA_V*velocity + (1-ALPHA_V)*tracker.velocity
    tracker.omega = ALPHA_OMEGA*omega + (1-ALPHA_OMEGA)*tracker.omega
    tracker.axis = axis
    tracker.centroid = centroid_next
    tracker.motion_history.append(np.linalg.norm(velocity))
    tracker.omega_history.append(tracker.omega)
    tracker.centroid_history.append(centroid_next)
    if np.mean(tracker.motion_history)<MIN_MOTION_THRESHOLD:
        tracker.static_counter +=1
    else:
        tracker.static_counter =0
    if tracker.static_counter>STATIC_FRAMES_THRESHOLD:
        tracker.omega = 0.0

    # Visualization
    hull = cv2.convexHull(cluster_next_pts.astype(np.int32))
    rotation_intensity = min(abs(tracker.omega)/5,1)
    rotation_color = (0, int(255*rotation_intensity), int(255*(1-rotation_intensity)))
    cv2.polylines(frame,[hull],True,rotation_color,2)

    c = tracker.centroid.astype(int)
    cv2.circle(frame, tuple(c), 5, (0,255,0), -1)
    cv2.line(frame, tuple(c), tuple(c + (tracker.velocity*10).astype(int)), (255,0,0),2)
    length=30
    x_axis = c + np.array([length,0])
    y_axis = c + np.array([0,length])
    cv2.line(frame, tuple(c), tuple(x_axis), (0,0,255),2)
    cv2.line(frame, tuple(c), tuple(y_axis), (0,255,0),2)
    cv2.putText(frame, f"w:{tracker.omega*180/np.pi:.1f}deg/s", tuple(c+np.array([0,-30])), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
    cv2.putText(frame, f"vel:{np.linalg.norm(tracker.velocity):.1f}px/s", tuple(c+np.array([0,-15])), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)
    cv2.putText(frame, f"Q:{quality:.2f}", tuple(c+np.array([0,0])), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,200,0),1)

    # feature refresh
    if frame_count - last_feature_refresh > 15 or len(prev_pts)<100:
        prev_pts = detect_new_features(gray)
        last_feature_refresh = frame_count

    prev_pts = valid_next
    prev_gray = gray
    cv2.imshow("Smooth Single Object Tracker", frame)
    key = cv2.waitKey(1) & 0xFF
    if key==27: break
    elif key==ord('r'):
        prev_pts=None; prev_gray=None; tracker=TrackedObject()
        print("Tracker reset")
    elif key==ord('f'):
        prev_pts = detect_new_features(gray)
        print(f"Forced feature refresh: {len(prev_pts)} features")

cap.release()
cv2.destroyAllWindows()
print("Tracking ended.")
