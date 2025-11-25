import cv2
import numpy as np
import time
from collections import deque
from scipy.spatial.transform import Rotation as R

# ORB feature detector
orb = cv2.ORB_create(5000)
VIDEO_SRC = 0

# Camera intrinsic parameters (adjust for your camera)
FOCAL_LENGTH = 800  # pixels
CAMERA_MATRIX = np.array([
    [FOCAL_LENGTH, 0, 640],
    [0, FOCAL_LENGTH, 360],
    [0, 0, 1]
], dtype=np.float32)
DIST_COEFFS = np.zeros((4, 1))

# Motion filtering
MIN_MOTION_THRESHOLD = 0.5
MIN_ROTATION_THRESHOLD = 0.01
STATIC_FRAMES_THRESHOLD = 5

# Smoothing
SMOOTHING_ALPHA = 0.2
HISTORY_LENGTH = 10

# Feature refresh
FEATURE_REFRESH_INTERVAL = 15
MIN_FEATURE_COUNT = 200

# Optical flow parameters
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# Assumed object size for depth estimation (meters)
ASSUMED_OBJECT_SIZE = 0.3

# ----------------- Helper Functions -----------------
def estimate_depth_from_size(bbox_size, assumed_real_size=ASSUMED_OBJECT_SIZE):
    pixel_size = max(bbox_size[0], bbox_size[1])
    if pixel_size < 10:
        return 5.0
    depth = (assumed_real_size * FOCAL_LENGTH) / pixel_size
    return max(0.5, min(depth, 20.0))

def pixel_to_3d(pixel_coords, depth):
    fx, fy = CAMERA_MATRIX[0, 0], CAMERA_MATRIX[1, 1]
    cx, cy = CAMERA_MATRIX[0, 2], CAMERA_MATRIX[1, 2]
    x = (pixel_coords[0] - cx) * depth / fx
    y = (pixel_coords[1] - cy) * depth / fy
    z = depth
    return np.array([x, y, z])

def project_3d_to_2d(point_3d):
    if point_3d[2] < 0.1:
        return None
    fx, fy = CAMERA_MATRIX[0, 0], CAMERA_MATRIX[1, 1]
    cx, cy = CAMERA_MATRIX[0, 2], CAMERA_MATRIX[1, 2]
    x_2d = (point_3d[0] * fx / point_3d[2]) + cx
    y_2d = (point_3d[1] * fy / point_3d[2]) + cy
    return (int(x_2d), int(y_2d))

def estimate_3d_rotation(prev_pts_2d, next_pts_2d, prev_depth, next_depth):
    prev_pts_3d = np.array([pixel_to_3d(pt, prev_depth) for pt in prev_pts_2d])
    next_pts_3d = np.array([pixel_to_3d(pt, next_depth) for pt in next_pts_2d])

    prev_centroid = np.mean(prev_pts_3d, axis=0)
    next_centroid = np.mean(next_pts_3d, axis=0)
    prev_centered = prev_pts_3d - prev_centroid
    next_centered = next_pts_3d - next_centroid

    H = prev_centered.T @ next_centered
    U, S, Vt = np.linalg.svd(H)
    R_matrix = Vt.T @ U.T
    if np.linalg.det(R_matrix) < 0:
        Vt[-1, :] *= -1
        R_matrix = Vt.T @ U.T

    try:
        rot = R.from_matrix(R_matrix)
        rotvec = rot.as_rotvec()
        angle = np.linalg.norm(rotvec)
        axis = rotvec / angle if angle > 1e-6 else np.array([0,0,1])
        angle = angle if angle > 1e-6 else 0
    except:
        axis = np.array([0,0,1])
        angle = 0

    residuals = np.linalg.norm(next_centered - (R_matrix @ prev_centered.T).T, axis=1)
    quality = 1.0 / (1.0 + np.mean(residuals))
    return R_matrix, axis, angle, quality

def compute_3d_velocity(prev_pos_3d, curr_pos_3d, dt):
    return (curr_pos_3d - prev_pos_3d) / dt if dt > 0.001 else np.zeros(3)

def detect_new_features(gray, existing_pts=None, mask_radius=20):
    mask = None
    if existing_pts is not None and len(existing_pts) > 0:
        mask = np.ones(gray.shape, dtype=np.uint8)*255
        for pt in existing_pts:
            cv2.circle(mask, tuple(pt.astype(int)), mask_radius, 0, -1)
    kp = orb.detect(gray, mask)
    if kp and len(kp) > 0:
        new_pts = np.array([k.pt for k in kp], dtype=np.float32)
        if existing_pts is not None and len(existing_pts) > 0:
            return np.vstack([existing_pts, new_pts])
        return new_pts
    return existing_pts

# ----------------- Visualization Functions -----------------
def draw_3d_bounding_box(frame, center_3d, size, rotation_matrix, depth):
    half_size = size / 2
    corners_3d = np.array([
        [-half_size, -half_size, -half_size],
        [half_size, -half_size, -half_size],
        [half_size, half_size, -half_size],
        [-half_size, half_size, -half_size],
        [-half_size, -half_size, half_size],
        [half_size, -half_size, half_size],
        [half_size, half_size, half_size],
        [-half_size, half_size, half_size]
    ])
    rotated_corners = (rotation_matrix @ corners_3d.T).T + center_3d
    corners_2d = [project_3d_to_2d(c) for c in rotated_corners]
    if any(c is None for c in corners_2d): return
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for s,e in edges: cv2.line(frame, corners_2d[s], corners_2d[e], (0,255,255),2)
    for s,e in [(4,5),(5,6),(6,7),(7,4)]: cv2.line(frame, corners_2d[s], corners_2d[e], (0,255,0),3)

def draw_3d_axes(frame, center_3d, rotation_matrix, scale=0.1):
    axes_3d = np.array([[scale,0,0],[0,scale,0],[0,0,scale]])
    rotated_axes = (rotation_matrix @ axes_3d.T).T + center_3d
    origin_2d = project_3d_to_2d(center_3d)
    if origin_2d is None: return
    axes_2d = [project_3d_to_2d(a) for a in rotated_axes]
    for idx,color in enumerate([(0,0,255),(0,255,0),(255,0,0)]):
        cv2.line(frame, origin_2d, axes_2d[idx], color,4)
        cv2.circle(frame, axes_2d[idx],5,color,-1)

# ----------------- Initialize -----------------
cap = cv2.VideoCapture(VIDEO_SRC)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

prev_gray = None
prev_pts_all = None
tracked_object = None
prev_time = time.time()
frame_count = 0
last_feature_refresh = 0

print("Single-Object 3D Rotation Tracker\n" + "="*50)

# ----------------- Main Loop -----------------
while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)

    now = time.time()
    dt = max(now - prev_time, 0.001)
    prev_time = now
    frame_count += 1

    # ----------------- Initialize features -----------------
    if prev_gray is None or prev_pts_all is None:
        kp = orb.detect(gray, None)
        if kp and len(kp) >= 8:
            prev_pts_all = np.array([k.pt for k in kp], dtype=np.float32)
            print(f"Initialized with {len(prev_pts_all)} features")
        prev_gray = gray.copy()
        continue

    # ----------------- Periodic feature refresh -----------------
    if frame_count - last_feature_refresh > FEATURE_REFRESH_INTERVAL:
        if len(prev_pts_all) < MIN_FEATURE_COUNT:
            prev_pts_all = detect_new_features(gray, prev_pts_all, mask_radius=15)
            last_feature_refresh = frame_count

    # ----------------- Optical Flow -----------------
    next_pts_all, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts_all, None, **LK_PARAMS)
    if next_pts_all is None: 
        prev_gray = gray.copy()
        continue

    status = status.flatten()
    error = error.flatten()
    good_mask = (status==1) & (error<50)

    valid_prev_pts = prev_pts_all[good_mask]
    valid_next_pts = next_pts_all[good_mask]

    if len(valid_next_pts) < 8:
        prev_pts_all = detect_new_features(gray, None)
        prev_gray = gray.copy()
        continue

    # ----------------- Single Object Tracking -----------------
    centroid_2d = np.mean(valid_next_pts, axis=0)
    prev_centroid_2d = np.mean(valid_prev_pts, axis=0)
    x,y,w,h = cv2.boundingRect(valid_next_pts.astype(np.int32))
    motion_mag = np.mean(np.linalg.norm(valid_next_pts - valid_prev_pts, axis=1))
    if motion_mag < MIN_MOTION_THRESHOLD:
        prev_pts_all = valid_next_pts
        prev_gray = gray.copy()
        continue

    depth_curr = estimate_depth_from_size((w,h))
    object_size = max(w,h) * depth_curr / FOCAL_LENGTH
    position_3d = pixel_to_3d(centroid_2d, depth_curr)

    if tracked_object is None:
        tracked_object = {
            'prev_pts': valid_next_pts.copy(),
            'position_3d': position_3d,
            'accumulated_rotation': np.eye(3),
            'omega': 0,
            'axis': np.array([0,0,1]),
            'motion_history': deque([motion_mag], maxlen=HISTORY_LENGTH),
            'static_counter': 0,
            'depth': depth_curr
        }
    else:
        tracked_object['motion_history'].append(motion_mag)
        if np.mean(tracked_object['motion_history']) < MIN_MOTION_THRESHOLD:
            tracked_object['static_counter'] += 1
            if tracked_object['static_counter'] > STATIC_FRAMES_THRESHOLD:
                prev_pts_all = valid_next_pts
                prev_gray = gray.copy()
                continue
        else:
            tracked_object['static_counter'] = 0

        # ----------- Rotation estimation with fixed correspondence -----------
        min_len = min(len(tracked_object['prev_pts']), len(valid_next_pts))
        if min_len >= 8:
            R_matrix, axis, angle, quality = estimate_3d_rotation(
                tracked_object['prev_pts'][:min_len],
                valid_next_pts[:min_len],
                tracked_object['depth'],
                depth_curr
            )
            if quality > 0.5:
                R_accum = R.from_matrix(R_matrix) * R.from_matrix(tracked_object['accumulated_rotation'])
                tracked_object['accumulated_rotation'] = R_accum.as_matrix()
                tracked_object['omega'] = SMOOTHING_ALPHA*angle/dt + (1-SMOOTHING_ALPHA)*tracked_object['omega']
                tracked_object['axis'] = SMOOTHING_ALPHA*axis + (1-SMOOTHING_ALPHA)*tracked_object['axis']
                tracked_object['axis'] /= np.linalg.norm(tracked_object['axis'])
        else:
            R_matrix = np.eye(3)
            axis = np.array([0,0,1])
            angle = 0
            quality = 0

        velocity_3d = compute_3d_velocity(tracked_object['position_3d'], position_3d, dt)

        # Update tracked object
        tracked_object['prev_pts'] = valid_next_pts.copy()
        tracked_object['position_3d'] = position_3d
        tracked_object['depth'] = depth_curr

        # ----------------- Visualization -----------------
        draw_3d_bounding_box(frame, position_3d, object_size, tracked_object['accumulated_rotation'], depth_curr)
        draw_3d_axes(frame, position_3d, tracked_object['accumulated_rotation'], scale=object_size*0.5)

        cv2.putText(frame, f"Pos:[{position_3d[0]:.2f},{position_3d[1]:.2f},{position_3d[2]:.2f}]m", 
                    (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0),1)
        cv2.putText(frame, f"Vel:[{velocity_3d[0]:.2f},{velocity_3d[1]:.2f},{velocity_3d[2]:.2f}]m/s", 
                    (x, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)
        cv2.putText(frame, f"omega:{tracked_object['omega']*180/np.pi:.1f}deg/s", (x, y-20), 
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        cv2.putText(frame, f"Axis:[{tracked_object['axis'][0]:.2f},{tracked_object['axis'][1]:.2f},{tracked_object['axis'][2]:.2f}]", 
                    (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,0),1)

        for pt in valid_next_pts:
            cv2.circle(frame, tuple(pt.astype(int)), 3, (255,0,255), -1)

    info = f"Frame:{frame_count} | Features:{len(valid_next_pts)}"
    cv2.putText(frame, info, (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    # cv2.putText(frame, "3D TRACKING MODE", (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow("3D Rotation Tracker", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break
    elif key == ord('r'):
        tracked_object = None
        prev_pts_all = None
        prev_gray = None
        print("Tracking reset")
    elif key == ord('f'):
        prev_pts_all = detect_new_features(gray, None)
        last_feature_refresh = frame_count
        print(f"Feature refresh: {len(prev_pts_all)} features")

    prev_gray = gray.copy()
    prev_pts_all = valid_next_pts
cap.release()
cv2.destroyAllWindows()
print("3D Tracking session ended.")
