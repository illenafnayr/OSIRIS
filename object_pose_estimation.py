import cv2
import numpy as np
import time
from sklearn.cluster import DBSCAN
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

# Clustering parameters
MIN_FEATURES_PER_OBJECT = 12
DBSCAN_EPS = 40
DBSCAN_MIN_SAMPLES = 6
MAX_CENTROID_DIST = 150

# Motion filtering
MIN_MOTION_THRESHOLD = 0.5
MIN_ROTATION_THRESHOLD = 0.05
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

def cluster_keypoints(pts):
    """Cluster keypoints using DBSCAN"""
    if len(pts) < DBSCAN_MIN_SAMPLES:
        return {}
    
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

def estimate_depth_from_size(bbox_size, assumed_real_size=ASSUMED_OBJECT_SIZE):
    """Estimate depth using object size"""
    pixel_size = max(bbox_size[0], bbox_size[1])
    if pixel_size < 10:
        return 5.0
    depth = (assumed_real_size * FOCAL_LENGTH) / pixel_size
    return max(0.5, min(depth, 20.0))

def pixel_to_3d(pixel_coords, depth):
    """Convert 2D pixel coordinates to 3D camera coordinates"""
    fx = CAMERA_MATRIX[0, 0]
    fy = CAMERA_MATRIX[1, 1]
    cx = CAMERA_MATRIX[0, 2]
    cy = CAMERA_MATRIX[1, 2]
    
    x = (pixel_coords[0] - cx) * depth / fx
    y = (pixel_coords[1] - cy) * depth / fy
    z = depth
    
    return np.array([x, y, z])

def project_3d_to_2d(point_3d):
    """Project 3D point to 2D pixel coordinates"""
    if point_3d[2] < 0.1:
        return None
    
    fx = CAMERA_MATRIX[0, 0]
    fy = CAMERA_MATRIX[1, 1]
    cx = CAMERA_MATRIX[0, 2]
    cy = CAMERA_MATRIX[1, 2]
    
    x_2d = (point_3d[0] * fx / point_3d[2]) + cx
    y_2d = (point_3d[1] * fy / point_3d[2]) + cy
    
    return (int(x_2d), int(y_2d))

def estimate_3d_rotation(prev_pts_2d, next_pts_2d, prev_depth, next_depth):
    """Estimate 3D rotation using Kabsch algorithm"""
    if len(prev_pts_2d) < 8:
        return np.eye(3), np.array([0, 0, 1]), 0, 0
    
    # Convert to 3D points
    prev_pts_3d = np.array([pixel_to_3d(pt, prev_depth) for pt in prev_pts_2d])
    next_pts_3d = np.array([pixel_to_3d(pt, next_depth) for pt in next_pts_2d])
    
    # Center the point clouds
    prev_centroid_3d = np.mean(prev_pts_3d, axis=0)
    next_centroid_3d = np.mean(next_pts_3d, axis=0)
    
    prev_centered = prev_pts_3d - prev_centroid_3d
    next_centered = next_pts_3d - next_centroid_3d
    
    # Remove points too close to centroid
    distances = np.linalg.norm(prev_centered, axis=1)
    valid_mask = distances > 0.01
    
    if np.sum(valid_mask) < 5:
        return np.eye(3), np.array([0, 0, 1]), 0, 0
    
    prev_centered = prev_centered[valid_mask]
    next_centered = next_centered[valid_mask]
    
    # Compute rotation using Kabsch algorithm (SVD)
    H = prev_centered.T @ next_centered
    U, S, Vt = np.linalg.svd(H)
    rotation_matrix = Vt.T @ U.T
    
    # Ensure proper rotation matrix (det = 1)
    if np.linalg.det(rotation_matrix) < 0:
        Vt[-1, :] *= -1
        rotation_matrix = Vt.T @ U.T
    
    # Convert rotation matrix to axis-angle
    try:
        rot = R.from_matrix(rotation_matrix)
        rotvec = rot.as_rotvec()
        angle = np.linalg.norm(rotvec)
        if angle > 1e-6:
            axis = rotvec / angle
        else:
            axis = np.array([0, 0, 1])
            angle = 0
    except:
        axis = np.array([0, 0, 1])
        angle = 0
    
    # Calculate quality metric
    residuals = np.linalg.norm(next_centered - (rotation_matrix @ prev_centered.T).T, axis=1)
    quality = 1.0 / (1.0 + np.mean(residuals))
    
    return rotation_matrix, axis, angle, quality

def draw_3d_bounding_box(frame, center_3d, size, rotation_matrix, depth):
    """Draw a 3D bounding box around the object"""
    # Define corners of a cube centered at origin
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
    
    # Rotate corners
    rotated_corners = (rotation_matrix @ corners_3d.T).T
    
    # Translate to object center
    world_corners = rotated_corners + center_3d
    
    # Project to 2D
    corners_2d = []
    for corner in world_corners:
        pt_2d = project_3d_to_2d(corner)
        if pt_2d is None:
            return  # Skip if projection fails
        corners_2d.append(pt_2d)
    
    # Draw the 12 edges of the cube
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
    ]
    
    for start, end in edges:
        cv2.line(frame, corners_2d[start], corners_2d[end], (0, 255, 255), 2)
    
    # Highlight front face
    front_face = [(4, 5), (5, 6), (6, 7), (7, 4)]
    for start, end in front_face:
        cv2.line(frame, corners_2d[start], corners_2d[end], (0, 255, 0), 3)

def draw_3d_axes(frame, center_3d, rotation_matrix, scale=0.1):
    """Draw 3D coordinate axes"""
    # Define axis vectors in 3D (X=Red, Y=Green, Z=Blue)
    axes_3d = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, scale]
    ])
    
    # Rotate axes
    rotated_axes = (rotation_matrix @ axes_3d.T).T
    
    # Get world positions
    world_axes = rotated_axes + center_3d
    
    # Project to 2D
    origin_2d = project_3d_to_2d(center_3d)
    if origin_2d is None:
        return
    
    axes_2d = []
    for axis in world_axes:
        pt_2d = project_3d_to_2d(axis)
        if pt_2d is None:
            return
        axes_2d.append(pt_2d)
    
    # Draw axes with thick lines
    cv2.line(frame, origin_2d, axes_2d[0], (0, 0, 255), 4)    # X - Red
    cv2.line(frame, origin_2d, axes_2d[1], (0, 255, 0), 4)    # Y - Green
    cv2.line(frame, origin_2d, axes_2d[2], (255, 0, 0), 4)    # Z - Blue
    
    # Draw circles at axis endpoints
    cv2.circle(frame, axes_2d[0], 5, (0, 0, 255), -1)
    cv2.circle(frame, axes_2d[1], 5, (0, 255, 0), -1)
    cv2.circle(frame, axes_2d[2], 5, (255, 0, 0), -1)

def compute_3d_velocity(prev_pos_3d, curr_pos_3d, dt):
    """Compute 3D velocity vector"""
    if dt < 0.001:
        return np.zeros(3)
    return (curr_pos_3d - prev_pos_3d) / dt

def match_cluster_to_tracked(cluster_centroid, cluster_motion, tracked_objects):
    """Match detected cluster to existing tracked object"""
    best_match = None
    best_dist = MAX_CENTROID_DIST
    
    for obj_id, obj in tracked_objects.items():
        predicted_pos_2d = obj['centroid_2d'] + obj['velocity_2d'] * 0.033
        dist = np.linalg.norm(cluster_centroid - predicted_pos_2d)
        
        if dist < best_dist:
            best_dist = dist
            best_match = obj_id
    
    return best_match

def is_object_static(motion_history):
    """Determine if object is static"""
    if len(motion_history) < 3:
        return False
    return np.mean(motion_history) < MIN_MOTION_THRESHOLD

def detect_new_features(gray, existing_pts=None, mask_radius=20):
    """Detect new features"""
    mask = None
    if existing_pts is not None and len(existing_pts) > 0:
        mask = np.ones(gray.shape, dtype=np.uint8) * 255
        for pt in existing_pts:
            cv2.circle(mask, tuple(pt.astype(int)), mask_radius, 0, -1)
    
    kp = orb.detect(gray, mask)
    if kp and len(kp) > 0:
        new_pts = np.array([k.pt for k in kp], dtype=np.float32)
        if existing_pts is not None and len(existing_pts) > 0:
            return np.vstack([existing_pts, new_pts])
        return new_pts
    return existing_pts


# Initialize
cap = cv2.VideoCapture(VIDEO_SRC)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

prev_gray = None
prev_pts_all = None
tracked_objects = {}
object_id_counter = 0
prev_time = time.time()
frame_count = 0
last_feature_refresh = 0

print("3D Rotation Tracker for Space Debris")
print("=" * 50)
print("Tracking objects in 3D space with full rotation estimation")
print("Red/Green/Blue axes = X/Y/Z orientation")
print("Cyan box = 3D bounding box (green face = front)")
print("Press ESC to exit, 'r' to reset, 'f' to refresh features")
print()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame, exiting...")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    now = time.time()
    dt = now - prev_time
    if dt < 0.001:
        dt = 0.033
    prev_time = now
    frame_count += 1
    
    # Initialize or reinitialize
    if prev_gray is None or prev_pts_all is None:
        kp = orb.detect(gray, None)
        if kp and len(kp) >= MIN_FEATURES_PER_OBJECT:
            prev_pts_all = np.array([k.pt for k in kp], dtype=np.float32)
            print(f"Initialized with {len(prev_pts_all)} features")
        prev_gray = gray.copy()
        continue
    
    # Periodic feature refresh
    if frame_count - last_feature_refresh > FEATURE_REFRESH_INTERVAL:
        if len(prev_pts_all) < MIN_FEATURE_COUNT:
            prev_pts_all = detect_new_features(gray, prev_pts_all, mask_radius=15)
            last_feature_refresh = frame_count
    
    # Optical flow tracking
    next_pts_all, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, prev_pts_all, None, **LK_PARAMS
    )
    
    if next_pts_all is None or len(next_pts_all) == 0:
        kp = orb.detect(gray, None)
        if kp and len(kp) > 0:
            prev_pts_all = np.array([k.pt for k in kp], dtype=np.float32)
        prev_gray = gray.copy()
        continue
    
    # Filter good points
    status = status.flatten()
    error = error.flatten()
    good_mask = (status == 1) & (error < 50)
    
    valid_prev_pts = prev_pts_all[good_mask]
    valid_next_pts = next_pts_all[good_mask]
    
    # Emergency feature refresh
    if len(valid_next_pts) < MIN_FEATURE_COUNT:
        new_features = detect_new_features(gray, valid_next_pts, mask_radius=15)
        if new_features is not None and len(new_features) > len(valid_next_pts):
            prev_pts_all = new_features
            prev_gray = gray.copy()
            last_feature_refresh = frame_count
            continue
    
    # Cluster to identify objects
    clusters = cluster_keypoints(valid_next_pts)
    current_objects = {}
    
    for cluster_lbl, indices in clusters.items():
        cluster_prev_pts = valid_prev_pts[indices]
        cluster_next_pts = valid_next_pts[indices]
        
        # 2D properties
        centroid_2d = np.mean(cluster_next_pts, axis=0)
        prev_centroid_2d = np.mean(cluster_prev_pts, axis=0)
        
        x, y, w, h = cv2.boundingRect(cluster_next_pts.astype(np.int32))
        motion_mag = np.mean(np.linalg.norm(cluster_next_pts - cluster_prev_pts, axis=1))
        
        # Skip static objects
        if motion_mag < MIN_MOTION_THRESHOLD:
            continue
        
        # Estimate depth and size
        depth_curr = estimate_depth_from_size((w, h))
        object_size = max(w, h) * depth_curr / FOCAL_LENGTH
        
        # Get 3D position
        position_3d = pixel_to_3d(centroid_2d, depth_curr)
        
        # Match to tracked object
        matched_id = match_cluster_to_tracked(centroid_2d, motion_mag, tracked_objects)
        
        if matched_id is None:
            object_id_counter += 1
            obj_id = object_id_counter
            motion_history = deque([motion_mag], maxlen=HISTORY_LENGTH)
            accumulated_rotation = np.eye(3)
            static_counter = 0
            prev_position_3d = position_3d
            depth_prev = depth_curr
        else:
            obj_id = matched_id
            motion_history = tracked_objects[obj_id]['motion_history']
            motion_history.append(motion_mag)
            accumulated_rotation = tracked_objects[obj_id]['accumulated_rotation']
            static_counter = tracked_objects[obj_id]['static_counter']
            prev_position_3d = tracked_objects[obj_id]['position_3d']
            depth_prev = tracked_objects[obj_id]['depth']
        
        # Check if static
        if is_object_static(motion_history):
            static_counter += 1
            if static_counter > STATIC_FRAMES_THRESHOLD:
                continue
        else:
            static_counter = 0
        
        # Estimate 3D rotation
        if len(cluster_prev_pts) >= 8:
            rotation_matrix, axis, angle, quality = estimate_3d_rotation(
                cluster_prev_pts, cluster_next_pts, depth_prev, depth_curr
            )
            # Accumulate rotation
            accumulated_rotation = rotation_matrix @ accumulated_rotation
        else:
            rotation_matrix = np.eye(3)
            axis = np.array([0, 0, 1])
            angle = 0
            quality = 0
        
        # Calculate angular velocity (rad/s)
        omega = angle / dt if dt > 0 else 0
        
        # Skip if minimal rotation
        if abs(omega) < MIN_ROTATION_THRESHOLD and motion_mag < MIN_MOTION_THRESHOLD * 2:
            continue
        
        # Calculate 3D velocity
        velocity_3d = compute_3d_velocity(prev_position_3d, position_3d, dt)
        velocity_2d = (centroid_2d - prev_centroid_2d) / dt
        
        # Store object data
        current_objects[obj_id] = {
            'indices': indices,
            'centroid_2d': centroid_2d,
            'position_3d': position_3d,
            'depth': depth_curr,
            'velocity_2d': velocity_2d,
            'velocity_3d': velocity_3d,
            'rotation_matrix': rotation_matrix,
            'accumulated_rotation': accumulated_rotation,
            'axis': axis,
            'omega': omega,
            'quality': quality,
            'motion_history': motion_history,
            'static_counter': static_counter,
            'bbox': (x, y, w, h),
            'object_size': object_size
        }
        
        # === VISUALIZATION ===
        
        # Draw 3D bounding box
        draw_3d_bounding_box(frame, position_3d, object_size, accumulated_rotation, depth_curr)
        
        # Draw 3D coordinate axes
        draw_3d_axes(frame, position_3d, accumulated_rotation, scale=object_size * 0.5)
        
        # Display information
        cv2.putText(frame, f"ID:{obj_id}", (x, y - 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Pos:[{position_3d[0]:.2f},{position_3d[1]:.2f},{position_3d[2]:.2f}]m", 
                    (x, y - 63), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        cv2.putText(frame, f"Vel:[{velocity_3d[0]:.2f},{velocity_3d[1]:.2f},{velocity_3d[2]:.2f}]m/s", 
                    (x, y - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(frame, f"omega:{omega*180/np.pi:.1f}deg/s", (x, y - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Axis:[{axis[0]:.2f},{axis[1]:.2f},{axis[2]:.2f}]", 
                    (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)
        cv2.putText(frame, f"Depth:{depth_curr:.2f}m Q:{quality:.2f}", 
                    (x, y - 0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        
        # Draw motion trail in 2D
        if matched_id and matched_id in tracked_objects:
            old_centroid = tracked_objects[matched_id]['centroid_2d']
            cv2.arrowedLine(frame, tuple(old_centroid.astype(int)), 
                          tuple(centroid_2d.astype(int)), (0, 255, 0), 2, tipLength=0.3)
        
        # Draw feature points
        for pt in cluster_next_pts:
            cv2.circle(frame, tuple(pt.astype(int)), 3, (255, 0, 255), -1)
    
    # Display global info
    info = f"Frame:{frame_count} | Objects:{len(current_objects)} | Features:{len(valid_next_pts)}"
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "3D TRACKING MODE", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow("3D Rotation Tracker", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('r'):
        tracked_objects = {}
        object_id_counter = 0
        prev_pts_all = None
        prev_gray = None
        print("Tracking reset")
    elif key == ord('f'):
        prev_pts_all = detect_new_features(gray, None)
        last_feature_refresh = frame_count
        print(f"Feature refresh: {len(prev_pts_all)} features")
    
    # Prepare next frame
    prev_gray = gray.copy()
    prev_pts_all = valid_next_pts
    tracked_objects = current_objects

cap.release()
cv2.destroyAllWindows()
print("\n3D Tracking session ended.")