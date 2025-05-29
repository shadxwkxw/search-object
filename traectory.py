import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

FOV = 90
FRAME_STEP = 2
FRAME_DIR = './video_frames'
INITIAL_HEIGHT = 1.0

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

trajectory = [np.array([0.0, 0.0])]
heights = [INITIAL_HEIGHT]
prev_kp, prev_des = None, None
person_position = None
prev_scale_factor = 1.0

frames = sorted([f for f in os.listdir(FRAME_DIR) if f.endswith('.jpg') or f.endswith('.png')])
frames = frames[::FRAME_STEP]

for i, fname in enumerate(frames):
    frame_path = os.path.join(FRAME_DIR, fname)
    frame_color = cv2.imread(frame_path)
    frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

    kp, des = orb.detectAndCompute(frame_gray, None)

    # вычисление траектории (X, Y)
    if prev_kp is not None and prev_des is not None and des is not None:
        matches = bf.match(prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > 8:
            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            distances_prev = np.linalg.norm(pts1[1:] - pts1[:-1], axis=2).mean()
            distances_curr = np.linalg.norm(pts2[1:] - pts2[:-1], axis=2).mean()
            if distances_prev > 0:
                scale_factor = distances_curr / distances_prev
            else:
                scale_factor = prev_scale_factor

            new_height = heights[-1] / scale_factor
            heights.append(new_height)
            prev_scale_factor = scale_factor

            M, inliers = cv2.estimateAffine2D(pts1, pts2)
            if M is not None:
                dx, dy = M[0, 2], M[1, 2]
                new_pos = trajectory[-1] + np.array([dx, dy])
                trajectory.append(new_pos)
            else:
                trajectory.append(trajectory[-1])
        else:
            trajectory.append(trajectory[-1])
            heights.append(heights[-1])
    else:
        trajectory.append(trajectory[-1])
        heights.append(heights[-1])

    prev_kp, prev_des = kp, des

    boxes, _ = hog.detectMultiScale(frame_gray)
    if len(boxes) > 0:
        (x, y, w, h) = boxes[0]
        px = x + w / 2
        fw = frame_gray.shape[1]
        offset = (px - fw / 2) / (fw / 2)
        angle = offset * (FOV / 2)

        drone_pos = trajectory[-1]
        direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])

        est_distance = heights[-1] * 10
        est_person_pos = drone_pos + direction * est_distance
        person_position = est_person_pos

traj_3d = np.array([[p[0], p[1], h] for p, h in zip(trajectory, heights)])

scale_xy = 0.1
scale_z = 1.0

traj_3d_meters = np.zeros_like(traj_3d)
traj_3d_meters[:, 0] = traj_3d[:, 0] * scale_xy
traj_3d_meters[:, 1] = traj_3d[:, 1] * scale_xy
traj_3d_meters[:, 2] = traj_3d[:, 2] * scale_z

if person_position is not None:
    person_position_meters = person_position * scale_xy
else:
    person_position_meters = None

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(np.min(traj_3d_meters[:, 0]) - 5, np.max(traj_3d_meters[:, 0]) + 5)
ax.set_ylim(np.min(traj_3d_meters[:, 1]) - 5, np.max(traj_3d_meters[:, 1]) + 5)
ax.set_zlim(0, np.max(traj_3d_meters[:, 2]) + 5)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m, altitude)')
ax.set_title('3D Drone Path and Person Location')

drone_line, = ax.plot([], [], [], 'b-', label='Drone Trajectory')
drone_point, = ax.plot([], [], [], 'bo', markersize=6, label='Drone')

if person_position_meters is not None:
    person_point, = ax.plot([person_position_meters[0]], [person_position_meters[1]],
                            [0],  # Человек всегда на земле
                            'ro', markersize=8, label='Person')
else:
    person_point = None

ax.legend()

def init():
    drone_line.set_data([], [])
    drone_line.set_3d_properties([])
    drone_point.set_data([], [])
    drone_point.set_3d_properties([])
    return [drone_line, drone_point] + ([person_point] if person_point is not None else [])

def update(frame):
    drone_line.set_data(traj_3d_meters[:frame+1, 0], traj_3d_meters[:frame+1, 1])
    drone_line.set_3d_properties(traj_3d_meters[:frame+1, 2])
    drone_point.set_data([traj_3d_meters[frame, 0]], [traj_3d_meters[frame, 1]])
    drone_point.set_3d_properties([traj_3d_meters[frame, 2]])
    return [drone_line, drone_point] + ([person_point] if person_point is not None else [])

ani = FuncAnimation(fig, update, frames=len(traj_3d_meters), init_func=init, interval=100, blit=True)

plt.show()