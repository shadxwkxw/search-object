import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from mpl_toolkits.mplot3d import Axes3D

FOV = 90
FRAME_STEP = 2
FRAME_DIR = './video_frames'

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
trajectory = [np.array([0.0, 0.0])]

frames = sorted([f for f in os.listdir(FRAME_DIR) if f.endswith('.jpg') or f.endswith('.png')])
frames = frames[::FRAME_STEP]

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

prev_kp, prev_des = None, None
prev_frame = None
person_position = None

for i, fname in enumerate(frames):
    frame_path = os.path.join(FRAME_DIR, fname)
    frame_color = cv2.imread(frame_path)
    frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

    kp, des = orb.detectAndCompute(frame_gray, None)

    if prev_kp is not None and prev_des is not None:
        matches = bf.match(prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > 8:
            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, inliers = cv2.estimateAffine2D(pts1, pts2)
            if M is not None:
                dx, dy = M[0, 2], M[1, 2]
                new_pos = trajectory[-1] + np.array([dx, dy])
                trajectory.append(new_pos)
            else:
                trajectory.append(trajectory[-1])
        else:
            trajectory.append(trajectory[-1])
    else:
        trajectory.append(trajectory[-1])

    prev_kp, prev_des = kp, des
    prev_frame = frame_gray

    if person_position is None:
        boxes, _ = hog.detectMultiScale(frame_gray)
        if len(boxes) > 0:
            (x, y, w, h) = boxes[0]
            px = x + w / 2
            fw = frame_gray.shape[1]
            offset = (px - fw / 2) / (fw / 2)
            angle = offset * (FOV / 2)

            drone_pos = trajectory[-1]
            direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
            est_person_pos = drone_pos + direction * 100
            person_position = est_person_pos

print("Траектория дрона:")
for i, pos in enumerate(trajectory):
    print(f"{i}: {pos}")

if person_position is not None:
    print(f"\nОценочные координаты человека: {person_position}")
else:
    print("\nЧеловек не обнаружен.")

output_data = {
    "trajectory": [pos.tolist() + [i * 5] for i, pos in enumerate(trajectory)],
    "person_position": person_position.tolist() + [len(trajectory) * 5] if person_position is not None else None
}

with open("drone_path.json", "w") as f:
    json.dump(output_data, f, indent=2)

traj = np.array(trajectory)
plt.figure()
plt.plot(traj[:, 0], traj[:, 1], label='Drone trajectory')
if person_position is not None:
    plt.plot(person_position[0], person_position[1], 'ro', label='Person')
plt.legend()
plt.title('Drone Path and Person Location')
plt.axis('equal')
plt.grid()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

traj_3d = np.array([[p[0], p[1], i * 5] for i, p in enumerate(trajectory)])
ax.plot(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2], label='Drone Trajectory')

if person_position is not None:
    ax.scatter(person_position[0], person_position[1], len(trajectory) * 5, c='r', marker='o', label='Person')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (altitude)')
ax.set_title('3D Drone Path and Person Location')
ax.legend()
plt.show()