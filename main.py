import os
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
from ultralytics import YOLO

def detect_people_in_video(video_path, model_path, frames_dir, labels_dir, frame_interval_sec=1):
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * frame_interval_sec)

    model = YOLO(model_path)

    frame_idx = 0
    saved_frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            results = model(frame)
            out_path = Path(frames_dir) / f"frame_{saved_frame_idx:04d}.jpg"
            label_path = Path(labels_dir) / f"frame_{saved_frame_idx:04d}.txt"

            cv2.imwrite(str(out_path), frame)

            with open(label_path, 'w') as f:
                for r in results:
                    for box in r.boxes:
                        if int(box.cls) == 0:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf)
                            xc = ((x1 + x2) / 2) / frame.shape[1]
                            yc = ((y1 + y2) / 2) / frame.shape[0]
                            bw = (x2 - x1) / frame.shape[1]
                            bh = (y2 - y1) / frame.shape[0]
                            f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imwrite(str(out_path), frame)
            saved_frame_idx += 1

        frame_idx += 1

    cap.release()
    print(f" Видео обработано. Сохранено {saved_frame_idx} кадров в {frames_dir}")

if __name__ == "__main__":
    detect_people_in_video(
        video_path="./video/IMG_1402.MOV", # путь к видео
        model_path="./runs/best.pt", # путь к дообученной модели
        frames_dir="./video_frames", # куда сохранять кадры
        labels_dir="./video_labels",
        frame_interval_sec = 5 # куда сохранять YOLO-аннотации
    )
