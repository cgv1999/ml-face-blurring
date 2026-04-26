"""Размытие лиц на видео"""
import cv2
import numpy as np
from tqdm import tqdm

def apply_mosaic_effect(face_roi, pixel_size=10):
    if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
        return face_roi
    h, w, _ = face_roi.shape
    small_h = max(1, h // pixel_size)
    small_w = max(1, w // pixel_size)
    downscaled = cv2.resize(face_roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaiced_face = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_NEAREST)
    return mosaiced_face

def process_video(input_path, output_path, cascade_path, pixel_size=100):
    face_cascade = cv2.CascadeClassifier(cascade_path)
    video_capture = cv2.VideoCapture(input_path)
    
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    with tqdm(total=frame_count, desc="Обработка видео") as pbar:
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            pbar.update(1)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                mosaiced_face = apply_mosaic_effect(face_roi, pixel_size)
                frame[y:y+h, x:x+w] = mosaiced_face
            
            video_writer.write(frame)
    
    video_capture.release()
    video_writer.release()
    print(f"Видео обработано: {output_path}")

if __name__ == "__main__":
    process_video("data/raw/input.mp4", "data/processed/output.mp4", 
                  "models/haarcascade_frontalface_default.xml")
