# Crea un script de verificación (verify_videos.py)
import cv2
import os

def check_video(path):
    if not os.path.exists(path):
        print(f"Error: El archivo {path} no existe")
        return False
    
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: No se puede abrir el video {path}")
        return False
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
    
    cap.release()
    print(f"Video {path}: {frame_count} frames")
    return frame_count > 0

# Verifica los videos
video_paths = [
    "Columbia_Dataset/columbia/pycrop/test1.avi",
    "Columbia_Dataset/columbia/pycrop_body/test1.avi"
]

for path in video_paths:
    check_video(path)