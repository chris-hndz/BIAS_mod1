import pickle
import os
import cv2

# Directorio base
base_dir = "Columbia_Dataset/columbia"
pycrop_dir = os.path.join(base_dir, "pycrop")
pywork_dir = os.path.join(base_dir, "pywork")

# Asegúrate de que el directorio pywork exista
os.makedirs(pywork_dir, exist_ok=True)

tracks = []
video_files = [f for f in os.listdir(pycrop_dir) if f.endswith('.avi')]

for video_file in video_files:
    video_path = os.path.join(pycrop_dir, video_file)
    
    # Obtener el número real de frames
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if frame_count > 0:
        # Crea un track con el número correcto de frames
        track = {
            'track': {'frame': list(range(frame_count))},
            'proc_track': {
                's': [50] * frame_count,  # Tamaño del rostro
                'x': [112] * frame_count,  # Posición x del rostro
                'y': [112] * frame_count   # Posición y del rostro
            }
        }
        tracks.append(track)
        print(f"Procesado {video_file} con {frame_count} frames")
    else:
        print(f"Error: No se pudieron leer frames de {video_file}")

# Guarda los tracks en el archivo
with open(os.path.join(pywork_dir, "tracks.pckl"), "wb") as f:
    pickle.dump(tracks, f)

print(f"Archivo tracks.pckl generado con {len(tracks)} tracks")