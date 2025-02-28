import pickle
import os

# Crear una estructura básica para tracks.pckl
tracks = []
# Busca los videos en tu carpeta pycrop
video_files = [f for f in os.listdir("Columbia_Dataset/columbia/pycrop") if f.endswith('.avi')]

for video_file in video_files:
    # Crea un track básico para cada video
    track = {
        'track': {'frame': list(range(100))},  # Asume 100 frames por video
        'proc_track': {
            's': [50] * 100,  # Tamaño del rostro
            'x': [112] * 100,  # Posición x del rostro
            'y': [112] * 100   # Posición y del rostro
        }
    }
    tracks.append(track)

# Guarda los tracks en el archivo
with open("Columbia_Dataset/columbia/pywork/tracks.pckl", "wb") as f:
    pickle.dump(tracks, f)

print(f"Archivo tracks.pckl generado con {len(tracks)} tracks")