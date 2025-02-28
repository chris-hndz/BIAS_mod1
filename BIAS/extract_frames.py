import cv2
import os
import glob

# Directorios
base_dir = "Columbia_Dataset"
pycrop_dir = os.path.join(base_dir, "columbia", "pycrop")
pyframes_dir = os.path.join(base_dir, "columbia", "pyframes")

# Crear directorio si no existe
os.makedirs(pyframes_dir, exist_ok=True)

# Obtener lista de videos
video_files = glob.glob(os.path.join(pycrop_dir, "*.avi"))

for video_path in video_files:
    # Obtener nombre del video sin extensión
    video_name = os.path.basename(video_path).split('.')[0]
    print(f"Procesando video: {video_name}")
    
    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se puede abrir el video {video_path}")
        continue
    
    # Extraer frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir a escala de grises
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Redimensionar a 224x224
        gray_frame = cv2.resize(gray_frame, (224, 224))
        
        # Guardar frame
        frame_filename = f"{video_name}_{frame_count:04d}.jpg"
        frame_path = os.path.join(pyframes_dir, frame_filename)
        cv2.imwrite(frame_path, gray_frame)
        
        frame_count += 1
    
    cap.release()
    print(f"Se extrajeron {frame_count} frames del video {video_name}")

print("Proceso completado. Todas las imágenes han sido extraídas.")