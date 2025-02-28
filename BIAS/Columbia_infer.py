import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, python_speech_features

from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score

from bias import bias

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description = "Columbia ASD Evaluation")

parser.add_argument('--videoName',             type=str, default="columbia",   help='Demo video name')
parser.add_argument('--videoFolder',           type=str, default="colDataPath",  help='Path for inputs, tmps and outputs')
parser.add_argument('--pretrainModel',         type=str, default="BIAS_Columbia.model",   help='Path for the pretrained model')

parser.add_argument('--facedetScale',          type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
parser.add_argument('--minTrack',              type=int,   default=10,   help='Number of min frames for each shot')
parser.add_argument('--numFailedDet',          type=int,   default=10,   help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--minFaceSize',           type=int,   default=1,    help='Minimum face size in pixels')
parser.add_argument('--cropScale',             type=float, default=0.40, help='Scale bounding box')

parser.add_argument('--start',                 type=int, default=0,   help='The start time of the video')
parser.add_argument('--duration',              type=int, default=0,  help='The duration of the video, when set as 0, will extract the whole video')

parser.add_argument('--evalCol',               dest='evalCol', action='store_true', help='Evaluate on Columbia dataset')
parser.add_argument('--colSavePath',           type=str, default="colDataPath",  help='Path for inputs, tmps and outputs')

# TalkNet
parser.add_argument('--lr',           type=float, default=0.0001,help='Learning rate')
parser.add_argument('--lrDecay',      type=float, default=0.95,  help='Learning rate decay rate')
parser.add_argument('--maxEpoch',     type=int,   default=25,    help='Maximum number of epochs')
parser.add_argument('--testInterval', type=int,   default=1,     help='Test and save every [testInterval] epochs')
parser.add_argument('--batchSize',    type=int,   default=2000,  help='Dynamic batch size, default is 2500 frames, other batchsize (such as 1500) will not affect the performance')
parser.add_argument('--nDataLoaderThread', type=int, default=1,  help='Number of loader threads')

args = parser.parse_args()

def create_video(fileName, dir):
    video_path = os.path.join(dir, os.path.basename(fileName) + '.avi')
    if not os.path.exists(video_path):
        print(f"Error: El archivo de video {video_path} no existe")
        return numpy.zeros((0,))
        
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: No se puede abrir el video {video_path}")
        return numpy.zeros((0,))
        
    videoFeature = []
    frame_count = 0
    while video.isOpened():
        ret, frames = video.read()
        if ret == True:
            face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (224,224))
            face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
            videoFeature.append(face)
            frame_count += 1
        else:
            break
    video.release()
    
    if frame_count == 0:
        print(f"Advertencia: No se encontraron frames en el video {video_path}")
        return numpy.zeros((0,))
        
    videoFeature = numpy.array(videoFeature)
    
    # Verifica si el arreglo tiene la forma correcta
    if len(videoFeature.shape) < 3:
        print(f"Advertencia: videoFeature tiene forma {videoFeature.shape}")
        # Si solo hay un frame, añade una dimensión
        if len(videoFeature.shape) == 2:
            videoFeature = videoFeature.reshape(1, videoFeature.shape[0], videoFeature.shape[1])
        # Si es unidimensional (caso extremo), crea un arreglo vacío con la forma correcta
        elif len(videoFeature.shape) == 1:
            print("Error: No se pudieron cargar frames correctamente del video")
            videoFeature = numpy.zeros((0,))
    
    return videoFeature


def evaluate_network(files, args):
    # GPU: active speaker detection by pretrained model
    s = bias(**vars(args))
    ckpt = torch.load(args.pretrainModel, map_location='cuda')
    s.load_state_dict(ckpt)

    sys.stderr.write("Model %s loaded from previous state! \r\n"%args.pretrainModel)
    s.eval()
    allScores = []
    durationSet = {12,24,48,60} 
    
    for file in tqdm.tqdm(files, total = len(files)):
        fileName = os.path.splitext(file.split('/')[-1])[0] # Load audio and video
        try:
            # Intenta cargar el audio
            _, audio = wavfile.read(os.path.join(args.pycropPath, os.path.basename(fileName) + '.wav'))
            audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)

            # Intenta cargar los videos
            videoFeature = create_video(fileName, args.pycropPath)
            videoFeatureBody = create_video(fileName, args.pycropPathBody)
            
            # Verifica si los videos se cargaron correctamente
            if videoFeature.shape[0] == 0 or videoFeatureBody.shape[0] == 0:
                print(f"Error: No se pudieron cargar frames para el video {fileName}")
                # Crea tensores vacíos con la forma correcta
                if videoFeature.shape[0] == 0:
                    videoFeature = numpy.zeros((25, 112, 112))
                if videoFeatureBody.shape[0] == 0:
                    videoFeatureBody = numpy.zeros((25, 112, 112))
            
            # Asegúrate de que todos los tensores tengan al menos un frame
            min_frames = 25  # Número mínimo de frames
            if videoFeature.shape[0] < min_frames:
                # Repite el último frame hasta alcanzar min_frames
                last_frame = videoFeature[-1] if videoFeature.shape[0] > 0 else numpy.zeros((112, 112))
                padding = numpy.array([last_frame] * (min_frames - videoFeature.shape[0]))
                videoFeature = numpy.vstack((videoFeature, padding)) if videoFeature.shape[0] > 0 else padding
            
            if videoFeatureBody.shape[0] < min_frames:
                last_frame = videoFeatureBody[-1] if videoFeatureBody.shape[0] > 0 else numpy.zeros((112, 112))
                padding = numpy.array([last_frame] * (min_frames - videoFeatureBody.shape[0]))
                videoFeatureBody = numpy.vstack((videoFeatureBody, padding)) if videoFeatureBody.shape[0] > 0 else padding
            
            # Asegúrate de que el audio tenga suficientes frames
            min_audio_frames = min_frames * 4  # 4 frames de audio por cada frame de video
            if audioFeature.shape[0] < min_audio_frames:
                last_frame = audioFeature[-1] if audioFeature.shape[0] > 0 else numpy.zeros((13,))
                padding = numpy.array([last_frame] * (min_audio_frames - audioFeature.shape[0]))
                audioFeature = numpy.vstack((audioFeature, padding)) if audioFeature.shape[0] > 0 else padding
            
            length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
            length = max(length, 1)  # Asegúrate de que length sea al menos 1
            
            audioFeature = audioFeature[:int(round(length * 100)),:]
            videoFeature = videoFeature[:int(round(length * 25)),:,:]
            videoFeatureBody = videoFeatureBody[:int(round(length * 25)),:,:]
            
            allScore = [] # Evaluation use model
            for duration in durationSet:
                batchSize = int(math.ceil(length / duration))
                batchSize = max(batchSize, 1)  # Asegúrate de que batchSize sea al menos 1
                scores = []
                with torch.no_grad():
                    for i in range(batchSize):
                        # Asegúrate de que los índices estén dentro de los límites
                        audio_end = min((i+1) * duration * 100, audioFeature.shape[0])
                        video_end = min((i+1) * duration * 25, videoFeature.shape[0])
                        
                        inputA = torch.FloatTensor(audioFeature[i * duration * 100:audio_end,:]).unsqueeze(0).cuda()
                        inputV = torch.FloatTensor(videoFeature[i * duration * 25:video_end,:,:]).unsqueeze(0).cuda()
                        inputVB = torch.FloatTensor(videoFeatureBody[i * duration * 25:video_end,:,:]).unsqueeze(0).cuda()

                        # Asegúrate de que todos los tensores tengan la misma longitud en la dimensión 1
                        min_len = min(inputA.size(1) // 4, inputV.size(1), inputVB.size(1))
                        # En la función evaluate_network, después de procesar los tensores:
                        if min_len == 0:
                            print(f"Error: Longitud mínima es 0 para el batch {i}")
                            # Crea tensores con valores predeterminados
                            dummy_tensor = torch.zeros(1, 1, 128).cuda()  # Ajusta las dimensiones según sea necesario
                            audioEmbed = dummy_tensor.clone()
                            visualEmbed = dummy_tensor.clone()
                            visualEmbedBody = dummy_tensor.clone()
                            min_len = 1
                            continue  # Salta al siguiente batch
                            
                        inputA = inputA[:, :min_len*4, :]
                        inputV = inputV[:, :min_len, :, :]
                        inputVB = inputVB[:, :min_len, :, :]

                        audioEmbed = s.model.forward_audio_frontend(inputA)
                        visualEmbed = s.model.forward_visual_frontend(inputV)    
                        visualEmbedBody = s.model.forward_visual_frontend_body(inputVB)    

                        # Verifica que todos los embeddings tengan la misma longitud
                        min_embed_len = min(audioEmbed.size(1), visualEmbed.size(1), visualEmbedBody.size(1))
                        audioEmbed = audioEmbed[:, :min_embed_len, :]
                        visualEmbed = visualEmbed[:, :min_embed_len, :]
                        visualEmbedBody = visualEmbedBody[:, :min_embed_len, :]

                        # Self-Attention
                        audioEmbed = s.model.a_att(src = audioEmbed, tar = audioEmbed)
                        visualEmbed = s.model.v_att(src = visualEmbed, tar = visualEmbed)
                        visualEmbedBody = s.model.vb_att(src = visualEmbedBody, tar = visualEmbedBody)
                        
                        # Feature combination
                        comb_feat = torch.cat((audioEmbed, visualEmbed, visualEmbedBody), dim=2).cuda()
                        outsComb = s.se(comb_feat)
                        outsComb = s.model.comb_att(src = outsComb, tar = outsComb)

                        out = s.model.forward_comb_backend(outsComb)

                        score = s.lossComb.forward(out, labels = None)
                        scores.extend(score)
                allScore.append(scores)
            
            # Si no hay scores, crea un score predeterminado
            if len(allScore) == 0 or all(len(scores) == 0 for scores in allScore):
                print(f"No se pudieron generar scores para {fileName}, usando valor predeterminado")
                allScore = [[0.5]]  # Valor neutral
                
            allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
            allScores.append(allScore)
            
        except Exception as e:
            print(f"Error procesando {fileName}: {str(e)}")
            # Agrega un score predeterminado
            allScores.append(numpy.array([0.5]))  # Valor neutral
            
    return allScores
    

def bb_intersection_over_union(boxA, boxB, evalCol = False):
	# CPU: IOU Function to calculate overlap between two image
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	if evalCol == True:
		iou = interArea / float(boxAArea)
	else:
		iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def evaluate_col_ASD(tracks, scores, args):
    # Inicializa diccionarios para almacenar resultados
    faces = {}
    scores_dict = {}
    gt_dict = {}
    
    # Procesa cada track y sus puntuaciones
    for tidx, track in enumerate(tracks):
        # Verifica si hay puntuación para este track
        if tidx >= len(scores):
            print(f"Error: No hay puntuación para el track {tidx}")
            continue
            
        # Procesa cada frame en el track
        for fidx, frame in enumerate(track['track']['frame']):
            # Verifica si hay puntuación para este frame
            if fidx >= len(scores[tidx]):
                print(f"Error: No hay puntuación para el frame {fidx} en track {tidx}")
                continue
                
            s = scores[tidx][fidx]
            
            # Verifica que los índices existan en proc_track
            if fidx >= len(track['proc_track']['s']) or fidx >= len(track['proc_track']['x']) or fidx >= len(track['proc_track']['y']):
                print(f"Error: Índice {fidx} fuera de rango para track {tidx}")
                continue
                
            # Inicializa la lista de caras para este frame si no existe
            if frame not in faces:
                faces[frame] = []
            
            # Añade la información de la cara y su puntuación
            faces[frame].append({
                'track': tidx, 
                'score': float(s),
                's': track['proc_track']['s'][fidx], 
                'x': track['proc_track']['x'][fidx], 
                'y': track['proc_track']['y'][fidx]
            })
    
    # Procesa los resultados para cada frame
    for frame in faces:
        faces[frame].sort(key=lambda x: x['score'], reverse=True)
        scores_dict[frame] = [x['score'] for x in faces[frame]]
    
    # Guarda los resultados en archivos
    savePath = os.path.join(args.colSavePath, 'columbia')
    os.makedirs(savePath, exist_ok=True)
    
    # Guarda las puntuaciones
    with open(os.path.join(savePath, 'scores.pckl'), 'wb') as fil:
        pickle.dump(scores_dict, fil)
    
    # Guarda las caras detectadas
    with open(os.path.join(savePath, 'faces.pckl'), 'wb') as fil:
        pickle.dump(faces, fil)
    
    # Calcula y muestra métricas si hay datos de ground truth
    if os.path.isfile(os.path.join(args.videoFolder, 'columbia', 'pywork', 'gt.pckl')):
        with open(os.path.join(args.videoFolder, 'columbia', 'pywork', 'gt.pckl'), 'rb') as fil:
            gt_dict = pickle.load(fil)
        
        # Calcula métricas
        acc, f1 = calc_col_metrics(scores_dict, gt_dict)
        print('Accuracy: %.4f, F1: %.4f'%(acc, f1))
    
    return scores_dict, faces
    
    
# Main function
def main():
	
	args.videoName = 'columbia'
	args.savePath = os.path.join(args.videoFolder, args.videoName)

	# Initialization 
	args.pyaviPath = os.path.join(args.savePath, 'pyavi')
	args.pyframesPath = os.path.join(args.savePath, 'pyframes')
	args.pyworkPath = os.path.join(args.savePath, 'pywork')
	args.pycropPath = os.path.join(args.savePath, 'pycrop')
	args.pycropPathBody = os.path.join(args.savePath, 'pycrop_body')

	savePath = os.path.join(args.pyworkPath, 'tracks.pckl')
	fil = open(savePath, 'rb')
	vidTracks = pickle.load(fil)

	# Active Speaker Detection
	files = glob.glob("%s/*.avi"%args.pycropPath)
	files.sort()
	scores = evaluate_network(files, args)

	evaluate_col_ASD(vidTracks, scores, args) 

if __name__ == '__main__':
    main()
