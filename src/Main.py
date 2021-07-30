import cv2
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import sys
from Camera import Camera
from F_formationDetectionStrategies import naiveStrategy, oCentersClusteringStrategy, visualFieldsClusteringStrategy
from Tools import getParameters, getValue, openCSV, isNaN, timeToFrame, getAlpha, getBeta, getGamma
from Tools import getManualAnnotations, getInfoForF_formationDetection
from Tools import videoFormatToFrameFormatForF_formation, f_formationCorrection, show_f_formation

# Trackbars callback functions
def change_keypointsAndOrientations_shown(val):
	global keypointsAndOrientations_shown
	keypointsAndOrientations_shown = val * True

def change_boundingBox_shown(val):
	global boundingBox_shown
	boundingBox_shown = val * True

def change_labels_shown(val):
	global labels_shown
	labels_shown = val * True

def change_humanDetection_shown(val):
	global humanDetection_shown
	humanDetection_shown = val * True
	if not humanDetection_shown:
		cv2.destroyWindow('Human Detector')

def change_f_formationGroundTruth_shown(val):
	global f_formationGroundTruth_shown
	f_formationGroundTruth_shown = val * True
	if not f_formationGroundTruth_shown:
		cv2.destroyWindow('F-formation Detection : Ground Truth')

def change_naiveStrategy_shown(val):
	global naiveStrategy_shown
	naiveStrategy_shown = val * True
	if not naiveStrategy_shown:
		cv2.destroyWindow('F-formation Detection : Naive Strategy')

def change_oCentersClusteringStrategy_shown(val):
	global oCentersClusteringStrategy_shown, oCentersClusteringStrategy_verbose
	oCentersClusteringStrategy_shown = val * True
	if not oCentersClusteringStrategy_shown:
		cv2.destroyWindow('F-formation Detection : O-Centers Clustering Strategy')
		if oCentersClusteringStrategy_verbose:
			cv2.destroyWindow('O-Centers Clustering Strategy : O-Centers')
			cv2.destroyWindow('O-Centers Clustering Strategy : Clustering')
			cv2.destroyWindow('O-Centers Clustering Strategy : F-formations')

def change_visualFieldsClusteringStrategy_shown(val):
	global visualFieldsClusteringStrategy_shown, visualFieldsClusteringStrategy_verbose
	visualFieldsClusteringStrategy_shown = val * True
	if not visualFieldsClusteringStrategy_shown:
		cv2.destroyWindow('F-formation Detection : Visual Fields Clustering Strategy')
		if visualFieldsClusteringStrategy_verbose:
			cv2.destroyWindow('Visual Fields Clustering Strategy : Visual Fields')
			cv2.destroyWindow('Visual Fields Clustering Strategy : Clustering')
			cv2.destroyWindow('Visual Fields Clustering Strategy : F-formations')

############################
#          SCRIPT          #
############################

# Recupere les parametres necessaire a la creation de la camera, le fonctionnement du mode evaluation et le fonctionnement de la recherche de parametres
PARAMS = getParameters()

# Les parametres 'day' et 'cam' peuvent etre passes en arguments au script
if len(sys.argv) == 3:
	argv1 = eval(sys.argv[1])
	argv2 = eval(sys.argv[2])
	if type(argv1) is int and type(argv2) is int:
		PARAMS['day'] = argv1
		PARAMS['cam'] = argv2

# Creation de la camera
datasetPath = getValue(PARAMS, 'datasetPath', './MatchNMingle/Mingle/')
savePath = getValue(PARAMS, 'savePath', './res/')
day = getValue(PARAMS, 'day', 2)
cam = getValue(PARAMS, 'cam', 1)
src = datasetPath+'videos/downsample/30min_day'+str(day)+'_cam'+str(cam)+'_20fps_960x540.MP4'
FPS = np.int64(src.split("fps")[0][-2:])
camera = Camera(day, cam, src, FPS, 'Sequence')

# Initialisation du mode evaluation (EM)
emActivated = getValue(PARAMS, 'emActivated', False)
camera.initEM(savePath, emActivated=emActivated)

# Initialisation de la recherche de parametres (PS)
frameStart  = getValue(PARAMS, 'frameStart', 0)
frameEnd    = getValue(PARAMS, 'frameEnd', 5000)
alphaMin    = getValue(PARAMS, 'alphaMin', 0)
alphaMax    = getValue(PARAMS, 'alphaMax', 120)
alphaStep   = getValue(PARAMS, 'alphaStep', 30)
betaMin     = getValue(PARAMS, 'betaMin', 0.05)
betaMax     = getValue(PARAMS, 'betaMax', 0.25)
betaStep    = getValue(PARAMS, 'betaStep', 0.05)
gammaMin    = getValue(PARAMS, 'gammaMin', 0)
gammaMax    = getValue(PARAMS, 'gammaMax', 1)
gammaStep   = getValue(PARAMS, 'gammaStep', 0.25)
psActivated = getValue(PARAMS, 'psActivated', False)
camera.initPS(savePath, frameStart, frameEnd, alphaMin, alphaMax, alphaStep, betaMin, betaMax, betaStep, gammaMin, gammaMax, gammaStep, psActivated=psActivated)

# Initialisation du detecteur d'humains (HD)
camera.initHD('./src/socialCuesDetection/')

# Initialisation de la memoire des participants (PM)
camera.initPM(savePath, 1)

# Nom des sliders qui apparaitront sous la video
keypointsAndOrientations_trackbar       = 'Keypoints and Orientations'
boundingBox_trackbar                    = 'Bounding Box'
labels_trackbar                         = 'Labels'
humanDetection_trackbar                 = 'Social Cues Detection : Human Detection'
f_formationGroundTruth_trackbar         = 'F-formation Detection : Ground Truth'
naiveStrategy_trackbar                  = 'F-formation Detection : Naive Strategy'
oCentersClusteringStrategy_trackbar     = 'F-formation Detection : O-Centers Clustering Strategy'
visualFieldsClusteringStrategy_trackbar = 'F-formation Detection : Visual Fields Clustering Strategy'

# Valeurs par defaut des sliders
keypointsAndOrientations_shown       = False
boundingBox_shown                    = True
labels_shown                         = True
humanDetection_shown                 = False
f_formationGroundTruth_shown         = False
naiveStrategy_shown                  = False
oCentersClusteringStrategy_shown     = False
visualFieldsClusteringStrategy_shown = camera.psActivated

# Valeurs d'affichage des etapes intermediaires de l'algorithme 2A et 3 pour la detection des F-formations
oCentersClusteringStrategy_verbose = True
visualFieldsClusteringStrategy_verbose = not camera.psActivated

# Les sliders sont ajoutes a la camera
camera.add_cbox(keypointsAndOrientations_trackbar, change_keypointsAndOrientations_shown, keypointsAndOrientations_shown)
camera.add_cbox(boundingBox_trackbar, change_boundingBox_shown, boundingBox_shown)
camera.add_cbox(labels_trackbar, change_labels_shown, labels_shown)
camera.add_cbox(humanDetection_trackbar, change_humanDetection_shown, humanDetection_shown)
camera.add_cbox(f_formationGroundTruth_trackbar, change_f_formationGroundTruth_shown, f_formationGroundTruth_shown)
camera.add_cbox(naiveStrategy_trackbar, change_naiveStrategy_shown, naiveStrategy_shown)
camera.add_cbox(oCentersClusteringStrategy_trackbar, change_oCentersClusteringStrategy_shown, oCentersClusteringStrategy_shown)
camera.add_cbox(visualFieldsClusteringStrategy_trackbar, change_visualFieldsClusteringStrategy_shown, visualFieldsClusteringStrategy_shown)

# PARTICIPANTS
filename = datasetPath+'manual_annotations/PARTICIPANTS.csv'
participants = openCSV(filename).astype(np.int64)
tmp = []
for i in range(participants.shape[0]):
	if participants[i,0] == day:
		# Global ID / Day ID
		tmp.append([i+1, participants[i,1]])
participants = np.array(tmp).astype(np.int64)
print("PARTICIPANTS array shape =", participants.shape)

# DATA
filename = datasetPath+'manual_annotations/DATA.csv'
data = openCSV(filename).astype(np.int64)
print("DATA array shape =", data.shape)

# LABELS
filename = datasetPath+'manual_annotations/ LABELS.csv'
labels = openCSV(filename).astype(np.int64)
print("LABELS array shape =", labels.shape)

# KEYPOINTS & ORIENTATIONS
keypointsAndOrientations = [None]*(participants.shape[0])
folderName = datasetPath+'manual_annotations/AdditionalAnnotations/keypointsAndOrientations/day'+str(day)
for filename in os.listdir(folderName):
	personId = int(filename.split(".")[0].split("n")[1])
	keypointsAndOrientations[personId-1] = openCSV(folderName+'/'+filename)[1:,:].tolist()
keypointsAndOrientations = np.array(keypointsAndOrientations)
tmp = np.full((data.shape[0], data.shape[1]//7*keypointsAndOrientations.shape[2]), 'NaN')
for i in range(keypointsAndOrientations.shape[0]):
	globalId = participants[i,0]
	for j in range(keypointsAndOrientations.shape[1]):
		frame_start = np.int64(np.float64(keypointsAndOrientations[i,j,0]))
		if j+1 < keypointsAndOrientations.shape[1]:
			frame_end = np.int64(np.float64(keypointsAndOrientations[i,j+1,0]))
		else:
			frame_end = tmp.shape[0]
		tmp[frame_start:frame_end, 13*(globalId-1):13*globalId] = keypointsAndOrientations[i,j]
keypointsAndOrientations = tmp
print("Keypoints & Orientations array shape =", keypointsAndOrientations.shape)

# F-FORMATION GROUND TRUTH
filename = datasetPath+'manual_annotations/F-formationsGT/Day'+str(day)+'.csv'
f_formationGroundTruth = openCSV(filename)[1:,:]
tmp1 = []
for i in range(f_formationGroundTruth.shape[0]):
	participants_f_formation = f_formationGroundTruth[i,0].split(" ")
	tmp2 = []
	for id in participants_f_formation:
		try:
			tmp2.append(np.int64(id))
		except:
			continue
	participants_f_formation = tmp2
	frame_start = timeToFrame(f_formationGroundTruth[i,1], camera.FPS)
	frame_end = timeToFrame(f_formationGroundTruth[i,2], camera.FPS)
	tmp1.append([participants_f_formation,frame_start,frame_end])
f_formationGroundTruth = tmp1
f_formationGroundTruth = videoFormatToFrameFormatForF_formation(f_formationGroundTruth, camera.numberOfFrame)
print("F-formations Ground Truth list size =", len(f_formationGroundTruth))

while True:
	status = camera.update()
	# S'il n'y a plus de frame a capture, si la capture n'est plus ouverture ou si le boutton 'q' (quitter) a ete presse par l'utilsateur, alors le programme se ferme
	if not status:
		break
	# Annotations principales de la frame actuelle
	manual_annotations = getManualAnnotations(camera.frameId, participants, keypointsAndOrientations, data, labels)
	# Informations necessaires a la bonne detection des F-formations
	participantsID, positions, orientations = getInfoForF_formationDetection(manual_annotations, participants, cam)
	# Correction apportee a la F-formation verite terrain de la frame actuelle
	f_formation_true = f_formationCorrection(f_formationGroundTruth[camera.frameId] if camera.frameId < len(f_formationGroundTruth) else [], participantsID)
	# Initialisation des F-formations predites
	f_formation_pred = None
	# Affichage des annotations
	for i, anno in enumerate(manual_annotations):
		if anno[0] is not None and anno[1] is not None and anno[2] is not None:
			if not isNaN(anno[0][1]) and int(anno[0][1]):
				if not isNaN(anno[0][2]) and int(anno[0][2]) == cam:
					if keypointsAndOrientations_shown:
						if 'NaN' not in anno[0]:
							sh1X = np.int64(np.float64(anno[0][3]))
							sh1Y = np.int64(np.float64(anno[0][4]))
							headX = np.int64(np.float64(anno[0][5]))
							headY = np.int64(np.float64(anno[0][6]))
							sh2X = np.int64(np.float64(anno[0][7]))
							sh2Y = np.int64(np.float64(anno[0][8]))
							gazeX = np.int64(np.float64(anno[0][9]))
							gazeY = np.int64(np.float64(anno[0][10]))
							camera.frame_copy = cv2.line(camera.frame_copy, (sh1X, sh1Y), (sh2X, sh2Y), (0, 255, 0), 2)
							camera.frame_copy = cv2.line(camera.frame_copy, (sh1X, sh1Y), (headX, headY), (0, 255, 0), 2)
							camera.frame_copy = cv2.line(camera.frame_copy, (headX, headY), (sh2X, sh2Y), (0, 255, 0), 2)
							camera.frame_copy = cv2.line(camera.frame_copy, (headX, headY), (gazeX, gazeY), (255, 0, 0), 2)
					if boundingBox_shown:
						x1 = anno[1][1]
						y1 = anno[1][2]
						x2 = anno[1][3]
						y2 = anno[1][4]
						camera.frame_copy = cv2.rectangle(camera.frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
						camera.frame_copy = cv2.putText(camera.frame_copy, 'Global ID : '+str(participants[i,0]), (x1, y1+15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0, 0, 255), 1)
						camera.frame_copy = cv2.putText(camera.frame_copy, 'Day ID : '+str(participants[i,1]), (x1, y1+30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0, 0, 255), 1)
					if labels_shown:
						cpt = 1 if not boundingBox_shown else 3
						x1 = anno[1][1]
						y1 = anno[1][2]
						if anno[2][0]:
							camera.frame_copy = cv2.putText(camera.frame_copy, 'Walking', (x1, y1+15*cpt), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (255, 0, 0), 1)
							cpt += 1
						if anno[2][1]:
							camera.frame_copy = cv2.putText(camera.frame_copy, 'Stepping', (x1, y1+15*cpt), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (255, 0, 0), 1)
							cpt += 1
						if anno[2][2]:
							camera.frame_copy = cv2.putText(camera.frame_copy, 'Drinking', (x1, y1+15*cpt), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (255, 0, 0), 1)
							cpt += 1
						if anno[2][3]:
							camera.frame_copy = cv2.putText(camera.frame_copy, 'Speaking', (x1, y1+15*cpt), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (255, 0, 0), 1)
							cpt += 1
						if anno[2][4]:
							camera.frame_copy = cv2.putText(camera.frame_copy, 'Hand Gesture', (x1, y1+15*cpt), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (255, 0, 0), 1)
							cpt += 1
						if anno[2][5]:
							camera.frame_copy = cv2.putText(camera.frame_copy, 'Head Gesture', (x1, y1+15*cpt), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (255, 0, 0), 1)
							cpt += 1
						if anno[2][6]:
							camera.frame_copy = cv2.putText(camera.frame_copy, 'Laugh', (x1, y1+15*cpt), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (255, 0, 0), 1)
							cpt += 1
						if anno[2][7]:
							camera.frame_copy = cv2.putText(camera.frame_copy, 'Hair Touching', (x1, y1+15*cpt), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (255, 0, 0), 1)
							cpt += 1
						if anno[2][8]:
							camera.frame_copy = cv2.putText(camera.frame_copy, 'Action Occluded', (x1, y1+15*cpt), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (255, 0, 0), 1)
							cpt += 1
	# Mise a jour de la sauvegarde des boites englobantes de la frame actuelle
	camera.hd.positiveSamplesSavingUpdate(manual_annotations)
	# Mise a jour du detecteur d'humains
	camera.hd.update(humanDetection_shown)
	# Affichage des F-formations verite terrain
	if f_formationGroundTruth_shown:
		f_formationGroundTruth_disp = show_f_formation(f_formation_true, participantsID, positions, camera.frame, (0, 255, 255))
		cv2.imshow('F-formation Detection : Ground Truth', f_formationGroundTruth_disp)
	# Detection et affichage des F-formations par les strategies de detections activees
	strategiesActivated = []
	# Algorithme 1 : Detection des F-formations par proximite spatiale des individus
	if naiveStrategy_shown:
		f_formation_pred = naiveStrategy(camera.frame, participantsID, positions)
		naiveStrategy_disp = show_f_formation(f_formation_pred, participantsID, positions, camera.frame, (255, 0, 255))
		cv2.imshow('F-formation Detection : Naive Strategy', naiveStrategy_disp)
		strategiesActivated.append(naiveStrategy.__name__)
	# Algorithme 2A : Detection des F-formations par clustering des centres des O-spaces votes par les individus
	if oCentersClusteringStrategy_shown:
		f_formation_pred = oCentersClusteringStrategy(camera.frame, participantsID, positions, orientations, verbose=oCentersClusteringStrategy_verbose)
		oCentersClusteringStrategy_disp = show_f_formation(f_formation_pred, participantsID, positions, camera.frame, (0, 0, 255))
		cv2.imshow('F-formation Detection : O-Centers Clustering Strategy', oCentersClusteringStrategy_disp)
		strategiesActivated.append(oCentersClusteringStrategy.__name__)
	# Algorithme 3 : Detection des F-formations par clustering des champs visuels des individus
	if visualFieldsClusteringStrategy_shown:
		f_formation_pred = visualFieldsClusteringStrategy(camera.frame, participantsID, positions, orientations, alpha=getAlpha(camera), beta=getBeta(camera), gamma=getGamma(camera), verbose=visualFieldsClusteringStrategy_verbose)
		visualFieldsClusteringStrategy_disp = show_f_formation(f_formation_pred, participantsID, positions, camera.frame, (0, 255, 0))
		cv2.imshow('F-formation Detection : Visual Fields Clustering Strategy', visualFieldsClusteringStrategy_disp)
		strategiesActivated.append(visualFieldsClusteringStrategy.__name__)
	# Affichage de la frame actuelle
	camera.show_frame()
	# Mise a jour du mode evaluation
	camera.em.update(participantsID, f_formation_true, f_formation_pred, strategiesActivated)
	# Mise a jour de la recherche de parametres
	camera.ps.update(participantsID, f_formation_true, f_formation_pred)
	# Mise a jour de la memoire des participants
	camera.pm.update(participantsID, positions, f_formation_pred, strategiesActivated)

cv2.destroyAllWindows()
camera.capture.release()
