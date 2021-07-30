import csv
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

###################
# Outils basiques #
###################

def getParameters():
	"""
	Retourne les parametres necessaire a l'initialisation d'une camera, du mode evaluation et de la rechercher de parametres.

	Return
	------
	PARAMETERS : dict{key: str, value: str | int | float | bool},
	"""
	with open('PARAMETERS.txt', 'r') as reader:
		lines = reader.readlines()
	key = []
	value = []
	for line in lines:
		if '=' in line:
			tmp = line.split('=')
			key.append(tmp[0].split(' ')[0])
			value.append(tmp[-1].split(' ')[-1].split('\n')[0])
	tmp = dict(zip(key,value))
	PARAMETERS = dict()
	for key, value in tmp.items():
		PARAMETERS[key] = eval(value)
	return PARAMETERS

def getValue(dico, key, defaultValue):
	"""
	Retourne la valeur d'une clef d'un dictionnaire si la clef existe, une valeur par defaut sinon.

	Parameters
	----------
	dico         : dict{key: str, value: str | int | float | bool},
	key          : str,
	defaultValue : str | int | float | bool,
	
	Return
	------
	value : str | int | float | bool,
	"""
	value = dico[key] if key in dico else defaultValue
	return value

def openCSV(filename):
	"""
	Convertie un fichier CSV en un array.

	Parameters
	----------
	filename : str,

	Return
	------
	arr : Any array,
	"""
	rows = []
	with open(filename) as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			rows.append(row)
	arr = np.array(rows)
	return arr

def isNaN(value):
	"""
	Retourne True si value est egal a 'NaN', False sinon.
	"""
	return value == 'NaN'

def timeToFrame(time, FPS):
	"""
	Convertie un temps (au format min:sec) en nombre de frames (necessite ainsi le nombre de frames par seconde de la video).

	Parameters
	----------
	time : str,
	FPS  : int,

	Return
	------
	frame : int,
	"""
	t = time.split(":")
	secondes = np.int64(t[0])*60 + np.int64(t[1])
	frame = secondes * FPS
	return frame

#######################################################
# Outils pour la manipulation du dataset MatchNMingle #
#######################################################

def getManualAnnotations(frameId, participants, keypointsAndOrientations, data, labels):
	"""
	Retourne les annotations de la frame actuelle du dataset Mingle.

	Parameters
	----------
	frameId                  : int,
	participants             : int [n_p_day, 2] array (rappel : n_p_day = nombre de participants au jour 'day'),
	keypointsAndOrientations : float [numberOfFrames, n_p_all*13] array (rappel : n_p_all = nombre de participants aux soirees cocktail),
	data                     : float [numberOfFrames, n_p_all*7] array,
	labels                   : int [numberOfFrames, n_p_all*9] array,

	Return
	------
	manual_annotations : list[list],
	"""
	manual_annotations = []
	for [globalId, _] in participants:
		manual_annotations_person = []
		## keypointsAndOrientations
		if frameId >= 0 and frameId < keypointsAndOrientations.shape[0]:
			manual_annotations_person.append(keypointsAndOrientations[frameId, 13*(globalId-1):13*globalId])
		else:
			manual_annotations_person.append(None)
		## boundingBox
		if frameId >= 0 and frameId < data.shape[0]:
			manual_annotations_person.append(data[frameId, 7*(globalId-1):7*globalId])
		else:
			manual_annotations_person.append(None)
		## labels
		if frameId >= 0 and frameId < labels.shape[0]:
			manual_annotations_person.append(labels[frameId, 9*(globalId-1):9*globalId])
		else:
			manual_annotations_person.append(None)
		manual_annotations.append(manual_annotations_person)
	return manual_annotations

def getInfoForF_formationDetection(manual_annotations, participants, cam):
	"""
	Retourne les informations necessaires a la bonne detection de F-formations.

	Parameters
	----------
	manual_annotations : list[list],
	participants       : int [n_p_day, 2] array,
	cam                : int,

	Returns
	-------
	participantsID : int [n_p] array (rappel : n_p = nombre de participants dans une scene),
	positions      : float [n_p, 2] array,
	orientations   : float [n_p] array,
	"""
	participantsID = []
	positions = []
	orientations = []
	for i, anno in enumerate(manual_annotations):
		if anno[0] is not None:
			if not isNaN(anno[0][1]) and int(anno[0][1]):
				if not isNaN(anno[0][2]) and int(anno[0][2]) == cam:
					if 'NaN' not in anno[0]:
						headX = np.float64(anno[0][5])
						headY = np.float64(anno[0][6])
						hOrient_deg = np.float64(anno[0][11])
						participantsID.append(participants[i,1])
						positions.append([headX,headY])
						orientations.append(hOrient_deg)
	participantsID = np.array(participantsID)
	positions = np.array(positions)
	orientations = np.radians(np.array(orientations))
	return participantsID, positions, orientations

###########################################################################
# Outils pour la detection des F-formations (des groupes) dans des images #
###########################################################################

def optimalClustering(DATA, numberOfParticipants):
	"""
	Algorithme 2B : Recherche du clustering optimal d'un ensemble de donnees (l'ensemble des participants a la scene).

	Parameters
	----------
	DATA                 : int [n_data, 2] array,
	numberOfParticipants : int,
	
	Returns
	-------
	labels  : list[int] (de taille n_data),
	centers : float [n_clusters, 2] array,
	"""
	lab = []
	cen = []
	sco = []
	for n in range(2, min(numberOfParticipants+1, DATA.shape[0])):
		kmeans = KMeans(n_clusters=n).fit(DATA)
		lab.append(kmeans.labels_)
		cen.append(kmeans.cluster_centers_)
		sco.append(silhouette_score(DATA, lab[-1], metric='euclidean'))
	n_clusters = np.array(sco).argmax() + 2
	labels = lab[n_clusters - 2]
	centers = cen[n_clusters - 2] 
	return labels, centers

def getAlpha(camera):
	"""
	Retourne le parametre alpha (optimal ou dependant de la recherche des parametres) pour l'algorithme 3.
	"""
	return camera.ps.alphaRange[camera.ps.cpt[0]] if camera.psActivated else 90

def getBeta(camera):
	"""
	Retourne le parametre beta (optimal ou dependant de la recherche des parametres) pour l'algorithme 3.
	"""
	return camera.ps.betaRange[camera.ps.cpt[1]] if camera.psActivated else 0.15

def getGamma(camera):
	"""
	Retourne le parametre gamma (optimal ou dependant de la recherche des parametres) pour l'algorithme 3.
	"""
	return camera.ps.gammaRange[camera.ps.cpt[2]] if camera.psActivated else 0.25

###############################################
# Outils pour la manipulation de F-formations #
###############################################

def videoFormatToFrameFormatForF_formation(f_formation_video, numberOfFrame):
	"""
	Convertie les F-formations qui sont en format video (participants, frame de debut, frame de fin) en une liste de F-formations (frame par frame).

	Parameters
	----------
	f_formation_video : list[list[int] (de taille 3)] (de taille le nombre de F-formations dans la sequence video),
	numberOfFrame     : int,

	Return
	------
	f_formation       : list[list[list[int]] (de taille le nombre de F-formations dans la frame)] (de taille numberOfFrame),

	"""
	f_formation = [[] for i in range(numberOfFrame)]
	for i in range(len(f_formation_video)):
		participants_f_formation = f_formation_video[i][0]
		frame_start = f_formation_video[i][1]
		frame_end = f_formation_video[i][2]
		for j in range(frame_start, frame_end+1):
			f_formation[j].append(participants_f_formation)
	return f_formation

def f_formationCorrection(f_formation, participantsID, keep_only_one_occurence='first'):
	"""
	Corrige la liste de F-formation.

	Parameters
	----------
	f_formation             : list[list[int]] (de taille le nombre de F-formations dans la scene),
	participantsID          : int [n_p] array,
	keep_only_one_occurence : str (optional),

	Return
	------
	f_formation_correction : list[list[int]] (de taille le bon nombre de F-formations dans la scene),

	Raise
	-----
	ValueError : dans le cas ou l'argument keep_only_one_occurence n'est pas reconnu par l'algorithme
	"""
	# Supprime des F-formations les participants qui ne sont pas dans la scene
	tmp1 = []
	for i in range(len(f_formation)):
		tmp2 = []
		for dayId in f_formation[i]:
			if dayId in participantsID:
				tmp2.append(dayId)
		# Supprime les F-formations vides
		if len(tmp2) > 0:
			tmp1.append(tmp2)
	f_formation_correction = tmp1
	# Ajoute si necessaire des F-formations singletons pour les participants qui sont dans la scene mais dans aucune F-formation pour le moment
	for dayId in participantsID:
		if len(f_formation_correction) > 0:
			if dayId not in np.concatenate(f_formation_correction):
				f_formation_correction.append([dayId])
		else:
			f_formation_correction.append([dayId])
	# Garde qu'une seule occurence de chaque participant
	for dayId in participantsID:
		tmp = []
		index = []
		for i in range(len(f_formation_correction)):
			if dayId in f_formation_correction[i]:
				index.append((i, len(f_formation_correction[i])))
		if len(index) >= 2:
			if keep_only_one_occurence == 'random':
				j = index[np.random.randint(len(index))][0]
			elif keep_only_one_occurence == 'first':
				j = index[0][0]
			elif keep_only_one_occurence == 'last':
				j = index[-1][0]
			else:
				index.sort(key = lambda x: x[1])
				if keep_only_one_occurence == 'smallest':
					j = index[0][0]
				elif keep_only_one_occurence == 'biggest':
					j = index[-1][0]
				else:
					raise ValueError(keep_only_one_occurence+" is not a correct way to keep only one occurence of each participant ID (choose between random, first, last, smallest and biggest).")
			for i in range(len(f_formation_correction)):
				if dayId not in f_formation_correction[i]:
					tmp.append(f_formation_correction[i])
				else:
					if i == j:
						tmp.append(f_formation_correction[i])
					else:
						l = list(filter(lambda x: x != dayId, f_formation_correction[i]))
						if len(l) > 0:
							tmp.append(l)
			f_formation_correction = tmp
	return f_formation_correction

def show_f_formation(f_formation, participantsID, positions, frame, color):
	"""
	Affichage des F-formations directement sur le frame.

	Parameters
	----------
	f_formation    : list[list[int]] (de taille le nombre de F-formations),
	participantsID : int [n_p] array,
	positions      : float [n_p, 2] array,
	frame          : int [h, w, 3] array,
	color          : Tuple[int*3],

	Return
	------
	image : int [h, w, 3] array,
	"""
	image = frame.copy()
	for k in range(len(f_formation)):
		coords = []
		for dayId in f_formation[k]:
			index = participantsID.tolist().index(dayId)
			coords.append(positions[index])
		if len(coords) == 1:
			image = cv2.circle(image, (np.int64(coords[0][0]), np.int64(coords[0][1])), 5, color, -1)
		else:
			for i in range(len(coords)):
				for j in range(i+1,len(coords)):
					image = cv2.line(image, (np.int64(coords[i][0]), np.int64(coords[i][1])), (np.int64(coords[j][0]), np.int64(coords[j][1])), color, 2)
	return image

def f_formationToLabels(f_formation, participantsID):
	"""
	Convertie une liste de F-formations en une liste de labels.
	Par exemple, si F = [[9,14,17,23],[1],[2,4]] (la liste de F-formations), alors X = [1,2,2,0,0,0,0] (la liste de labels).

	Parameters
	----------
	f_formation : list[list[int]],
	participantsID : int [n_p] array,

	Return
	------
	labels : int [n_p] array,
	"""
	labels = np.zeros_like(participantsID)
	for i in range(len(f_formation)):
		for dayId in f_formation[i]:
			index = participantsID.tolist().index(dayId)
			labels[index] = i
	return labels

######################################
# Outils pour la creation de figures #
######################################

def fig2img(fig):
	"""
	Convertie une figure en une image RGB (sans oublier de supprimer la figure de la memoire pour ne pas la surcharger).

	Parameter
	---------
	fig : Figure,

	Return
	------
	img : [h, w, 3] array,
	"""
	fig.canvas.draw()
	img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
	img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
	plt.close(fig)
	return img

def getColorPalette(nb_colors):
	"""
	Retourne une palette de couleur construite a partir des palettes 'Set1' et 'Set3' de matplotlib et contenant le nombre de couleurs voulu.
	Si le nombre de couleurs voulu est superieur au nombre de couleurs maximal des palettes 'Set1' et 'Set3',
	les couleurs restantes sont tirees aleatoirement dans l'espace RGBA (entre 0 et 1 pour RGB, la couleur de transparence etant fixee a 1).

	Parameter
	---------
	nb_colors : int,

	Return
	------
	colorPalette : list[Tuple[float**4]],
	"""
	if nb_colors < 0:
		return []
	cmap1 = cm.get_cmap(name='Set1')
	cmap2 = cm.get_cmap(name='Set3')
	colorPalette = []
	for i in range(cmap1.N):
		colorPalette.append(cmap1(i))
	for i in range(cmap2.N):
		colorPalette.append(cmap2(i))
	if nb_colors <= cmap1.N + cmap2.N:
		return colorPalette[:nb_colors]
	else:
		for i in range(cmap1.N + cmap2.N, nb_colors):
			colorPalette.append(tuple(np.random.rand(3).astype(np.float64).tolist()+np.float64([1]).tolist()))
		return colorPalette
