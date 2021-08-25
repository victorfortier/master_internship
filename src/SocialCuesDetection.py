import cv2
import numpy as np
import os
from PIL import Image
from skimage.feature import hog
from sklearn import svm
from Tools import isNaN

class HumanDetector(object):
	"""
	La classe HumanDetector permet de detecter les humains presents dans une scene sous forme de boites englobantes (et donc leurs positions).
	La detection se fait avec un classifieur HOG entraine par un SVM lineaire sur des images d'individus (positive samples)
	et des images ou aucun individu n'est present (negative samples). Ces images sont de taille 128x128.

	Attributes
	----------
	camera        : Camera, la camera a laquelle le detecteur d'individus est rattache
	savePath      : str, le chemin du dossier dans lequel l'image des individus englobes par les boites englobantes sont enregistres
	tol           : float, seuil a partir duquel une region de l'image detectee par l'algorithme de detection d'individus est reelement considere comme un invidu
	humanDetector : HOGDescriptor, detecteur se basant sur un classifieur HOG des individus de la scene
	boundingBoxes : float [n_p, 4] array (rappel : n_p = nombre de participants dans une scene), la liste des boites englobantes des individus detectes dans la scene

	Methods
	-------
	initHumanDetector           : private
	getSVMDetector              : private
	computeSVMDetector          : private
	update                      : public
	positiveSamplesSavingUpdate : public
	"""

	def __init__(self, camera, savePath, tol=0.3):
		"""
		Parameters
		----------
		camera   : Camera, la camera a laquelle le detecteur d'individus est rattache
		savePath : str, le chemin du dossier dans lequel l'image des individus englobes par les boites englobantes sont enregistres
		tol      : float, seuil a partir duquel une region de l'image detectee par l'algorithme de detection d'individus est reelement considere comme un invidu
		"""
		self.camera = camera
		self.savePath = savePath
		self.tol = tol
		self.humanDetector = None
		self.boundingBoxes = None
		self.__initHumanDetector()
	
	def __initHumanDetector(self):
		"""
		Initialisation du classifieur HOG.
		"""
		winSize = (128,128)
		blockSize = (16,16)
		blockStride = (8,8)
		cellSize = (8,8)
		nbins = 9
		derivAperture = 1
		winSigma = 4.
		histogramNormType = 0
		L2HysThreshold = 2.0000000000000001e-01
		gammaCorrection = 0
		nlevels = 64
		hogDescriptor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
										  histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
		SVMDetector = self.__getSVMDetector()
		hogDescriptor.setSVMDetector(SVMDetector)
		self.humanDetector = hogDescriptor

	def __getSVMDetector(self):
		"""
		Si le SVM lineaire a deja ete appris, on le recupere, sinon on l'apprend et on le sauvegarde.
		"""
		try:
			SVMDetector = np.loadtxt(self.savePath+'SVMDetector.csv', delimiter='\n')
		except IOError:
			SVMDetector = self.__computeSVMDetector()
		return SVMDetector
	
	def __computeSVMDetector(self):
		"""
		Calcul du SVM lineaire.
		"""
		features = []
		labels = []

		# Positive samples
		path = self.savePath+'trainingData/positiveSamples/'
		for _, _, files in os.walk(path):
			for filename in files:
				if filename.split('.')[-1] == 'jpg':
					img = cv2.imread(path+filename, 0)
					hist = hog(img, orientations=9, pixels_per_cell=(8, 8), 
							   cells_per_block=(2, 2), block_norm= 'L2')
					features.append(hist.tolist())
					labels.append(1)
		# Negative samples
		path = self.savePath+'trainingData/negativeSamples/'
		for _, _, files in os.walk(path):
			for filename in files:
				if filename.split('.')[-1] == 'jpg':
					img = cv2.imread(path+filename, 0)
					hist = hog(img, orientations=9, pixels_per_cell=(8, 8), 
							   cells_per_block=(2, 2), block_norm= 'L2')
					features.append(hist.tolist())
					labels.append(0)
		
		features = np.array(features, dtype=np.float32)
		labels = np.array(labels)

		clf = svm.SVC(kernel='linear')
		clf.fit(features,labels)

		SVMDetector = clf.coef_.flatten()
		np.savetxt(self.savePath+'SVMDetector.csv', SVMDetector, delimiter='\n')
		return SVMDetector
	
	def update(self, detectionActivated):
		"""
		Mise a jour du detecteur d'humains.

		Parameters
		----------
		detectionActivated : bool, True si la detection des individus de la scene est activee par l'utilisateur, False sinon
		"""
		if detectionActivated:
			gray = cv2.cvtColor(self.camera.frame, cv2.COLOR_BGR2GRAY)
			tmp = self.camera.frame.copy()
			boxes, weights = self.humanDetector.detectMultiScale(gray, winStride=(8,8))
			boxes = np.array([[x,y,x+w,y+h] for (x,y,w,h) in boxes])
			if boxes.size != 0:
				weights = weights.flatten()
				index = []
				for i, (x1, y1, x2, y2) in enumerate(boxes):
					if weights[i] >= self.tol:
						cv2.rectangle(tmp, (x1, y1), (x2, y2), (0, 0, 255), 2)
						index.append(i)
				if len(index) != 0:
					self.boundingBoxes = boxes[index]
				else:
					cv2.putText(tmp, 'No human detected!', (0, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0, 0, 255), 1)
					self.boundingBoxes = np.array([])
			else:
				cv2.putText(tmp, 'No human detected!', (0, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0, 0, 255), 1)
				self.boundingBoxes = np.array([])
			cv2.imshow('Human Detector', tmp)
	
	def positiveSamplesSavingUpdate(self, manual_annotations):
		"""
		Pour sauvegarder les images contenues dans les boites englobantes (redimensionner en images 128x128).

		Parameters
		----------
		manual_annotations : list[list], l'ensemble des annotations fournies par la base de donnes Mingle pour la frame courante
		"""
		if self.camera.savePositiveSamples:
			path = self.savePath+'trainingData/positiveSamples/'
			im_files = []
			for _, _, files in os.walk(path):
				im_files += files
			if len(im_files) != 0:
				im_files = [np.int64(im.split('.')[0]) for im in im_files]
				im_files.sort()
				im_filename = im_files[-1] + 1
			else:
				im_filename = 1
			cpt = 0
			for anno in manual_annotations:
				if anno[0] is not None and anno[1] is not None and anno[2] is not None:
					if not isNaN(anno[0][1]) and int(anno[0][1]):
						if not isNaN(anno[0][2]) and int(anno[0][2]) == self.camera.cam:
							x1 = anno[1][2]
							y1 = anno[1][1]
							x2 = anno[1][4]
							y2 = anno[1][3]
							x1 = max(np.int64(x1),0)
							y1 = max(np.int64(y1),0)
							x2 = min(np.int64(x2),self.camera.frame.shape[0]-1)
							y2 = min(np.int64(y2),self.camera.frame.shape[1]-1)
							image = Image.fromarray(cv2.cvtColor(self.camera.frame, cv2.COLOR_BGR2RGB))
							image = image.crop((y1,x1,y2,x2))
							image = image.resize((128,128))
							image.save(path+str(im_filename+cpt)+'.jpg')
							cpt += 1
			self.camera.savePositiveSamples = False
