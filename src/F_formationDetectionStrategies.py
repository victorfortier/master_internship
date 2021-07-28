import cv2
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import euclidean_distances
from Figures import plot_o_centers, plot_visualFields, plot_clustering, plot_f_formation
from Tools import optimalClustering

def naiveStrategy(image, participantsID, positions, beta=0.2):
    """
    Algorithme 1 : Detection des F-formations (des groupes) par proximite spatiale des individus seulement.

    Parameters
    ----------
    image          : int [h, w] array,
    participantsID : int [n_p] array (rappel : n_p = nombre de participants dans une scene),
    positions      : float [n_p, 2] array,
    beta           : float (optional),

    Return
    ------
    f_formation : list[list[int]],
    """
    if participantsID.shape[0] != positions.shape[0]:
        return []
    if participantsID.tolist() == [] and positions.tolist() == []:
        return []
    if beta < 0 or beta > 1:
        return []
    h, w = image.shape[:2]
    numberOfParticipants = participantsID.shape[0]
    distanceMax = beta * np.sqrt(h**2+w**2)
    scoreMatrix = np.full((numberOfParticipants, numberOfParticipants), np.inf)
    for i in range(numberOfParticipants):
        x_i = positions[i,0]
        y_i = positions[i,1]
        for j in range(numberOfParticipants):
            x_j = positions[j,0]
            y_j = positions[j,1]
            if i != j:
                scoreMatrix[i,j] = np.sqrt((x_j-x_i)**2+(y_j-y_i)**2)
    adjacencyMatrix = scoreMatrix <= distanceMax
    n_components, labels = connected_components(csgraph=csr_matrix(adjacencyMatrix), directed=False, return_labels=True)
    f_formation = []
    for i in range(n_components):
        f_formation.append(participantsID[np.argwhere(labels==i)].flatten().tolist())
    return f_formation

def oCentersClusteringStrategy(image, participantsID, positions, orientations, beta=0.1, verbose=True):
    """
    Algorithme 2A : Detection des F-formations (des groupes) par clustering des centres des O-spaces votes par les individus.

    Parameters
    ----------
    image          : int [h, w] array,
    participantsID : int [n_p] array,
    positions      : float [n_p, 2] array,
    orientations   : float [n_p] array,
    beta           : float (optional),
    verbose        : bool (optional),
    
    Return
    ------
    f_formation : list[list[int]],
    """
    if participantsID.shape[0] != positions.shape[0] or positions.shape[0] != orientations.shape[0]:
        return []
    if participantsID.tolist() == [] and positions.tolist() == [] and orientations.tolist() == []:
        return []
    if beta < 0 or beta > 1:
        return []
    h, w = image.shape[:2]
    numberOfParticipants = participantsID.shape[0]
    d = beta * np.sqrt(h**2+w**2)
    o_centers = np.zeros(shape=(numberOfParticipants, 2), dtype=np.int64)
    for i in range(numberOfParticipants):
        x = positions[i,0] + d*np.cos(orientations[i])
        y = positions[i,1] + d*np.sin(orientations[i])
        o_centers[i,0] = y
        o_centers[i,1] = x
    DATA = np.unique(o_centers, axis=0)
    labels, centers = optimalClustering(DATA, numberOfParticipants)
    distCenters = []
    for i in range(numberOfParticipants):
        x = positions[i,0] + d*np.cos(orientations[i])
        y = positions[i,1] + d*np.sin(orientations[i])
        distCenters.append(np.argmin(euclidean_distances(centers, [[y,x]])))
    f_formation = []
    for i in range(centers.shape[0]):
        f_formation_i = []
        for j in range(numberOfParticipants):
            if i == distCenters[j]:
                f_formation_i.append(participantsID[j])
        f_formation.append(f_formation_i)
    if verbose:
        cv2.imshow('O-Centers Clustering Strategy : O-Centers', plot_o_centers(image.shape[0], image.shape[1], d, participantsID, positions, orientations, o_centers))
        cv2.imshow('O-Centers Clustering Strategy : Clustering', plot_clustering(image.shape[0], image.shape[1], d, positions, DATA, labels))
        cv2.imshow('O-Centers Clustering Strategy : F-formations', plot_f_formation(image.shape[0], image.shape[1], participantsID, positions, f_formation))
    return f_formation

def visualFieldsClusteringStrategy(image, participantsID, positions, orientations, numberOfSamples=100, alpha=90, beta=0.15, gamma=0.25, verbose=True):
    """
    Algorithme 3 : Detection des F-formations (des groupes) par clustering des champs visuels des individus.

    Parameters
    ----------
    image           : int [h, w] array,
    participantsID  : int [n_p] array,
    positions       : float [n_p, 2] array,
    orientations    : float [n_p] array,
    numberOfSamples : int (optional),
    alpha           : float (optional),
    beta            : float (optional),
    gamma           : float (optional),
    verbose         : bool (optional),
    
    Return
    ------
    f_formation : list[list[int]],
    """
    if participantsID.shape[0] != positions.shape[0] or positions.shape[0] != orientations.shape[0]:
        return []
    if participantsID.tolist() == [] and positions.tolist() == [] and orientations.tolist() == []:
        return []
    if alpha < 0 or alpha > 360 or beta < 0 or beta > 1 or gamma < 0 or gamma > 1:
        return []
    h, w = image.shape[:2]
    numberOfParticipants = participantsID.shape[0]
    if numberOfParticipants == 1:
        return [[participantsID[0]]]
    d = beta * np.sqrt(h**2+w**2)
    eps = gamma * d
    visualFields = np.zeros(shape=(numberOfParticipants, numberOfSamples, 2), dtype=np.int64)
    clusters = np.zeros(shape=(numberOfParticipants, numberOfSamples), dtype=np.int64)
    for i in range(numberOfParticipants):
            D = eps * np.random.rand(numberOfSamples) + (d - eps)
            THETA = np.radians(alpha) * np.random.rand(numberOfSamples) + (orientations[i] - (np.radians(alpha) / 2))
            THETA = np.mod(THETA, 2*np.pi)
            X = (positions[i,0] + D*np.cos(THETA)).astype(np.int64)
            Y = (positions[i,1] + D*np.sin(THETA)).astype(np.int64)
            visualFields[i,:,0] = Y
            visualFields[i,:,1] = X
    DATA = np.unique(visualFields.reshape(-1, visualFields.shape[-1]), axis=0)
    labels, _ = optimalClustering(DATA, numberOfParticipants)
    for k, [x,y] in enumerate(DATA):
        index = np.where(np.all(visualFields == [x,y], axis=-1))
        index = np.array(list(zip(index[0], index[1])))
        clusters[index[:,0],index[:,1]] = labels[k]
    f_formation_labels = np.zeros(shape=(numberOfParticipants), dtype=np.int64)
    for i in range(numberOfParticipants):
        f_formation_labels[i] = Counter(clusters[i].tolist()).most_common(1)[0][0]
    f_formation = []
    for i in np.unique(f_formation_labels):
        f_formation_i = []
        for j in range(len(participantsID)):
            if i == f_formation_labels[j]:
                f_formation_i.append(participantsID[j])
        f_formation.append(f_formation_i)
    if verbose:
        cv2.imshow('Visual Fields Clustering Strategy : Visual Fields', plot_visualFields(image.shape[0], image.shape[1], alpha, d, participantsID, positions, orientations, visualFields))
        cv2.imshow('Visual Fields Clustering Strategy : Clustering', plot_clustering(image.shape[0], image.shape[1], d, positions, DATA, labels))
        cv2.imshow('Visual Fields Clustering Strategy : F-formations', plot_f_formation(image.shape[0], image.shape[1], participantsID, positions, f_formation))
    return f_formation
