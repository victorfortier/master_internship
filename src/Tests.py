import numpy as np
from Tools import f_formationCorrection, f_formationToLabels

################################
# TEST : f_formationCorrection #
################################

participantsID = np.array([1,2,4,9,14,17,23])

# Aucune correction necessaire
f_formation = [[9,14,17,23],[1],[2,4]]
f_formation_correction = f_formationCorrection(f_formation, participantsID)
assert(f_formation_correction == f_formation)

# Supprime des F-formations les participants qui ne sont pas dans la scene
f_formation = [[9,14,17,23],[1,19],[2,4],[21]]
f_formation_correction = f_formationCorrection(f_formation, participantsID)
assert(f_formation_correction == [[9,14,17,23],[1],[2,4]])

f_formation = [[9,14,17,23],[1,19],[2,4],[21],[19]]
f_formation_correction = f_formationCorrection(f_formation, participantsID)
assert(f_formation_correction == [[9,14,17,23],[1],[2,4]])

# Supprime les F-formations vides
f_formation = [[9,14,17,23],[],[1],[2,4],[]]
f_formation_correction = f_formationCorrection(f_formation, participantsID)
assert(f_formation_correction == [[9,14,17,23],[1],[2,4]])

# Ajoute si necessaire des F-formations singletons pour les participants qui sont dans la scene mais dans aucune F-formation pour le moment
f_formation = [[9,14,17,23],[2,4]]
f_formation_correction = f_formationCorrection(f_formation, participantsID)
assert(f_formation_correction == [[9,14,17,23],[2,4],[1]])

f_formation = []
f_formation_correction = f_formationCorrection(f_formation, participantsID)
assert(f_formation_correction == [[1],[2],[4],[9],[14],[17],[23]])

# Garde qu'une seule occurence (la premiere) de chaque participant
f_formation = [[9,14,17,23],[1,9],[2,4,9],[17]]
f_formation_correction = f_formationCorrection(f_formation, participantsID)
assert(f_formation_correction  == [[9,14,17,23],[1],[2,4]])

# Garde qu'une seule occurence (la derniere) de chaque participant
f_formation = [[9,14,17,23],[1,9],[2,4,9],[17]]
f_formation_correction = f_formationCorrection(f_formation, participantsID, keep_only_one_occurence='last')
assert(f_formation_correction  == [[14,23],[1],[2,4,9],[17]])

# Garde qu'une seule occurence (celle contenue dans la F-formation la plus petite) de chaque participant
f_formation = [[9,14,17,23],[1,9],[2,4,9],[17]]
f_formation_correction = f_formationCorrection(f_formation, participantsID, keep_only_one_occurence='smallest')
assert(f_formation_correction  == [[14,23],[1,9],[2,4],[17]])

# Garde qu'une seule occurence (celle contenue dans la F-formation la plus grande) de chaque participant
f_formation = [[9,14,17,23],[1,9],[2,4,9],[17]]
f_formation_correction = f_formationCorrection(f_formation, participantsID, keep_only_one_occurence='biggest')
assert(f_formation_correction  == [[9,14,17,23],[1],[2,4]])

##############################
# TEST : f_formationToLabels #
##############################

participantsID = np.array([1,2,4,9,14,17,23])

# F-formations habituelles
f_formation = [[9,14,17,23],[1],[2,4]]
labels = f_formationToLabels(f_formation, participantsID)
assert(labels.tolist() == [1,2,2,0,0,0,0])

# Que des F-formations singletons
f_formation = [[9],[14],[17],[23],[1],[2],[4]]
labels = f_formationToLabels(f_formation, participantsID)
assert(labels.tolist() == [4,5,6,0,1,2,3])

# Qu'une seule F-formation
f_formation = [[9,14,17,23,1,2,4]]
labels = f_formationToLabels(f_formation, participantsID)
assert(labels.tolist() == [0,0,0,0,0,0,0])

# Aucune F-formation (et donc par extension aucun participant dans la scene)
f_formation = []
participantsID = np.array([])
labels = f_formationToLabels(f_formation, participantsID)
assert(labels.tolist() == [])
