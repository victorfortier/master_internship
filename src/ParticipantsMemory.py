import cv2
import networkx as nx
import os
import numpy as np
from networkx.algorithms.clique import enumerate_all_cliques
from PIL import Image
from sklearn.metrics.cluster import adjusted_rand_score
from Figures import save_ARI_curve, plot_heatmap_memory, save_memory_evolution
from Tools import show_f_formation, f_formationToLabels

class ParticipantsMemory(object):
    """
    La classe ParticipantsMemory permet de modeliser la memoire de chaque participants dans une scene afin d'ameliorer la detection de groupes.
    
    Attributes
    ----------
    camera                       : Camera, la camera a laquelle la memoire de chaque individus est rattachee
    savePath                     : str, le chemin du dossier ou les resultats de la memoire de chaque individu sont sauvegardes
    dt                           : float, le parametre temporel controlant l'evolution de la memoire des individus
    tau_learn                    : float, le parametre controlant la vitesse d'apprentissage des individus de la scene
    tau_forget                   : float, le parametre controlant la vitesse d'oubli des individus de la scene
    memory_threshold             : float, le seuil de la memoire au dessus duquel deux individus sont consideres en interaction
    cpt                          : int, compteur
    detectionStrategy            : str, la methode de detection pour laquelle les F-formations retournees seront raffinees par la memoire de chaque participants
    participantsID               : int [n_p] array (rappel : n_p = nombre de participants dans une scene), les numeros des individus de la scene
    memory                       : dict{key: int, value: dict{key: int, value: float}}, la memoire de chaque individu de la scene (format dico)
    memory_array                 : float [n_p, n_p] array, la memoire de chaque individu de la scene (format array)
    groups_with_memory           : list[list[int]], les groupes obtenus a partir de la memoire de chaque individus
    groups_with_memory_corrected : list[list[int]], les groupes (corriges) a partir de la memoire de chaque individus
    memory_evolution             : list[dict{key: int, value: dict{key: int, value: float}}], l'evolution de la memoire de chaque individu de la scene
    memory_evolution_video       : VideoWriter, l'evolution de la memoire de chaque individu de la scene (ou chaque array correspond a une frame)
    frameIdList                  : list[int], la liste des frames parcourues
    ARI_list1                    : list[float], la liste correspondant aux mesures de similarite entre les groupes obtenus a partir de la memoire de chaque
                                                individus et les groupes de la verite terrain a chaque frame de la liste frameIdList
    ARI_list2                    : list[float], la liste correspondant aux mesures de similarite entre les groupes obtenus a partir de la memoire de chaque
                                                individus a l'instant t et ces memes groupes a l'instant t+1 a chaque frame de la liste frameIdList
    labels_pred_t0               : int [n_p] array, le numero de label de chaque participant au temps t
    logs                         : dict{key: str, value: int | float | list[Tuple[int, float]]}, les logs correspondant a l'evaluation menee

    Methods
    -------
    initParams              : private
    update                  : public
    learning                : private
    forgetting              : private
    show_memory             : private
    computeGroupsWithMemory : private
    membership              : private
    """

    def __init__(self, camera, savePath, dt, tau_learn=3, tau_forget=8, memory_threshold=0.5):
        """
        Parameters
        ----------
        camera           : Camera, la camera a laquelle la memoire de chaque individus est rattachee
        savePath         : str, le chemin du dossier ou les resultats de la memoire de chaque individu sont sauvegardes
        dt               : float, le parametre temporel controlant l'evolution de la memoire des individus
        tau_learn        : float (optional), le parametre controlant la vitesse d'apprentissage des individus de la scene
        tau_forget       : float (optional), le parametre controlant la vitesse d'oubli des individus de la scene
        memory_threshold : float (optional), le seuil de la memoire au dessus duquel deux individus sont consideres en interaction
        """
        #
        self.camera = camera
        #
        self.savePath = savePath
        #
        self.dt = dt
        #
        self.tau_learn = tau_learn
        #
        self.tau_forget = tau_forget
        #
        self.memory_threshold = memory_threshold
        #
        self.__initParams()
    
    def __initParams(self):
        """
        Initialisation (ou reanitialisation) des parametres pour la memoire des participants.
        """
        self.cpt = 0
        self.detectionStrategy = None
        self.participantsID = None
        self.memory = None
        self.memory_array = None
        self.groups_with_memory = None
        self.groups_with_memory_corrected = None
        self.memory_evolution = []
        self.memory_evolution_video = None
        self.frameIdList = []
        self.ARI_list1 = []
        self.ARI_list2 = []
        self.labels_pred_t0 = []
        self.logs = {
            'frame_start': 0,
            'frame_end': 0,
            'evaluation_coeff': 0.0,
            'stability_coeff': 0.0,
            'worst_evaluations': [],
            'greatest_instabilities': [],
	    }
    
    def update(self, participantsID, positions, groups_true, groups_pred, strategiesActivated):
        """
        Parameters
        ----------
        participantsID      : int [n_p] array, les numeros des individus de la scene
        positions           : float [n_p, 2] array, les positions des individus de la scene
        groups_true         : list[list[int]], les groupes verites terrains
        groups_pred         : list[list[int]], les groupes retournes par la methode de dection de F-formations activee
        strategiesActivated : list[str], la liste des methodes de detection de F-formations detectees
        """
        # Aucun groupe predit : aucune methode de detection de F-formations est en cours de traitement.
        if groups_pred is None:
            if self.camera.pmActivated:
                # Les participants ne peuvent pas memoriser les membres de leurs groupes si aucune F-formation n'est predite.
                if self.cpt == 0:
                    print('Participants Memory cannot start. No predict F-formation to process.')
                    self.camera.pmActivated = False
                    return
                # Les participants ne peuvent plus memoriser les membres de leurs groupes si plus aucune F-formation n'est predite.
                else:
                    print('Participants Memory cannot continue. No more predict F-formation.')
                    self.camera.pmActivated = False
                    return self.update(participantsID, positions, groups_true, [], [])
            return
        if self.camera.pmActivated:
            if self.cpt == 0:
                if len(strategiesActivated) != 1:
                    print('Participants Memory cannot start. More than one detection strategy is used.')
                    self.camera.pmActivated = False
                    return
                self.detectionStrategy = strategiesActivated[0]
                self.participantsID = participantsID
                self.memory = {}
                for id1 in self.participantsID:
                    self.memory[id1] = {}
                    for id2 in self.participantsID:
                        if id1 != id2:
                            self.memory[id1][id2] = self.memory_threshold
                self.memory_evolution_video = cv2.VideoWriter(self.savePath+'participantsMemory/tmp.avi', cv2.VideoWriter_fourcc('P','I','M','1'), self.camera.FPS, (800, 550))
                print('Participants Memory Activated!')
            else:
                if len(strategiesActivated) != 1:
                    print('Participants Memory cannot continue. More than one detection strategy is used.')
                    self.camera.pmActivated = False
                    self.update(participantsID, positions, groups_true, [], [self.detectionStrategy])
                if len(strategiesActivated) == 1 and strategiesActivated[0] != self.detectionStrategy:
                    print('Participants Memory cannot continue. The detection strategy changed.')
                    self.camera.pmActivated = False
                    self.update(participantsID, positions, groups_true, [], [self.detectionStrategy])
                # Suppression de la memoire des participants qui ont quittes la scene
                old_participants = list(set(self.participantsID.tolist()).difference(set(participantsID.tolist())))
                if len(old_participants) != 0:
                    for id in old_participants:
                        del self.memory[id]
                    for id1 in self.memory.keys():
                        for id in old_participants:
                            del self.memory[id1][id]
                # Ajout de la memoire des nouveaux participants a la scene
                new_participants = list(set(participantsID.tolist()).difference(set(self.participantsID.tolist())))
                if len(new_participants) != 0:
                    for id in new_participants:
                        self.memory[id] = {}
                        for id1 in self.participantsID:
                            self.memory[id][id1] = self.memory_threshold
                    for id1 in self.memory.keys():
                        for id in new_participants:
                            if id != id1:
                                self.memory[id1][id] = self.memory_threshold
            # Mise a jour de la memoire de chaque participants
            self.participantsID = participantsID
            labels = f_formationToLabels(groups_pred, self.participantsID)
            for id1 in self.memory.keys():
                g_i = labels[np.argwhere(self.participantsID == id1)]
                for id2 in self.memory[id1].keys():
                    g_j = labels[np.argwhere(self.participantsID == id2)]
                    if g_i == g_j:
                        self.memory[id1][id2] = self.__learning(self.memory[id1][id2])
                    else:
                        self.memory[id1][id2] = self.__forgetting(self.memory[id1][id2])
            # Conversion du dico en array
            n_p = self.participantsID.size
            self.memory_array = np.zeros((n_p, n_p), dtype=np.float32)
            for id1 in self.memory.keys():
                i = np.argwhere(self.participantsID == id1)
                for id2 in self.memory[id1].keys():
                    j = np.argwhere(self.participantsID == id2)
                    self.memory_array[i,j] = self.memory[id1][id2]
            self.__show_memory()
            self.__computeGroupsWithMemory()
            cv2.imshow(self.detectionStrategy+' with Individual Memory', show_f_formation(self.groups_with_memory, self.participantsID, positions, self.camera.frame, (0, 127, 255)))
            cv2.imshow(self.detectionStrategy+' with Individual Memory (Corrected)', show_f_formation(self.groups_with_memory_corrected, self.participantsID, positions, self.camera.frame, (158, 108, 253)))
            # Mesure de similarite
            labels_true = f_formationToLabels(groups_true, self.participantsID)
            labels_pred_t1 = f_formationToLabels(self.groups_with_memory_corrected, self.participantsID)
            self.ARI_list1.append(adjusted_rand_score(labels_true, labels_pred_t1))
            if len(self.frameIdList) > 0:
                if len(self.labels_pred_t0) == len(labels_pred_t1):
                    self.ARI_list2.append(adjusted_rand_score(self.labels_pred_t0, labels_pred_t1))
                else:
                    self.ARI_list2.append(-1)
            self.labels_pred_t0 = labels_pred_t1
            # Copie du dico (pour en garder une trace afin de tracer les courbes de memoires a la fin)
            memory_copy = {}
            for id1 in self.memory.keys():
                memory_id1_copy = {}
                for id2 in self.memory[id1].keys():
                    memory_id1_copy[id2] = self.memory[id1][id2]
                memory_copy[id1] = memory_id1_copy
            self.memory_evolution.append(memory_copy)
            self.frameIdList.append(self.camera.frameId)
            self.cpt += 1
            return
        else:
            # La memoire des participants vient juste d'etre stoppee par l'utilisateur. Il ne reste plus qu'a sauvegarder les resultats.
            if self.cpt > 0:
                print('Participants Memory Stopped!')
                cv2.destroyWindow('Participants Memory')
                cv2.destroyWindow(self.detectionStrategy+' with Individual Memory')
                cv2.destroyWindow(self.detectionStrategy+' with Individual Memory (Corrected)')
                self.memory_evolution_video.release()
                path = self.savePath+'participantsMemory/strategie='+self.detectionStrategy+'_day='+str(self.camera.day)+'_cam='+str(self.camera.cam)
                path += '_frame_start='+str(self.frameIdList[0])+'_frame_end='+str(self.frameIdList[-1])+'_dt='+('%.2f' % self.dt)
                if not os.path.exists(path):
                    os.mkdir(path)
                    #
                    os.rename(self.savePath+'participantsMemory/tmp.avi', path+'/memory_evolution.avi')
                    #
                    graphiques = {}
                    for memory, frameId in zip(self.memory_evolution, self.frameIdList):
                        for id1 in memory.keys():
                            # Creation du graphique pour le participant 'id1' 
                            if id1 not in graphiques:
                                graphiques[id1] = {}
                                for id2 in memory[id1].keys():
                                    graphiques[id1][id2] = ([memory[id1][id2]], [frameId])
                            else:
                                for id2 in memory[id1].keys():
                                    # Si la courbe 'id2' n'est pas dans le graphique pour le participant 'id1', on l'ajoute
                                    if id2 not in graphiques[id1]:
                                        graphiques[id1][id2] = ([memory[id1][id2]], [frameId])
                                    # Sinon on ajoute des points a la courbe 'id2' pour le graphique 'id1'
                                    else:
                                        graphiques[id1][id2][0].append(memory[id1][id2])
                                        graphiques[id1][id2][1].append(frameId)
                    for id1 in graphiques.keys():
                        graphiques[id1] = dict(sorted(graphiques[id1].items()))
                        save_memory_evolution(graphiques, id1, self.frameIdList, self.camera.frameStep, path+'/memory_'+str(id1))
                    #
                    self.logs['frame_start'] = self.frameIdList[0]
                    self.logs['frame_end'] = self.frameIdList[-1]
                    self.logs['evaluation_coeff'] = np.mean(np.array(self.ARI_list1))
                    self.logs['stability_coeff'] = np.mean(np.array(self.ARI_list2))
                    if len(self.frameIdList) >= 20:
                        sortByWorstEvaluation = list(zip(self.frameIdList, self.ARI_list1))
                        sortByWorstEvaluation.sort(key=lambda x: x[1])
                        self.logs['worst_evaluations'] = sortByWorstEvaluation[:30]
                        sortByGreatestInstabilities = list(zip(self.frameIdList, self.ARI_list2))
                        sortByGreatestInstabilities.sort(key=lambda x: x[1])
                        self.logs['greatest_instabilities'] = sortByGreatestInstabilities[:30]
                    f = open(path+"/logs.txt", 'w')
                    f.write("frame_start = "+str(self.logs['frame_start']))
                    f.write("\nframe_end = "+str(self.logs['frame_end']))
                    f.write("\nevaluation_coeff = "+str(self.logs['evaluation_coeff']))
                    f.write("\nstability_coeff = "+str(self.logs['stability_coeff']))
                    if self.logs['worst_evaluations'] != []:
                        f.write("\nworst_evaluations = "+str(self.logs['worst_evaluations']))
                    if self.logs['greatest_instabilities'] != []:
                        f.write("\ngreatest_instabilities = "+str(self.logs['greatest_instabilities']))
                    save_ARI_curve(self.ARI_list1, self.frameIdList, 
                                   title="Mesure de similarité entre la F-formation détectée et la vérité terrain pour le jour n°"
                                          +str(self.camera.day)+" et la caméra n°"+str(self.camera.cam), filename=path+'/ARI(truth,pred)')
                    save_ARI_curve(self.ARI_list2, self.frameIdList[:-1], 
                                   title="Mesure de similarité entre la F-formation détectée au temps t et celle détectée au temps t+1 pour le jour n°"
                                          +str(self.camera.day)+" et la caméra n°"+str(self.camera.cam), filename=path+'/ARI(t,t+1)')
                self.__initParams()
                return
            return

    def __learning(self, m):
        """
        Parameter
        ---------
        m : float, memoire d'un individu a un certain instant

        Return
        ------
        m_learn : float, memoire mise a jour (apprentissage)
        """
        m_learn = (1 - (self.dt / self.tau_learn)) * m + (self.dt / self.tau_learn)
        return m_learn
    
    def __forgetting(self, m):
        """
        Parameter
        ---------
        m : float, memoire d'un individu a un certain instant

        Return
        ------
        m_forget : float, memoire mise a jour (oubli)
        """
        m_forget = (1 - (self.dt / self.tau_forget)) * m
        return m_forget
    
    def __show_memory(self):
        """
        Affiche la memoire de tous les individus de la scene dans une heatmap.
        """
        img = plot_heatmap_memory(self.memory_array, self.participantsID)
        cv2.imshow('Participants Memory', img)
        if img.shape[0] == 550 and img.shape[1] == 800:
            tmp = img
        else:
            tmp = np.asarray(Image.fromarray(img).resize((800, 550)))
        self.memory_evolution_video.write(tmp)
    
    def __computeGroupsWithMemory(self):
        """
        Re-calcule les groupes a partir de la memoire de chaque individu, qui est elle meme mise a jour a partir des groupes detectes par la methode de detection.
        """
        # cliques du graphe correspondant a la matrice d'adjacacence construite a partir du seuil de la memoire de chaque individu
        pairwise_interaction = self.memory_array >= self.memory_threshold
        cliques = list(enumerate_all_cliques(nx.from_numpy_array(pairwise_interaction)))
        # selection des cliques qui sont maximales au sens de l'inclusion
        self.groups_with_memory = []
        for i, clq1 in enumerate(cliques):
            add_clq = True
            for j, clq2 in enumerate(cliques):
                if i != j:
                    if set(clq1).issubset(set(clq2)):
                        add_clq = False
                        break
            if add_clq:
                self.groups_with_memory.append(clq1)
        tmp1 = []
        for group in self.groups_with_memory:
            tmp2 = []
            for i in group:
                tmp2.append(self.participantsID[i])
            tmp1.append(tmp2)
        self.groups_with_memory = tmp1
        # selection d'un unique groupe par individu (chaque individu selectionne le groupe pour lequel il a le plus d'adherence)
        self.groups_with_memory_corrected = self.groups_with_memory
        for id in self.participantsID:
            tmp = []
            index = []
            for i, group in enumerate(self.groups_with_memory_corrected):
                if id in group:
                    index.append((i, self.__membership(id, group)))
            if len(index) >= 2:
                index.sort(key = lambda x: x[1])
                j = index[-1][0]
                for i, group in enumerate(self.groups_with_memory_corrected):
                    if id not in group:
                        tmp.append(group)
                    else:
                        if i == j:
                            tmp.append(group)
                        else:
                            l = list(filter(lambda x: x != id, group))
                            if len(l) > 0:
                                tmp.append(l)
                self.groups_with_memory_corrected = tmp
    
    def __membership(self, id, group, method='moy'):
        """
        Calcule l'adherence du participant possedant le numero id au groupe "group".

        Parameters
        ----------
        id     : int, le numero du participant en question
        group  : list[int], la liste des numeros des individus appartenant au meme groupe que id
        method : str (optional), la methode d'estimation de l'adherence a un groupe

        Return
        ------
        membership : float, adherence au groupe
        """
        membership = 0.0
        tmp = [self.memory[id][id1] for id1 in group if id != id1]
        if len(tmp) > 0:
            if method == 'moy':
                membership = np.mean(np.array(tmp))
            elif method == 'max':
                membership = np.max(np.array(tmp))
        return membership
