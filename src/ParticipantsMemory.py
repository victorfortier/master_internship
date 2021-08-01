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
    camera                       : Camera,
    savePath                     : str,
    dt                           : float,
    tau_learn                    : float,
    tau_forget                   : float,
    memory_threshold             : float,
    cpt                          : int,
    detectionStrategy            : str,
    participantsID               : int [n_p] array (rappel : n_p = nombre de participants dans une scene),
    memory                       : dict{key: int, value: dict{key: int, value: float}},
    memory_array                 : float [n_p, n_p] array,
    groups_with_memory           : list[list[int]],
    groups_with_memory_corrected : list[list[int]],
    memory_evolution             : list[dict{key: int, value: dict{key: int, value: float}}],
    memory_evolution_video       : VideoWriter,
    frameIdList                  : list[int],
    ARI_list1                    : list[float],
    ARI_list2                    : list[float],
    labels_pred_t0               : int [n_p] array,

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
        camera           : Camera,
        savePath         : str,
        dt               : float,
        tau_learn        : float (optional),
        tau_forget       : float (optional),
        memory_threshold : float (optional),
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
    
    def update(self, participantsID, positions, groups_true, groups_pred, strategiesActivated):
        """
        Parameters
        ----------
        participantsID      : int [n_p] array,
        positions           : float [n_p, 2] array,
        groups_true         : list[list[int]],
        groups_pred         : list[list[int]],
        strategiesActivated : list[str],
        """
        # Aucun groupe predit : aucune methode de detection de F-formations est en cours de traitement.
        if groups_pred is None:
            if self.camera.pmActivated:
                # Les participants ne peuvent pas memoriser les membres de leurs groupes si aucune F-formation n'est predite.
                if self.cpt == 0:
                    print('Participants Memory cannot start. No predict F-formation to process.')
                    self.camera.pmActivated = False
                    return
                # Les participants ne peuvent plus mémoriser les membres de leurs groupes si plus aucune F-formation n'est predite.
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
                    shift = 0.01
                    for id1 in graphiques.keys():
                        Y = []
                        for id2 in graphiques[id1].keys():
                            x = graphiques[id1][id2][1]
                            y = graphiques[id1][id2][0]
                            tmp = np.full((len(self.frameIdList),), np.nan)
                            for i, frameId in enumerate(self.frameIdList):
                                j = np.argwhere(np.array(x) == frameId)
                                if len(j) == 1:
                                    tmp[i] = y[j[0][0]]
                            Y.append(tmp.tolist())
                        Y = np.array(Y).astype(np.float32)
                        for j in range(Y.shape[1]):
                            frameId = self.frameIdList[j]
                            close_curves_pairwise = np.zeros((Y.shape[0], Y.shape[0]), dtype=int)
                            for i1 in range(Y.shape[0]):
                                if not np.isnan(Y[i1,j]):
                                    for i2 in range(i1, Y.shape[0]):
                                        if not np.isnan(Y[i2,j]):
                                            if i1 != i2:
                                                if abs(Y[i1,j]-Y[i2,j]) <= shift:
                                                    close_curves_pairwise[i1,i2] = 1
                            cliques = list(enumerate_all_cliques(nx.from_numpy_array(close_curves_pairwise)))
                            close_curves = []
                            for k1, clq1 in enumerate(cliques):
                                add_clq = True
                                for k2, clq2 in enumerate(cliques):
                                    if k1 != k2:
                                        if set(clq1).issubset(set(clq2)):
                                            add_clq = False
                                            break
                                if add_clq:
                                    close_curves.append(clq1)
                            for curves in close_curves:
                                tmp1 = list(zip(curves, Y[np.array(curves),j].tolist()))
                                tmp1.sort(key=lambda x:x[1])
                                tmp2 = []
                                for ki in range(len(tmp1)):
                                    tmp2.append(tmp1[((ki+1)//2)*(-1)**ki + len(tmp1)//2])
                                di = [-2 for _ in range(len(tmp2))]
                                if len(di) > 0:
                                    di[0] = 0
                                if len(di) > 1:
                                    di[1] = -1
                                for ki in range(1,len(tmp2)):
                                    dy = abs(tmp2[ki+di[ki]][1]-tmp2[ki][1])
                                    if (-1)**ki == 1:
                                        if tmp2[ki][1] >= tmp2[ki+di[ki]][1]:
                                            tmp2[ki] = (tmp2[ki][0], tmp2[ki][1]+(shift-dy)*(-1)**ki)
                                        else:
                                            tmp2[ki] = (tmp2[ki][0], tmp2[ki][1]+(shift+dy)*(-1)**ki)
                                    else:
                                        if tmp2[ki][1] <= tmp2[ki+di[ki]][1]:
                                            tmp2[ki] = (tmp2[ki][0], tmp2[ki][1]+(shift-dy)*(-1)**ki)
                                        else:
                                            tmp2[ki] = (tmp2[ki][0], tmp2[ki][1]+(shift+dy)*(-1)**ki)
                                for ki in range(len(tmp2)):
                                    Y[tmp2[ki][0],j] = tmp2[ki][1]
                        x = np.array(self.frameIdList)
                        for k, id2 in enumerate(graphiques[id1].keys()):
                            y = Y[k,:]
                            index = np.argwhere(~np.isnan(y)).flatten()
                            graphiques[id1][id2] = (y[index].tolist(), x[index].tolist())
                        graphiques[id1] = dict(sorted(graphiques[id1].items()))
                        save_memory_evolution(graphiques, id1, self.frameIdList, path+'/memory_'+str(id1), shift=0)
                    #
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
        m : float,

        Return
        ------
        m_learn : float,
        """
        m_learn = (1 - (self.dt / self.tau_learn)) * m + (self.dt / self.tau_learn)
        return m_learn
    
    def __forgetting(self, m):
        """
        Parameter
        ---------
        m : float,

        Return
        ------
        m_forget : float,
        """
        m_forget = (1 - (self.dt / self.tau_forget)) * m
        return m_forget
    
    def __show_memory(self):
        """
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
        """
        #
        pairwise_interaction = self.memory_array >= self.memory_threshold
        cliques = list(enumerate_all_cliques(nx.from_numpy_array(pairwise_interaction)))
        #
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
        #
        tmp1 = []
        for group in self.groups_with_memory:
            tmp2 = []
            for i in group:
                tmp2.append(self.participantsID[i])
            tmp1.append(tmp2)
        self.groups_with_memory = tmp1
        #
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
        Parameters
        ----------
        id     : int,
        group  : list[int],
        method : str (optional),

        Return
        ------
        membership : float,
        """
        membership = 0.0
        tmp = [self.memory[id][id1] for id1 in group if id != id1]
        if len(tmp) > 0:
            if method == 'moy':
                membership = np.mean(np.array(tmp))
            elif method == 'max':
                membership = np.max(np.array(tmp))
        return membership
