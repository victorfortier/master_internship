import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns
from Tools import show_f_formation, f_formationToLabels, fig2img
from networkx.algorithms.clique import enumerate_all_cliques

mpl.style.use('seaborn')
fontsize = plt.rcParams['axes.titlesize']

class ParticipantsMemory(object):
    """
    """

    def __init__(self, camera, dt, tau_learn=3, tau_forget=8, memory_threshold=0.5):
        """
        """
        #
        self.camera = camera
        #
        self.dt = dt
        #
        self.tau_learn = tau_learn
        #
        self.tau_forget = tau_forget
        #
        self.memory_threshold = memory_threshold
        #
        self.cpt = 0
        #
        self.detectionStrategy = None
        self.participantsID = None
        self.positions = None
        self.groups = None
        self.groups_with_memory = None
        self.memory = None
        self.memory_array = None

    def update(self, participantsID, positions, groups, strategiesActivated):
        """
        """
        if self.camera.pmActivated:
            self.positions = positions
            self.groups = groups
            labels = f_formationToLabels(self.groups, participantsID)
            if self.cpt == 0:
                self.detectionStrategy = strategiesActivated[0]
                self.participantsID = participantsID
                self.memory = {}
                for id1 in self.participantsID:
                    self.memory[id1] = {}
                    for id2 in self.participantsID:
                        if id1 != id2:
                            self.memory[id1][id2] = self.memory_threshold
            else:
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
            cv2.imshow(self.detectionStrategy+' with Individual Memory', show_f_formation(self.groups_with_memory, self.participantsID, self.positions, self.camera.frame, (0, 127, 255)))
            self.cpt += 1
    
    def __learning(self, m):
        """
        """
        m_learn = (1 - (self.dt / self.tau_learn)) * m + (self.dt / self.tau_learn)
        return m_learn
    
    def __forgetting(self, m):
        """
        """
        m_forget = (1 - (self.dt / self.tau_forget)) * m
        return m_forget
    
    def __show_memory(self):
        """
        """
        index = ['%d' % id for id in self.participantsID.tolist()]
        columns = ['%d' % id for id in self.participantsID.tolist()]
        df = pd.DataFrame(self.memory_array, index=index, columns=columns)
        svm = sns.heatmap(df, annot=True, fmt=".2f", vmin=0, vmax=1)
        plt.title("Participants memory", fontsize=fontsize/1.34)
        plt.xlabel("Participants")
        plt.ylabel("Participants")
        fig = svm.get_figure()
        img = fig2img(fig)
        cv2.imshow('Participants Memory', img)
    
    def __computeGroupsWithMemory(self):
        """
        """
        pairwise_interaction = self.memory_array >= self.memory_threshold
        tmp = enumerate_all_cliques(nx.from_numpy_array(pairwise_interaction))
        cliques = []
        for clq in tmp:
            cliques.append(clq)
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
