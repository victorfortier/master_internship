import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from Tools import f_formationToLabels, fig2img

mpl.style.use('seaborn')
fontsize = plt.rcParams['axes.titlesize']

class GroupsMemory(object):

    def __init__(self, detectionMethod, dt, tau_learn=3, tau_forget=8, memory_threshold=0.5):
        self.detectionMethod = detectionMethod
        self.dt = dt
        self.tau_learn = tau_learn
        self.tau_forget = tau_forget
        self.memory_threshold = memory_threshold
        self.cpt = 0
        self.memory = None
        self.participantsID = None
        
    def update(self, image, participantsID, positions, orientations):
        groups = self.detectionMethod(image, participantsID, positions, orientations)
        labels = f_formationToLabels(groups, participantsID)
        print(groups)
        if self.cpt == 0:
            self.memory = []
            self.participantsID = participantsID
            for i, id1 in enumerate(self.participantsID):
                memory_id1 = []
                for j, id2 in enumerate(self.participantsID):
                    if i != j:
                        memory_id1.append((id2, self.memory_threshold))
                memory_id1 = dict(memory_id1)
                self.memory.append((id1, memory_id1))
            self.memory = dict(self.memory)
            print(self.memory)
        self.cpt += 1
        # Ajout de la memoire des nouveaux participants a la scene
        # TO DO
        # Suppression de la memoire des participants qui ont quittes la scene
        # TO DO
        # Mise à jour de la memoire de chaque participants
        self.participantsID = participantsID
        M = []
        for id1, memory_id1 in self.memory.items():
            i = labels[np.argwhere(self.participantsID == id1)]
            M_id1 = []
            for id2, memory_id1_id2 in memory_id1.items():
                j = labels[np.argwhere(self.participantsID == id2)]
                if i == j:
                    # Learning
                    M_id1_id2 = self.__learning(memory_id1_id2)
                else:
                    # Forgetting
                    M_id1_id2 = self.__forgetting(memory_id1_id2)
                M_id1.append((id2, M_id1_id2))
            M_id1 = dict(M_id1)
            M.append((id1, M_id1))
        M = dict(M)
        self.memory = M
        self.__show_memory()
        print(self.memory)
    
    def __learning(self, m):
        return (1 - (self.dt / self.tau_learn)) * m + (self.dt / self.tau_learn)
    
    def __forgetting(self, m):
        return (1 - (self.dt / self.tau_forget)) * m
    
    def __show_memory(self):
        index = ['%d' % id for id in self.participantsID.tolist()]
        columns = ['%d' % id for id in self.participantsID.tolist()]
        n_p = self.participantsID.size
        heatmap = np.zeros((n_p, n_p), dtype=np.float32)
        for id1, memory_id1 in self.memory.items():
            i = np.argwhere(self.participantsID == id1)
            for id2, memory_id1_id2 in memory_id1.items():
                j = np.argwhere(self.participantsID == id2)
                heatmap[i,j] = memory_id1_id2
        df = pd.DataFrame(heatmap, index=index, columns=columns)
        svm = sns.heatmap(df, annot=True, fmt=".2f", vmin=0, vmax=1)
        plt.title("Memory", fontsize=fontsize/1.34)
        plt.xlabel("Participants")
        plt.ylabel("Participants")
        fig = svm.get_figure()
        img = fig2img(fig)
        cv2.imshow('Memory of participants', img)
