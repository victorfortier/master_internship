import numpy as np
from Figures import save_heatmap2D, save_heatmap3D

class ParameterSearching(object):
    """
    La classe ParameterSearching permet de lancer de multiples fois le mode evaluation sur l'algorithme 3 de detection des F-formations
    afin d'obtenir les parametres optimaux de cette methode pour la sequence video etudiee.

    Attributes
    ----------
    camera     : Camera, la camera a laquelle la "recherche de parametres" est rattachee
    savePath   : str, le chemin correspondant au dossier ou les resultats de la "recherche de parametres" s'enregistrent
    frameStart : int, la frame de depart pour la "recherche de parametres"
    frameEnd   : int, la frame de fin pour la "recherche de parametres"
    alphaRange : float [n_x] array, les differentes valeurs du premier parametre de l'algorithme 3
    betaRange  : float [n_y] array, les differentes valeurs du deuxieme parametre de l'algorithme 3
    gammaRange : float [n_z] array, les differentes valeurs du troisieme parametre de l'algorithme 3
    cpt        : list[int] (de taille 3), compteur
    heatMap_EC : float [n_x, n_y, n_z] array, la heatmap correspondant a l'evaluation des F-formations retournees par l'algorithme 3 (leurs similarites a la verite terrain)
    heatMap_SC : float [n_x, n_y, n_z] array, la heatmap correspondant a la stabilite des F-formations retournees par l'algorithme 3

    Method
    ------
    update : public
    """

    def __init__(self, camera, savePath, frameStart, frameEnd, alphaMin, alphaMax, alphaStep, betaMin, betaMax, betaStep, gammaMin, gammaMax, gammaStep):
        """
        Parameters
        ----------
        camera     : Camera, la camera a laquelle la "recherche de parametres" est rattachee
        savePath   : str, le chemin correspondant au dossier ou les resultats de la "recherche de parametres" s'enregistrent
        frameStart : int, la frame de depart pour la "recherche de parametres"
        frameEnd   : int, la frame de fin pour la "recherche de parametres"
        alphaMin    : float, valeur minimale pour le premier parametre de l'algorithme 3 de detection de F-formations
        alphaMax    : float, valeur maximale pour le premier parametre de l'algorithme 3 de detection de F-formations
        alphaStep   : float, valeur du pas pour le premier parametre de l'algorithme 3 de detection de F-formations
        betaMin     : float, valeur minimale pour le deuxieme parametre de l'algorithme 3 de detection de F-formations
        betaMax     : float, valeur maximale pour le deuxieme parametre de l'algorithme 3 de detection de F-formations
        betaStep    : float, valeur du pas pour le deuxieme parametre de l'algorithme 3 de detection de F-formations
        gammaMin    : float, valeur minimale pour le troisieme parametre de l'algorithme 3 de detection de F-formations
        gammaMax    : float, valeur maximale pour le troisieme parametre de l'algorithme 3 de detection de F-formations
        gammaStep   : float, valeur du pas pour le troisieme parametre de l'algorithme 3 de detection de F-formations
        """
        self.camera = camera
        self.savePath = savePath
        self.frameStart = frameStart
        self.frameEnd = frameEnd
        self.alphaRange = np.arange(alphaMin, alphaMax+alphaStep, alphaStep)
        self.betaRange = np.arange(betaMin, betaMax+betaStep, betaStep)
        self.gammaRange = np.arange(gammaMin, gammaMax+gammaStep, gammaStep)
        self.cpt = [0,0,0]
        self.heatMap_EC = np.zeros(shape=(self.alphaRange.size, self.betaRange.size, self.gammaRange.size))
        self.heatMap_SC = np.zeros_like(self.heatMap_EC)
    
    def update(self, participantsID, f_formation_true, f_formation_pred):
        """
        Mise a jour de la recherche de parametres.

        Parameters
        ----------
        participantsID   : int [n_p] array (rappel : n_p = nombre de participants dans une scene), les numero des participants a la scene
        f_formation_true : list[list[int]], les F-formations verites terrains
        f_formation_pred : list[list[int]], les F-formations detectees par l'algorithme evalue
        """
        if self.camera.psActivated:
            if self.camera.frameId >= self.frameEnd:
                self.camera.emActivated = False
                suffix = 'alpha='+('%.2f' % self.alphaRange[self.cpt[0]])+'_beta='+('%.2f' % self.betaRange[self.cpt[1]])+'_gamma='+('%.2f' % self.gammaRange[self.cpt[2]])
                detectionStrategy = self.camera.em.detectionStrategy
                path = self.camera.em.update(participantsID, f_formation_true, f_formation_pred, strategiesActivated=[detectionStrategy], suffix=suffix)
                with open(path+'/logs.txt', 'r') as reader:
                    lines = reader.readlines()
                    EC = np.float64(lines[2].split(' ')[-1].split('\n')[0])
                    SC = np.float64(lines[3].split(' ')[-1].split('\n')[0])
                # Mise a jour de la heatmap EC
                self.heatMap_EC[self.cpt[0],self.cpt[1],self.cpt[2]] = EC
                # Mise a jour de la heatmap SC
                self.heatMap_SC[self.cpt[0],self.cpt[1],self.cpt[2]] = SC
                # Mise a jour du compte cpt
                if self.cpt[2] == self.gammaRange.size-1:
                    if self.cpt[1] == self.betaRange.size-1:
                        # Sauvegarde de la Heat Map 2D pour la valeur de alpha actuelle
                        save_heatmap2D(self.heatMap_EC[self.cpt[0]], self.betaRange, self.gammaRange, 
                                       xlabel="gamma", ylabel="beta",
                                       title="Heatmap du coefficient d'évaluation pour les frames "+str(self.frameStart)+" à "+str(self.frameEnd)+
                                             " du jour n°"+str(self.camera.day)+" et de la caméra n°"+str(self.camera.cam)+
                                             " et pour alpha = "+('%.0f' % self.alphaRange[self.cpt[0]])+"°",
                                       filename=self.savePath+detectionStrategy+'/heatMap_EC_'+str(self.camera.day)+'_'+str(self.camera.cam)+'_'
                                                                                              +str(self.frameStart)+'_'+str(self.frameEnd)+'_'
                                                                                              +('%.2f' % self.betaRange[0])+'_'
                                                                                              +('%.2f' % self.betaRange[-1])+'_'
                                                                                              +str(self.betaRange.size)+'_'
                                                                                              +('%.2f' % self.gammaRange[0])+'_'
                                                                                              +('%.2f' % self.gammaRange[-1])+'_'
                                                                                              +str(self.gammaRange.size)+'_'
                                                                                              +('%.0f' % self.alphaRange[self.cpt[0]]))
                        save_heatmap2D(self.heatMap_SC[self.cpt[0]], self.betaRange, self.gammaRange, 
                                       xlabel="gamma", ylabel="beta",
                                       title="Heatmap du coefficient de stabilité pour les frames "+str(self.frameStart)+" à "+str(self.frameEnd)+
                                             " du jour n°"+str(self.camera.day)+" et de la caméra n°"+str(self.camera.cam)+
                                             " et pour alpha = "+('%.0f' % self.alphaRange[self.cpt[0]])+"°",
                                       filename=self.savePath+detectionStrategy+'/heatMap_SC_'+str(self.camera.day)+'_'+str(self.camera.cam)+'_'
                                                                                              +str(self.frameStart)+'_'+str(self.frameEnd)+'_'
                                                                                              +('%.2f' % self.betaRange[0])+'_'
                                                                                              +('%.2f' % self.betaRange[-1])+'_'
                                                                                              +str(self.betaRange.size)+'_'
                                                                                              +('%.2f' % self.gammaRange[0])+'_'
                                                                                              +('%.2f' % self.gammaRange[-1])+'_'
                                                                                              +str(self.gammaRange.size)+'_'
                                                                                              +('%.0f' % self.alphaRange[self.cpt[0]]))
                        if self.cpt[0] == self.alphaRange.size-1:
                            # Sauvegarde de la Heat Map 3D + Fin de la recherche 
                            save_heatmap3D(self.heatMap_EC, self.alphaRange, self.betaRange, self.gammaRange, 'EC', 'alpha', 'beta', 'gamma',
                                           title="Heatmap du coefficient d'évaluation pour les frames "+str(self.frameStart)+" à "+str(self.frameEnd)+
                                                 " du jour n°"+str(self.camera.day)+" et de la caméra n°"+str(self.camera.cam),
                                           filename=self.savePath+detectionStrategy+'/heatMap_EC_'+str(self.camera.day)+'_'+str(self.camera.cam)+'_'
                                                                                                  +str(self.frameStart)+'_'+str(self.frameEnd)+'_'
                                                                                                  +('%.0f' % self.alphaRange[0])+'_'
                                                                                                  +('%.0f' % self.alphaRange[-1])+'_'
                                                                                                  +str(self.alphaRange.size)+'_'
                                                                                                  +('%.2f' % self.betaRange[0])+'_'
                                                                                                  +('%.2f' % self.betaRange[-1])+'_'
                                                                                                  +str(self.betaRange.size)+'_'
                                                                                                  +('%.2f' % self.gammaRange[0])+'_'
                                                                                                  +('%.2f' % self.gammaRange[-1])+'_'
                                                                                                  +str(self.gammaRange.size))
                            save_heatmap3D(self.heatMap_SC, self.alphaRange, self.betaRange, self.gammaRange, 'SC', 'alpha', 'beta', 'gamma',
                                           title="Heatmap du coefficient de stabilité pour les frames "+str(self.frameStart)+" à "+str(self.frameEnd)+
                                                 " du jour n°"+str(self.camera.day)+" et de la caméra n°"+str(self.camera.cam),
                                           filename=self.savePath+detectionStrategy+'/heatMap_SC_'+str(self.camera.day)+'_'+str(self.camera.cam)+'_'
                                                                                                  +str(self.frameStart)+'_'+str(self.frameEnd)+'_'
                                                                                                  +('%.0f' % self.alphaRange[0])+'_'
                                                                                                  +('%.0f' % self.alphaRange[-1])+'_'
                                                                                                  +str(self.alphaRange.size)+'_'
                                                                                                  +('%.2f' % self.betaRange[0])+'_'
                                                                                                  +('%.2f' % self.betaRange[-1])+'_'
                                                                                                  +str(self.betaRange.size)+'_'
                                                                                                  +('%.2f' % self.gammaRange[0])+'_'
                                                                                                  +('%.2f' % self.gammaRange[-1])+'_'
                                                                                                  +str(self.gammaRange.size))
                            self.camera.quit = True
                        else:
                            # Retour a la ligne pour gamma et beta
                            self.cpt[0] += 1
                            self.cpt[1] = 0
                            self.cpt[2] = 0
                    else:
                        # Retour a la ligne pour gamma
                        self.cpt[1] += 1
                        self.cpt[2] = 0
                else:
                    self.cpt[2] += 1
                self.camera.frameId = self.frameStart
                self.camera.emActivated = True
