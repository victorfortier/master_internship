import numpy as np
import os
from sklearn.metrics.cluster import adjusted_rand_score
from Figures import save_ARI_curve
from Tools import f_formationToLabels

class EvaluationMode(object):
    """
    La classe EvaluationMode permet d'evaluer un certain algoritme de detection de F-formations (en terme de qualite des F-formations retournees
    et de dynamiques des groupes detectes) pendant un sequence video selectionne par l'utilisateur.

    Attributes
    ----------
    camera   : Camera,
    savePath : str,
    detectionStrategy : str,
    frameIdList : list[int],
    ARI_list1 : list[float],
    ARI_list2 : list[float],
    labels_pred_t0 : int [n_p] array (rappel : n_p = nombre de participants dans une scene),
    logs : dict{key: str, value: int | float | list[Tuple[int, float]]},

    Method
    ------
    initParams : private
    update     : public
    """

    def __init__(self, camera, savePath):
        """
        Parameters
        ----------
        camera   : Camera,
        savePath : str,
        """
        self.camera = camera
        self.savePath = savePath
        self.__initParams()
    
    def __initParams(self):
        """
        Initialisation (ou reanitialisation) des parametres pour le mode evaluation.
        """
        self.detectionStrategy = None
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
    
    def update(self, participantsID, f_formation_true, f_formation_pred, strategiesActivated, suffix=None):
        """
        Mise a jour du mode evaluation.

        Parameters
        ----------
        participantsID      : int [n_p] array,
        f_formation_true    : list[list[int]],
        f_formation_pred    : list[list[int]],
        strategiesActivated : list[str],
        suffix              : str (optional),

        Return
        ------
        path : str,
        """
        # Aucune F-formation predite : aucune methode de detection de F-formations est en cours de traitement.
        if f_formation_pred is None:
            if self.camera.emActivated:
                # Le mode evaluation ne peut pas commencer si aucune F-formation n'est predite.
                if len(self.frameIdList) == 0:
                    print('Evaluation Mode cannot start. No predict F-formation to process.')
                    self.camera.emActivated = False
                    return None
                # Le mode evaluation ne peut pas continuer si plus aucune F-formation n'est predite.
                else:
                    print('Evaluation Mode cannot continue. No more predict F-formation.')
                    self.camera.emActivated = False
                    # Il faut toutefois sauvegarder l'evaluation avant de l'arreter.
                    return self.update(participantsID, f_formation_true, [], [], suffix=suffix)
            return None
        if self.camera.emActivated:
            if self.frameIdList == []:
                if len(strategiesActivated) != 1:
                    print('Evaluation Mode cannot start. More than one detection strategy is used.')
                    self.camera.emActivated = False
                    return None
                print("--------------------------------------------------------------------------------")
                print("Evaluation Mode Activated!")
                self.detectionStrategy = strategiesActivated[0]
            if len(strategiesActivated) != 1:
                print('Evaluation Mode cannot continue. More than one detection strategy is used.')
                self.camera.emActivated = False
                return self.update(participantsID, f_formation_true, [], [self.detectionStrategy], suffix=suffix)
            if len(strategiesActivated) == 1 and strategiesActivated[0] != self.detectionStrategy:
                print('Evaluation Mode cannot continue. The detection strategy changed.')
                self.camera.emActivated = False
                return self.update(participantsID, f_formation_true, [], [self.detectionStrategy], suffix=suffix)
            labels_true = f_formationToLabels(f_formation_true, participantsID)
            labels_pred_t1 = f_formationToLabels(f_formation_pred, participantsID)
            print("frameId =", self.camera.frameId, "(n°"+str(len(self.frameIdList)+1)+")")
            self.ARI_list1.append(adjusted_rand_score(labels_true, labels_pred_t1))
            if len(self.frameIdList) > 0:
                if len(self.labels_pred_t0) == len(labels_pred_t1):
                    self.ARI_list2.append(adjusted_rand_score(self.labels_pred_t0, labels_pred_t1))
                else:
                    self.ARI_list2.append(-1)
            self.labels_pred_t0 = labels_pred_t1
            self.frameIdList.append(self.camera.frameId)
            return None
        else:
            # Le mode evaluation vient juste d'etre stoppe par l'utilisateur. Il ne reste plus qu'a sauvegarder cette evaluation.
            if len(self.frameIdList) > 0:
                # L'algorithme n'a ete evalue que sur 0,1 ou 2 frames. L'evaluation est impossible.
                if len(self.frameIdList) < 3:
                    print("Evaluation Mode Stopped! Not enough data to process.")
                    self.__initParams()
                    return None
                print("Evaluation Mode Stopped!")
                self.logs['frame_start'] = self.frameIdList[0]
                self.logs['frame_end'] = self.frameIdList[-1]
                self.logs['evaluation_coeff'] = np.mean(np.array(self.ARI_list1))
                self.logs['stability_coeff'] = np.mean(np.array(self.ARI_list2))
                if len(self.frameIdList) >= 20:
                    sortByWorstEvaluation = list(zip(self.frameIdList, self.ARI_list1))
                    sortByWorstEvaluation.sort(key=lambda x: x[1])
                    self.logs['worst_evaluations'] = sortByWorstEvaluation[:10]
                    sortByGreatestInstabilities = list(zip(self.frameIdList, self.ARI_list2))
                    sortByGreatestInstabilities.sort(key=lambda x: x[1])
                    self.logs['greatest_instabilities'] = sortByGreatestInstabilities[:10]
                path = self.savePath+self.detectionStrategy+'/day='+str(self.camera.day)+'_cam='+str(self.camera.cam)
                path += '_frame_start='+str(self.logs['frame_start'])+'_frame_end='+str(self.logs['frame_end'])+('_'+suffix if suffix is not None else '')
                if not os.path.exists(path):
                    os.mkdir(path)
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
                return path
            return None
