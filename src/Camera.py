import cv2
from EvaluationMode import EvaluationMode
from ParameterSearching import ParameterSearching
from SocialCuesDetection import HumanDetector
from ParticipantsMemory import ParticipantsMemory

class Camera(object):
    """
    La classe Camera permet de visualiser la video et d'interagir avec par des sliders ou par des touches du clavier afin d'y superposer des informations
    sur les groupes (F-formations) et sur les individus eux memes (la position des boites englobantes, la direction du regard de chaque individu, etc).

    Attributes
    ----------
    day                 : int, le numero du jour (1, 2 ou 3)
    cam                 : int, le numero de la camera (1, 2 ou 3)
    capture             : VideoCapture, la capture
    FPS                 : int, le nombre d'images par seconde de la video etudiee
    FPS_MS              : int, le nombre de millisecondes entre chaque frame
    window_name         : str, le nom de la fenetre ou la video s'affiche
    frameId             : int, le numero de la frame courante
    frameStep           : int, le pas de frames
    numberOfFrame       : int, le nombre total de frames de la video
    frameChanged        : bool, True si la frame est modifiee par l'utilisateur avec la barre de lecture de la video, False sinon
    frameCannotChanged  : bool, True si la frame ne peut pas etre modifiee, False sinon
    frame_trackbar      : str, le nom de la barre de lecture qui permet de controler la video
    em                  : EvaluationMode, le mode "evaluation" rattache a la video
    ps                  : ParameterSearching, la "recherche de parametres" rattachee a la video
    hd                  : HumanDetector, le detecteur d'individus rattache a la video
    pm                  : ParticipantsMemory, la memoire de chaque participants apparaissant sur la video
    pause               : bool, True si l'utilisateur a mis en pause la video, False sinon
    quit                : bool, True si l'utilisateur souhaite fermer la video, False sinon
    emActivated         : bool, True si le mode "evaluation" est en cours de fonctionnement, False sinon
    psActivated         : bool, True si la "recherche de parametres" est en cours de fonctionnement, False sinon
    savePositiveSamples : bool, True si l'utilisateur souhaite enregistrer les individus contenus dans les boites englobantes, False sinon
    pmActivated         : bool, True si la memoire de chaque participants est en cours de fonctionnement, False sinon
    
    Methods
    -------
    initEM         : public
    initPS         : public
    initHD         : public
    initPM         : public
    update         : public
    show_frame     : public
    add_cbox       : public
    change_frameId : private
    keyStatus      : private
    """
    
    def __init__(self, day, cam, src, FPS, window_name, frameId=0, frameStep=1):
        """
        Parameters
        ----------
        day         : int, le numero du jour (1, 2 ou 3)
        cam         : int, le numero de la camera (1, 2 ou 3)
        src         : str, le chemin correspondant a la video a etudiee
        FPS         : int, le nombre d'images par seconde de la video etudiee
        window_name : str, le nom de la fenetre ou la video s'affiche
        frameId     : int (optional), le numero de la frame courante
        frameStep   : int (optional), le pas de frames
        """
        # Initialise le jour et la camera de la video etudiee (du dataset MatchNMingle)
        self.day = day
        self.cam = cam
        # Initialise la capture
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        # Initialise les FPS
        self.FPS = FPS
        self.FPS_MS = int(1 / self.FPS * 1000)
        # Initialise le nom de la fenetre ou la video apparait + creation de cette fenetre
        self.window_name = window_name
        cv2.namedWindow(self.window_name)
        # Initialise le numero de la frame de depart, le pas de temps et le nombre total de frame
        self.frameId = frameId
        self.frameStep = frameStep
        self.numberOfFrame = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # Initialise deux booleans permettant de controler la barre de lecture de la video
        self.frameChanged = False
        self.frameCannotChanged = False
        # Initialise le nom de la barre de lecture de la video + creation de cette barre
        self.frame_trackbar = 'Frame'
        cv2.createTrackbar(self.frame_trackbar, self.window_name, 0, self.numberOfFrame-1, self.__change_frameId)
        cv2.setTrackbarPos(self.frame_trackbar, self.window_name, self.frameId)
        # Initialise (pointeur null pour le moment) les objets EM, PS, HD et PM
        self.em = None
        self.ps = None
        self.hd = None
        self.pm = None
        # Initialise deux booleans qui controlent la video (pour mettre la video sur pause ou la fermer)
        self.pause = False
        self.quit = False
        # Initialise deux booleans qui controlent les objets EM et PS (pour activer ou non la mise a jour de ces objets)
        self.emActivated = False
        self.psActivated = False
        # Initialise un boolean qui controle la sauvegarde des boite englobantes de la frame actuelle
        self.savePositiveSamples = False
        # Initialise un boolean qui controle l'evolution de la memoire des participants a la scene
        self.pmActivated = False
    
    def initEM(self, savePath, emActivated=False):
        """
        Initialisation du mode evaluation.

        Parameters
        ----------
        savePath    : str, le chemin correspondant au dossier ou les resultats du mode "evaluation" s'enregistrent
        emActivated : bool (optional), True si le mode "evaluation" se met en route des son initialisation, False sinon
        """
        self.emActivated = emActivated
        self.em = EvaluationMode(self, savePath)
    
    def initPS(self, savePath, frameStart, frameEnd, alphaMin, alphaMax, alphaStep, betaMin, betaMax, betaStep, gammaMin, gammaMax, gammaStep, psActivated=False):
        """
        Initialisation de la recherche de parametres.

        Parameters
        ----------
        savePath    : str, le chemin correspondant au dossier ou les resultats de la "recheche de parametres" s'enregistrent
        frameStart  : int, la frame de depart pour la "recherche de parametres"
        frameEnd    : int, la frame de fin pour la "recherche de parametres"
        alphaMin    : float, valeur minimale pour le premier parametre de l'algorithme 3 de detection de F-formations
        alphaMax    : float, valeur maximale pour le premier parametre de l'algorithme 3 de detection de F-formations
        alphaStep   : float, valeur du pas pour le premier parametre de l'algorithme 3 de detection de F-formations
        betaMin     : float, valeur minimale pour le deuxieme parametre de l'algorithme 3 de detection de F-formations
        betaMax     : float, valeur maximale pour le deuxieme parametre de l'algorithme 3 de detection de F-formations
        betaStep    : float, valeur du pas pour le deuxieme parametre de l'algorithme 3 de detection de F-formations
        gammaMin    : float, valeur minimale pour le troisieme parametre de l'algorithme 3 de detection de F-formations
        gammaMax    : float, valeur maximale pour le troisieme parametre de l'algorithme 3 de detection de F-formations
        gammaStep   : float, valeur du pas pour le troisieme parametre de l'algorithme 3 de detection de F-formations
        psActivated : bool (optional), True si la "recherche de parametre" se met en route des son initialisation, False sinon
        """
        self.psActivated = psActivated
        self.ps = ParameterSearching(self, savePath, frameStart, frameEnd, alphaMin, alphaMax, alphaStep, betaMin, betaMax, betaStep, gammaMin, gammaMax, gammaStep)
        if self.psActivated:
            self.frameId = self.ps.frameStart
            self.frameChanged = True
            self.emActivated = True
    
    def initHD(self, savePath):
        """
        Initialisation du detecteur d'humains.

        Parameters
        ----------
        savePath : str, le chemin correspondant au dossier ou les resultats du detecteur d'humains s'enregistrent
        """
        self.hd = HumanDetector(self, savePath)

    def initPM(self, savePath, dt, tau_learn=3, tau_forget=8, memory_threshold=0.5):
        """
        Initialise la memoire des participants.

        Parameters
        ----------
        savePath         : str, le chemin correspondant au dossier ou les resultats de la memoire de chaque participants a la scene s'enregistrent
        dt               : float, parametre temporelle pour l'evolution de la memoire de chaque individu
        tau_learn        : float (optional), parametre qui controle la vitesse d'apprentissage entre deux individus
        tau_forget       : float (optional), parametre qui controle la vitesse d'oubli entre deux individus
        memory_threshold : float (optional), parametre qui controle le seuil de la memoire au dessus duquel deux individus sont consideres en interaction
        """
        self.pm = ParticipantsMemory(self, savePath, dt, tau_learn=tau_learn, tau_forget=tau_forget, memory_threshold=memory_threshold)

    def update(self):
        """
        Mise a jour de la capture. Prend en compte le cas ou la frame a ete modifiee par l'utilisateur avec la barre de lecture.
        """
        if not self.quit:
            if self.capture.isOpened():
                if self.frameChanged:
                    self.capture.set(1, self.frameId - 1)
                    self.frameChanged = False
                self.status, self.frame = self.capture.read()
                if self.status:
                    self.frame_copy = self.frame.copy()
                    self.frameCannotChanged = True
                    if self.emActivated:
                        self.frameStep = 20
                    elif self.pmActivated:
                        self.frameStep = int(self.FPS * self.pm.dt)
                    else:
                        self.frameStep = 1
                    self.frameId += self.frameStep
                    cv2.setTrackbarPos(self.frame_trackbar, self.window_name, self.frameId)
                    self.frameCannotChanged = False
                    self.frameChanged = self.emActivated or self.pmActivated
                    return True
        return False
    
    def show_frame(self):
        """
        Permet d'afficher la frame actuelle dans la fenetre de capture. Verifie egalement si une touche du clavier a ete pressee.
        """
        cv2.imshow(self.window_name, self.frame_copy)
        self.__keyStatus(self.FPS_MS)
    
    def add_cbox(self, name_trackbar, callback_function, initial_state):
        """
        Ajoute un CBOX (un slider avec seulement deux etats 0 ou 1) a la capture.

        Parameters
        ----------
        name_trackbar     : str, le nom de la fenetre ou la video s'affiche
        callback_function : function, le nom de la fonction qui est appellee lorsque le slider est manipule par l'utilisateur
        initial_state     : bool, l'etat initial du slider (0 ou 1)
        """
        cv2.createTrackbar(name_trackbar, self.window_name, 0, 1, callback_function)
        cv2.setTrackbarPos(name_trackbar, self.window_name, 1 if initial_state else 0)
    
    def __change_frameId(self, value):
        """
        Modifie l'attribut frameId si la barre de lecture a ete utilisee par l'utilisateur.

        Parameter
        ---------
        value : int, 
        """
        if not self.frameCannotChanged:
            self.frameId = value
            self.frameChanged = True
    
    def __keyStatus(self, delay):
        """
        Verifie si une touche du clavier (q, p, e, s ou m) a ete pressee par l'utilisateur.

        Parameter
        ---------
        delay : float, le nombre de millisecondes entre chaque frames (par defaut FPS_MS, sinon 0 lorsque la video est en pause)
        """
        key = cv2.waitKey(delay) & 0xff
        if key == ord('q'):
            self.quit = True
        elif key == ord('p'):
            self.pause = not self.pause
            if self.pause:
                self.__keyStatus(0)
        elif key == ord('e'):
            if not self.psActivated:
                self.emActivated = not self.emActivated
            if self.pause:
                self.__keyStatus(0)
        elif key == ord('s'):
            self.savePositiveSamples = True
        elif key == ord('m'):
            self.pmActivated = not self.pmActivated
            if self.pause:
                self.__keyStatus(0)
        else:
            if self.pause:
                self.__keyStatus(0)
