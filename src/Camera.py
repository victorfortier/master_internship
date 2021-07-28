import cv2
from EvaluationMode import EvaluationMode
from ParameterSearching import ParameterSearching
from SocialCuesDetection import HumanDetector

class Camera(object):
    """
    La classe Camera permet de visualiser la video et d'interagir avec par les sliders ou par des touches du clavier.

    Attributes
    ----------
    day                 : int,
    cam                 : int,
    capture             : VideoCapture,
    FPS                 : int,
    FPS_MS              : int,
    window_name         : str,
    frameId             : int
    frameStep           : int
    numberOfFrame       : int
    frameChanged        : bool,
    frameCannotChanged  : bool,
    frame_trackbar      : str,
    em                  : EvaluationMode,
    ps                  : ParameterSearching,
    hd                  : HumanDetector,
    pause               : bool,
    quit                : bool,
    emActivated         : bool,
    psActivated         : bool,
    savePositiveSamples : bool,
    
    Methods
    -------
    initEM         : public
    initPS         : public
    initHD         : public
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
        day         : int,
        cam         : int,
        src         : str,
        FPS         : int,
        window_name : str,
        frameId     : int (optional),
        frameStep   : int (optional),
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
        # Initialise (pointeur null pour le moment) les objets EM, PS & HD
        self.em = None
        self.ps = None
        self.hd = None
        # Initialise deux booleans qui controlent la video (pour mettre la video sur pause ou la fermer)
        self.pause = False
        self.quit = False
        # Initialise deux booleans qui controlent les objets EM et PS (pour activer ou non la mise a jour de ces objets)
        self.emActivated = False
        self.psActivated = False
        # Initialise un boolean qui controle la sauvegarde des boite englobantes de la frame actuelle
        self.savePositiveSamples = False
    
    def initEM(self, savePath, emActivated=False):
        """
        Initialisation du mode evaluation.

        Parameters
        ----------
        savePath    : str,
        emActivated : bool (optional),
        """
        self.emActivated = emActivated
        self.em = EvaluationMode(self, savePath)
    
    def initPS(self, savePath, frameStart, frameEnd, alphaMin, alphaMax, alphaStep, betaMin, betaMax, betaStep, gammaMin, gammaMax, gammaStep, psActivated=False):
        """
        Initialisation de la recherche de parametres.

        Parameters
        ----------
        savePath    : str,
        frameStart  : int,
        frameEnd    : int,
        alphaMin    : float,
        alphaMax    : float,
        alphaStep   : float,
        betaMin     : float,
        betaMax     : float,
        betaStep    : float,
        gammaMin    : float,
        gammaMax    : float,
        gammaStep   : float,
        psActivated : bool (optional),
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
        savePath : str,
        """
        self.hd = HumanDetector(self, savePath)
    
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
                    self.frameStep = 20 if self.emActivated else 1
                    self.frameId += self.frameStep
                    cv2.setTrackbarPos(self.frame_trackbar, self.window_name, self.frameId)
                    self.frameCannotChanged = False
                    self.frameChanged = self.emActivated
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
        name_trackbar     : str,
        callback_function : function,
        initial_state     : bool,
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
        Verifie si une touche du clavier (q, p, e ou c) a ete pressee par l'utilisateur.

        Parameter
        ---------
        delay : float,
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
        elif key == ord('c'):
            self.savePositiveSamples = True
        else:
            if self.pause:
                self.__keyStatus(0)
