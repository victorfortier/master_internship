Créateur : Victor Fortier

Voici un petit guide d'utilisation du logiciel implémenté dans le cadre de mon stage de seconde année de Master du parcours IMA de Sorbonne Université.

Avant toute chose, pensez à indiquer dans le fichier PARAMETERS.txt et à la ligne datasetPath le chemin du dataset Mingle (le chemin doit se terminer par 'Mingle/').  
Il est donc nécessaire de posséder le dataset MatchNMingle pour que le code fonctionne : http://matchmakers.ewi.tudelft.nl/matchnmingle/pmwiki/index.php?n=Main.MatchNMingle.

# Installation et exécution

## Langage de programmation

Le programme est implémenté sous Python (version 3.8.10 mais des versions plus récentes devraient fonctionner).

## Installer les packages nécessaires

Pour installer tous les packages nécessaires, lancer le script build.sh par la commande suivante : **$ ./build.sh**

Si vous ne souhaitez pas lancer le script build.sh, assurez-vous d'avoir les bibliothèques suivantes installées :

+ opencv (version 4.5.1.48)
+ numpy
+ matplotlib
+ scikit-learn
+ scikit-image
+ scipy
+ networkx
+ pandas
+ seaborn
+ plotly (version 5.0.0)
+ Pillow

## Exécuter le code

Pour exécuter le code, lancer le script run.sh par la commande suivante : **$ ./run.sh**  
Vous pouvez ajouter deux arguments au lancement de l'exécution qui permettent de sélectionner une video du dataset Mingle : **$ ./run.sh arg1 arg2**  
Avec arg1 le numéro de jour de la vidéo (1, 2 ou 3) et arg2 le numéro de caméra de la vidéo (1, 2 ou 3).

Patientez quelques instants (~30 secondes) le temps que le code recupère les annotations du dataset Mingle.

# Interface utilisateur

## Sliders

Plusieurs sliders vous permettent d'interagir avec le programme :

+ Frame : la barre de lecture, pour sélectionner le numéro de la frame de la vidéo ;
+ Keypoints and Orientations : slider à 2 états (0 ou 1). Si activé, quelques points clés des individus (tête et épaules en vert) et la direction du regard de chaque individu (en bleu) sont affichés ;
+ Bounding Box : slider à 2 états (0 ou 1). Si activé, les boîtes englobantes des individus et leurs identifiants (en rouge) sont affichés ;
+ Labels : slider à 2 états (0 ou 1). Si activé, affiche l'état et les actions (ex : Drinking, Speaking, Action Occluded) de chaque individus (en bleu) ;
+ Social Cues Detection (Human Detection) : slider à 2 états (0 ou 1). Si activé, détecte les individus de la scene avec un detecteur HOG entrainé à partir d'un SVM obtenu à partir d'un classifieur HOG et affiche leurs boîtes englobantes (en rouge) dans une nouvelle fenêtre ;
+ F-formation Detection (Ground Truth) : slider à 2 états (0 ou 1). Si activé, affiche les groupes vérités terrains (en jaune) dans une nouvelle fenêtre ;
+ F-formation Detection (Naive Strategy) : slider à 2 états (0 ou 1). Si activé, affiche les groupes obtenus (en rose) avec l'algorithme se basant uniquement sur la proximité spatiale des individus dans une nouvelle fenêtre ;
+ F-formation Detection (O-Centers Clustering Strategy): slider à 2 états (0 ou 1). Si activé, affiche les groupes obtenus (en rouge) avec l'algorithme se basant sur le clustering des centres de O-spaces votés par les individus et ses étapes intermédiaires dans de nouvelles fenêtres ;
+ F-formation Detection (Visual Fields Clustering Strategy): slider à 2 états (0 ou 1). Si activé, affiche les groupes obtenus (en vert) avec l'algorithme se basant sur le clustering des champs visuels des individus et ses étapes intermédiaires dans de nouvelles fenêtres ; 

## Touches du clavier

Plusieurs touches du clavier vous permettent d'interagir avec le programme :

+ "p" : mettre le programme en pause ;
+ "q" : quitter le programme ;
+ "e" : lancer/stopper le mode "évaluation", qui permet d'évaluer un algorithme de détection de F-formations en terme de qualité des F-formations retournées ;
      (la similarité avec la vérité terrain) et la stabilité des F-formations retournées ;
+ "s" : sauvegarde l'image des individus contenus dans les boîtes englobantes et les redimensionnent en image 128x128 (pour le détecteur d'individus) ;
+ "m" : lancer/stopper la mémoire de chaque participants de la scène (dans le but d'améliorer les groupes détectés par une intégration temporelle) et affiche les paires d'interactions (en orange) dans une nouvelle fenêtre et les groupes obtenus à partir de ces paires et d'une mesure d'adhérence (en rose) dans une nouvelle fenêtre ;

## Rechercher les paramètres optimaux de l'algorithme 3

Si vous souhaitez lancer la recherche des paramètres optimaux de l'algorithme 3, assignez True à la ligne psActivated du fichier PARAMETERS.txt.  
Pensez à régler les paramètres précédents (frameStart, frameEnd, alphaMin, alphaMax, alphaStep, betaMin, beteMax, betaStep, gammaMin, gammaMax, gammaStep) car
les paramètres de base sont tels que la recherche des paramètres optimaux de l'algorithme 3 a un temps d'exécution très long (~ 5 heures).
