Créateur : Victor Fortier

Voici un petit guide d'utilisation du logiciel implémenté dans le cadre de mon stage de seconde année de Master du parcours IMA de Sorbonne Université.

Avant toutes choses, pensez à indiquer dans le fichier PARAMETERS.txt et à la ligne datasetPath le chemin du dataset Mingle (le chemin doit se terminer par 'Mingle/').  
Il est donc nécessaire de posséder le dataset MatchNMingle pour que le code fonctionne : http://matchmakers.ewi.tudelft.nl/matchnmingle/pmwiki/index.php?n=Main.MatchNMingle.

# Installation et exécution

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

et bien évidemment Python 3.7.

## Exécuter le code

Pour exécuter le code, lancer le script run.sh par la commande suivante : **$ ./run.sh**

Patientez quelques instants (~30 secondes) le temps que le code recupère les annotations du dataset Mingle.

# Interface utilisateur

## Sliders

## Touches du clavier

+ "p" : mettre le programme en pause
+ "q" : quitter le programme
+ "e" : lancer/stopper le mode "évaluation", qui permet d'évaluer un algorithme de détection de F-formations en terme de qualité des F-formations retournées
      (la similarité avec la vérité terrain) et la stabilité des F-formations retournées
+ "s" : sauvegarde l'image des individus contenus dans les boîtes englobantes et les redimensionnent en image 128x128 (pour le détecteur d'individus)
+ "m" : lancer/stopper la mémoire de chaque participants de la scène (dans le but d'améliorer les groupes détectés par une intégration temporelle)

## Rechercher les paramètres optimaux de l'algorithme 3

Si vous souhaitez lancer la recherche des paramètres optimaux de l'algorithme 3, assignez True à la ligne psActivated du fichier PARAMETERS.txt.  
Pensez à regler les paramètres précédents (frameStart, frameEnd, alphaMin, alphaMax, alphaStep, betaMin, beteMax, betaStep, gammaMin, gammaMax, gammaStep) car
les paramètres de base sont tels que la recherche des paramètres optimaux de l'algorithme 3 a un temps d'exécution très long (~ 5 heures).
