import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib.patches import Circle, Polygon
from Tools import f_formationToLabels
from Tools import fig2img, getColorPalette

mpl.style.use('seaborn')
dpi = plt.rcParams['figure.dpi']
fontsize = plt.rcParams['axes.titlesize']

###########################################################
# Figures pour les methodes de detection des F-formations #
###########################################################

def plot_o_centers(nbRows, nbColumns, d, participantsID, positions, orientations, o_centers):
    """
    Pour l'affichage des votes des O-spaces realises par les individus, de leurs positions et de leurs orientations.

    Parameters
    ----------
    nbRows         : int, le nombre de lignes de l'array correspondant a l'image de la scene
    nbColumns      : int, le nombre de colonnes de l'array correspondant a l'image de la scene
    d              : float, la distance a laquelle les centres de O-spaces sont votes
    participantsID : int [n_p] array (rappel : n_p = nombre de participants dans une scene), les numeros des participants de la scene
    positions      : float [n_p, 2] array, les positions des participants a la scene
    orientations   : float [n_p] array, l'orientation de chaque participants a la scene
    o_centers      : int [n_p, 2] array, les centres des O-spaces votes par les individus

    Return
    ------
    img : int [h, w, 3] array, la figure illustrant les individus (leurs positions et leurs orientations) et leurs votes de centre de O-space
    """
    delta = np.int64(d)
    wfig, hfig = nbColumns+2*delta, nbRows+2*delta
    fig = plt.figure()
    fig.set_size_inches(wfig/dpi, hfig/dpi)
    ax = fig.gca()
    ax.set_xlim([0,wfig-1])
    ax.set_ylim([0,hfig-1])
    ax.invert_yaxis()
    colorPalette = getColorPalette(o_centers.shape[0])
    o_centers_points = []
    for i in range(o_centers.shape[0]):
        tmp, = plt.plot(o_centers[i,1]+delta, o_centers[i,0]+delta, marker='o', linestyle='None', color=colorPalette[i], zorder=2)
        o_centers_points.append(tmp)
    participants_points, = plt.plot(positions[:,0] + delta, positions[:,1] + delta, marker='D', linestyle='None', color='black', zorder=3)
    for i in range(len(participantsID)):
        x1 = positions[i,0] + delta
        y1 = positions[i,1] + delta
        x2 = x1 + d * np.cos(orientations[i])
        y2 = y1 + d * np.sin(orientations[i])
        orientation_line, = plt.plot([x1,x2], [y1,y2], linestyle='dashed', color='limegreen', zorder=1)
    legend1 = ax.legend(o_centers_points, ['O-center du participant '+str(participantsID[i]) for i in range(len(participantsID))], loc='upper left')
    legend2 = ax.legend([participants_points, orientation_line], ['Participants', 'Orientation des partipants'], loc='upper right')
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    fig.tight_layout()
    img = fig2img(fig)
    return img

def plot_visualFields(nbRows, nbColumns, alpha, d, participantsID, positions, orientations, visualFields):
    """
    Pour l'affichage des cones de vision des individus, de leurs positions et de leurs champs visuels maximaux.

    Parameters
    ----------
    nbRows         : int, le nombre de lignes de l'array correspondant a l'image de la scene
    nbColumns      : int, le nombre de colonnes de l'array correspondant a l'image de la scene
    alpha          : float, l'angle de vision de chaque individu de la scene
    d              : float, la distance maximale a laquelle les individus votent
    participantsID : int [n_p] array, les numeros des individus de la scene
    positions      : float [n_p, 2] array, les positions des individus de la scene
    orientations   : float [n_p] array, l'orientation de chaque individu de la scene
    visualFields   : int [n_p, n_samples, 2] array (rappel : n_samples = nombre de votes realises par chaque individu), les champs visuels de chaque individu

    Return
    ------
    img : int [h, w, 3] array, la figure illustrant les individus (leurs positions et les limites de leurs champs visuels) et leurs champs visuels
    """
    delta = np.int64(d)
    def plot_visualFieldsLimits():
        for i in range(len(participantsID)):
            x1 = positions[i,0] + delta
            y1 = positions[i,1] + delta
            theta = np.mod(orientations[i]-np.radians(alpha/2), 2*np.pi) 
            x2 = x1 + d * np.cos(theta)
            y2 = y1 + d * np.sin(theta)
            theta = np.mod(orientations[i]+np.radians(alpha/2), 2*np.pi)
            x3 = x1 + d * np.cos(theta)
            y3 = y1 + d * np.sin(theta)
            visualFieldsLimits, = plt.plot([x1,x2], [y1,y2], linestyle='dashed', color='limegreen', zorder=2)
            visualFieldsLimits, = plt.plot([x1,x3], [y1,y3], linestyle='dashed', color='limegreen', zorder=2)
            X = []
            Y = []
            THETA = np.linspace(-(alpha/2), alpha/2, num=100, endpoint=True)
            for theta in THETA:
                ori = np.mod(orientations[i]+np.radians(theta), 2*np.pi)
                x4 = x1 + d * np.cos(ori)
                y4 = y1 + d * np.sin(ori)
                X.append(x4)
                Y.append(y4)
            visualFieldsLimits, = plt.plot(X, Y, linestyle='dashed', color='limegreen', zorder=2)
        return visualFieldsLimits
    wfig, hfig = nbColumns+2*delta, nbRows+2*delta
    fig = plt.figure()
    fig.set_size_inches(wfig/dpi, hfig/dpi)
    ax = fig.gca()
    ax.set_xlim([0,wfig-1])
    ax.set_ylim([0,hfig-1])
    ax.invert_yaxis()
    colorPalette = getColorPalette(visualFields.shape[0])
    visualFields_points = []
    for i in range(visualFields.shape[0]):
        tmp, = plt.plot(visualFields[i,:,1]+delta, visualFields[i,:,0]+delta, marker='o', linestyle='None', color=colorPalette[i], zorder=1)
        visualFields_points.append(tmp)
    participants_points, = plt.plot(positions[:,0]+delta, positions[:,1]+delta, marker='D', linestyle='None', color='black', zorder=3)
    visualFieldsLimits = plot_visualFieldsLimits()
    legend1 = ax.legend(visualFields_points, ['Cône de vision du participant '+str(participantsID[i]) for i in range(len(participantsID))], loc='upper left')
    legend2 = ax.legend([participants_points, visualFieldsLimits], ['Participants', 'Limites des champs de vision des participants'], loc='upper right')
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    fig.tight_layout()
    img = fig2img(fig)
    return img

def plot_clustering(nbRows, nbColumns, d, positions, DATA, labels):
    """
    Pour l'affichage du clustering d'un ensemble de donnees votes par les individus et de leurs positions.

    Parameters
    ----------
    nbRows    : int, le nombre de lignes de l'array correspondant a l'image de la scene
    nbColumns : int, le nombre de colonnes de l'array correspondant a l'image de la scene
    d         : float, la distance maximale a laquelle chaque individu peut voter
    positions : float [n_p, 2] array, les positions des individus de la scene
    DATA      : int [n_p, 2] array, les positions des donnees votees par les individus de la scene
    labels    : list[int] (de taille n_data), les labels des donnes votees par les individus de la scene

    Return
    ------
    img : int [h, w, 3] array, la figure illustrant la position des individus et les clusters de donnees
    """
    delta = np.int64(d)
    wfig, hfig = nbColumns+2*delta, nbRows+2*delta
    fig = plt.figure()
    fig.set_size_inches(wfig/dpi, hfig/dpi)
    ax = fig.gca()
    ax.set_xlim([0,wfig-1])
    ax.set_ylim([0,hfig-1])
    ax.invert_yaxis()
    colorPalette = getColorPalette(np.unique(labels).size)
    clusters_points= []
    for i in range(np.unique(labels).size):
        cluster_i = DATA[np.argwhere(labels == i)]
        tmp = np.zeros(shape=(cluster_i.shape[0], 2), dtype=np.int64)
        for j in range(cluster_i.shape[0]):
            tmp[j,:] = cluster_i[j,0]
        cluster_i = tmp
        tmp, = plt.plot(cluster_i[:,1]+delta, cluster_i[:,0]+delta, marker='o', linestyle='None', color=colorPalette[i], zorder=1)
        clusters_points.append(tmp)
    participants_points, = plt.plot(positions[:,0]+delta, positions[:,1]+delta, marker='D', linestyle='None', color='black', zorder=2)
    legend1 = ax.legend(clusters_points, ['Cluster '+str(i+1) for i in range(len(clusters_points))], loc='upper left')
    legend2 = ax.legend([participants_points], ['Participants'], loc='upper right')
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    fig.tight_layout()
    img = fig2img(fig)
    return img

def plot_f_formation(nbRows, nbColumns, participantsID, positions, f_formation):
    """
    Pour l'affichage de F-formations et de la position des individus de la scene.

    Parameters
    ----------
    nbRows         : int, le nombre de lignes de l'array correspondant a l'image de la scene
    nbColumns      : int, le nombre de colonnes de l'array correspondant a l'image de la scene
    participantsID : int [n_p] array, les numeros des individus de la scene
    positions      : float [n_p, 2] array, les positions des individus de la scene
    f_formation    : list[list[int]], les F-formations a afficher

    Return
    ------
    img : int [h, w, 3] array, la figure illustrant les F-formations de la scene
    """
    fig = plt.figure()
    fig.set_size_inches(nbColumns/dpi, nbRows/dpi)
    ax = fig.gca()
    ax.set_xlim([0,nbColumns-1])
    ax.set_ylim([0,nbRows-1])
    ax.invert_yaxis()
    participants_points, = plt.plot(positions[:,0], positions[:,1], marker='D', linestyle='None', color='black')
    labels = f_formationToLabels(f_formation, participantsID)
    colorPalette = getColorPalette(np.unique(labels).size)
    for i in range(np.unique(labels).size):
        coords_i = positions[np.argwhere(labels == i)]
        tmp = np.zeros(shape=(coords_i.shape[0], 2), dtype=np.float64)
        for j in range(coords_i.shape[0]):
            tmp[j,:] = coords_i[j,0]
        coords_i = tmp
        if coords_i.shape[0] == 1:
            circle = Circle(tuple(coords_i[0]), 20, color=colorPalette[i])
            ax.add_patch(circle)
        elif coords_i.shape[0] == 2:
            x1 = coords_i[0,0]
            y1 = coords_i[0,1]
            x2 = coords_i[1,0]
            y2 = coords_i[1,1]
            x3 = (x1 + x2) / 2
            y3 = (y1 + y2) / 2
            r = np.sqrt((x2-x3)**2 + (y2-y3)**2)
            circle = Circle((x3,y3), r, color=colorPalette[i])
            ax.add_patch(circle)
        else:
            # Calcul des centroides
            cent = (np.sum(coords_i[:,0])/coords_i.shape[0], np.sum(coords_i[:,1])/coords_i.shape[0])
            tmp = coords_i.tolist()
            # Trie par angle polaire
            tmp.sort(key=lambda p: np.arctan2(p[1]-cent[1],p[0]-cent[0]))
            polygon = Polygon(tmp, True, color=colorPalette[i])
            ax.add_patch(polygon)
    legend = ax.legend([participants_points], ['Participants'], loc='best')
    ax.add_artist(legend)
    fig.tight_layout()
    img = fig2img(fig)
    return img

########################################################################
# Figures pour l'evaluation d'une methode de detection de F-formations #
########################################################################

def save_ARI_curve(ARI_list, frameIdList, title, filename):
    """
    Pour la sauvegarde de la courbe de la mesure de similarite ARI en fonction du temps (des frames de la video).

    Parameters
    ----------
    ARI_list    : list[float], liste de mesure de similarite
    frameIdList : list[int], liste des numeros de frame
    title       : str, le titre du graphique
    filename    : str, le nom du fichier a sauvegarder
    """
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim([frameIdList[0],frameIdList[-1]])
    ax.set_ylim([-1.25,1.25])
    plt.plot(frameIdList, ARI_list)
    plt.title(title, fontsize=fontsize/1.34)
    plt.xlabel('Frames')
    plt.ylabel(filename.split('/')[-1])
    fig.savefig(filename+'.png', dpi=dpi)
    plt.close(fig)

def save_heatmap2D(heatmap, rows_range, columns_range, xlabel, ylabel, title, filename):
    """
    Pour la sauvegarde d'une heatmap 2D.

    Parameters
    ----------
    heatmap       : float [n_x, n_y] array, une heatmap 2D
    rows_range    : float [n_x] array, le nombre de lignes de la heatmap
    columns_range : float [n_y] array, le nombre de colonnes de la heatmap
    xlabel        : str, le label de l'axe x
    ylabel        : str, le label de l'axe y
    title         : str, le titre de la heatmap
    filename      : str, le nom du fichier a sauvegarder
    """
    index = ['%.2f' % r for r in rows_range.tolist()]
    columns = ['%.2f' % c for c in columns_range.tolist()]
    df = pd.DataFrame(heatmap, index=index, columns=columns)
    svm = sns.heatmap(df, annot=True, fmt=".5f", vmin=-1, vmax=1, center=0)
    plt.title(title, fontsize=fontsize/1.34)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig = svm.get_figure()
    fig.savefig(filename+'.png', dpi=dpi)
    plt.close(fig)

def save_heatmap3D(heatmap, x_range, y_range, z_range, heat_label, x_label, y_label, z_label, title, filename):
    """
    Pour la sauvegarde d'une heatmap 3D (au format .html).

    Parameters
    ----------
    heatmap    : float [n_x, n_y, n_z] array, une heatmap 3D
    x_range    : float [n_x] array, les valeurs prises par l'axe x
    y_range    : float [n_y] array, les valeurs prises par l'axe y
    z_range    : float [n_z] array, les valeurs prises par l'axe z
    heat_label : str, SC (Stability Coefficient) ou EC (Evaluation Coefficient)
    x_label    : str, le label de l'axe x
    y_label    : str, le label de l'axe y
    z_label    : str, le label de l'axe z
    title      : str, le titre de la heatmap
    filename   : str, le nom du fichier a sauvegarder
    """
    df = []
    for i in range(x_range.size):
        for j in range(y_range.size):
            for k in range(z_range.size):
                df.append([heatmap[i,j,k], x_range[i], y_range[j], z_range[k]])
    df = np.array(df)
    df = pd.DataFrame(df, index=list(range(x_range.size*y_range.size*z_range.size)), columns=[heat_label, x_label, y_label, z_label])
    fig = px.scatter_3d(df, x=x_label, y=y_label, z=z_label, color=heat_label, color_continuous_scale=px.colors.sequential.Magma, title=title)
    fig.write_html(filename+'.html')

########################################################
# Figures pour la memoire des participants d'une scene #
########################################################

def plot_heatmap_memory(heatmap, rows_and_columns):
    """
    Pour l'affichage d'une heatmap.

    Parameters
    ----------
    heatmap          : float [n_p, n_p] array, la heatmap correspondante a la memoire de chaque individu de la scene
    rows_and_columns : int [n_p] array, les valeurs prises par chaque axe

    Return
    ------
    img : int [h, w, 3] array, la figure illustrant la memoire de chaque individu de la scene
    """
    index = ['%d' % rc for rc in rows_and_columns.tolist()]
    columns = ['%d' % rc for rc in rows_and_columns.tolist()]
    df = pd.DataFrame(heatmap, index=index, columns=columns)
    svm = sns.heatmap(df, annot=True, fmt=".2f", vmin=0, vmax=1, cmap='inferno')
    plt.title("Participants memory")
    plt.xlabel("Participants")
    plt.ylabel("Participants")
    fig = svm.get_figure()
    fig.tight_layout()
    img = fig2img(fig)
    return img

def save_memory_evolution(graphiques, id1, frameIdList, frameStep, filename):
    """
    Parameters
    ----------
    graphiques  : dict{key: int, value: dict{key: int, value: Tuple[list[float], list[int]]}}, evolution de la memoire de chaque individu de la scene
    id1         : int, le numero de l'individu pour lequel la courbe de la memoire sera enregistree
    frameIdList : list[int], la liste des frames parcourues
    frameStep   : int, le pas de frames
    filename    : str, le nom du fichier a sauvegarder
    """
    graphique = graphiques[id1]
    nbX, nbY = int(np.ceil(len(graphique)/3)), 3
    fig, axes = plt.subplots(nbX, nbY, figsize=(300*nbX/dpi, 250*nbY/dpi), sharex=True, sharey=True, constrained_layout=True)
    plt.setp(axes, xlim=(frameIdList[0], frameIdList[-1]), ylim=(-0.1, 1.1))
    colorPalette = getColorPalette(len(graphique), palette2='Dark2')
    for i, id2 in enumerate(graphique.keys()):
        x = graphique[id2][1]
        y = graphique[id2][0]
        x_tmp = [[x[0]]]
        y_tmp = [[y[0]]]
        if len(x) > 1:
            for j in range(1, len(x)):
                if x_tmp[-1][-1] == x[j]-frameStep:
                    x_tmp[-1].append(x[j])
                    y_tmp[-1].append(y[j])
                else:
                    x_tmp.append([x[j]])
                    y_tmp.append([y[j]])
        for (X,Y) in zip(x_tmp, y_tmp):
            if len(X) == 1:
                axes[(nbX-1)-i//nbY, i%nbY].plot(X[0], Y[0], color=colorPalette[i])
            else:
                axes[(nbX-1)-i//nbY, i%nbY].plot(X, Y, color=colorPalette[i])
            axes[(nbX-1)-i//nbY, i%nbY].set_title('memory('+str(id1)+','+str(id2)+')', fontsize=fontsize/1.34)
    fig.suptitle('Mémoire du participant '+str(id1))
    plt.setp(axes[-1, :], xlabel='Frames (time)')
    plt.setp(axes[:, 0], ylabel='Mémoire')
    fig.savefig(filename+'.png', dpi=dpi)
    plt.close(fig)
