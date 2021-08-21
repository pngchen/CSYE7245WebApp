# backend/nowcast.py

import uuid
import sys
sys.path.append('./src/')
import gcsfs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from display.display import get_cmap


# Model that implements persistence forecast that just repeasts last frame of input
class persistence:
    def predict(self, x_test):
        return np.tile(x_test[:, :, :, -1:], [1, 1, 1, 12])


def plot_hit_miss_fa(ax, y_true, y_pred, thres):
    hmf_colors = np.array([
        [82, 82, 82],
        [252, 141, 89],
        [255, 255, 191],
        [145, 191, 219]
    ]) / 255

    mask = np.zeros_like(y_true)
    mask[np.logical_and(y_true >= thres, y_pred >= thres)] = 4
    mask[np.logical_and(y_true >= thres, y_pred < thres)] = 3
    mask[np.logical_and(y_true < thres, y_pred >= thres)] = 2
    mask[np.logical_and(y_true < thres, y_pred < thres)] = 1
    cmap = ListedColormap(hmf_colors)
    ax.imshow(mask, cmap=cmap)


# def visualize_result(modelName, datapath, idx):
def visualize_result(model, x_test, y_test, idx, modelName):
    # x_test, y_test = dataPipeline.run(datapath, 1)
    FS = gcsfs.GCSFileSystem(project="Assignment1",
                         token="hardy-portal-318606-3c8e02bd3a5d.json")

    # MODEL_PATH = "./models/nowcast/"

    norm = {'scale': 47.54, 'shift': 33.44}
    hmf_colors = np.array([
        [82, 82, 82],
        [252, 141, 89],
        [255, 255, 191],
        [145, 191, 219]
    ]) / 255

    fig, ax = plt.subplots(4, 7, figsize=(20, 8), gridspec_kw={'width_ratios': [1, .2, 1, .2, 1, 1, 1]})

    # with FS.open(f'gs://assignment1-data/models/nowcast/{modelName}.h5', 'rb') as model_file:
    #     model_gcs = h5py.File(model_file, 'r')
    #     model = tf.keras.models.load_model(model_gcs, compile=False, custom_objects={"tf": tf})
    # mse_file = './models/nowcast/mse_model.h5'
    # model = tf.keras.models.load_model(mse_file, compile=False, custom_objects={"tf": tf})

    fs = 10
    cmap_dict = lambda s: {'cmap': get_cmap(s, encoded=True)[0],
                           'norm': get_cmap(s, encoded=True)[1],
                           'vmin': get_cmap(s, encoded=True)[2],
                           'vmax': get_cmap(s, encoded=True)[3]}

    for i in range(1, 13, 3):
        xt = x_test[idx, :, :, i] * norm['scale'] + norm['shift']
        ax[(i - 1) // 3][0].imshow(xt, **cmap_dict('vil'))
    ax[0][0].set_title('Inputs', fontsize=fs)

    pers = persistence().predict(x_test[idx:idx + 1])
    pers = pers * norm['scale'] + norm['shift']
    x_test = x_test[idx:idx + 1]
    y_test = y_test[idx:idx + 1] * norm['scale'] + norm['shift']

    yp = model.predict(x_test)
    if isinstance(yp, (list,)):
        yp = yp[0]
    y_pred = yp * norm['scale'] + norm['shift']

    for i in range(0, 12, 3):
        ax[i // 3][2].imshow(y_test[0, :, :, i], **cmap_dict('vil'))
    ax[0][2].set_title('Target', fontsize=fs)

    # Plot Persistence
    for i in range(0, 12, 3):
        plot_hit_miss_fa(ax[i // 3][4], y_test[0, :, :, i], pers[0, :, :, i], 74)
    ax[0][4].set_title('Persistence\nScores', fontsize=fs)

    for i in range(0, 12, 3):
        ax[i // 3][5].imshow(y_pred[0, :, :, i], **cmap_dict('vil'))
        plot_hit_miss_fa(ax[i // 3][5 + 1], y_test[0, :, :, i], y_pred[0, :, :, i], 74)

    ax[0][5].set_title(modelName, fontsize=fs)
    ax[0][5 + 1].set_title(modelName + '\nScores', fontsize=fs)

    for j in range(len(ax)):
        for i in range(len(ax[j])):
            ax[j][i].xaxis.set_ticks([])
            ax[j][i].yaxis.set_ticks([])
    for i in range(4):
        ax[i][1].set_visible(False)
    for i in range(4):
        ax[i][3].set_visible(False)
    ax[0][0].set_ylabel('-45 Minutes')
    ax[1][0].set_ylabel('-30 Minutes')
    ax[2][0].set_ylabel('-15 Minutes')
    ax[3][0].set_ylabel('  0 Minutes')
    ax[0][2].set_ylabel('+15 Minutes')
    ax[1][2].set_ylabel('+30 Minutes')
    ax[2][2].set_ylabel('+45 Minutes')
    ax[3][2].set_ylabel('+60 Minutes')

    legend_elements = [Patch(facecolor=hmf_colors[1], edgecolor='k', label='False Alarm'),
                       Patch(facecolor=hmf_colors[2], edgecolor='k', label='Miss'),
                       Patch(facecolor=hmf_colors[3], edgecolor='k', label='Hit')]
    ax[-1][-1].legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(-5.4, -.35),
                      ncol=5, borderaxespad=0, frameon=False, fontsize='16')
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    # name = f"/storage/{str(uuid.uuid4())}.png"
    name = f"{str(uuid.uuid4())}.png"
    imgRes = FS.open(f'gs://assignment1-data/res/img/{name}', 'wb')
    fig.savefig(imgRes, bbox_inches='tight')

    return name