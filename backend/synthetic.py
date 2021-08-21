import sys
import uuid
sys.path.append('./src/')
import gcsfs
import pickle
import numpy as np
import matplotlib.pyplot as plt
from display.display import get_cmap

def run_synrad(model,x_test,batch_size=32):
    return model.predict([x_test[k] for k in ['ir069','ir107','lght']],batch_size=batch_size)

def visualize_result(x_test, y_test,y_preds,idx,ax):
    cmap_dict = lambda s: {'cmap':get_cmap(s,encoded=True)[0], 'norm':get_cmap(s,encoded=True)[1],
                           'vmin':get_cmap(s,encoded=True)[2], 'vmax':get_cmap(s,encoded=True)[3]}
    ax[0].imshow(x_test['ir069'][idx,:,:,0],**cmap_dict('ir069'))
    ax[1].imshow(x_test['ir107'][idx,:,:,0],**cmap_dict('ir107'))
    ax[2].imshow(x_test['lght'][idx,:,:,0],cmap='hot',vmin=0,vmax=10)
    ax[3].imshow(y_test['vil'][idx,:,:,0],**cmap_dict('vil'))

    if isinstance(y_preds,(list,)):
        yp=y_preds[0]
    else:
        yp=y_preds
    ax[4].imshow(yp[idx,:,:,0],**cmap_dict('vil'))

    for i in range(len(ax)):
        ax[i].xaxis.set_ticks([])
        ax[i].yaxis.set_ticks([])

def main(modelName, x_test, y_test, y_preds):
    FS = gcsfs.GCSFileSystem(project="Assignment1",
                         token="hardy-portal-318606-3c8e02bd3a5d.json")
                         
    test_idx = [15, 30, 45]
    N = len(test_idx)
    fig, ax = plt.subplots(N, 5, figsize=(12, 4))
    for k, i in enumerate(test_idx):
        visualize_result(x_test, y_test, y_preds, i, ax[k])

    ax[0][0].set_title('Input ir069')
    ax[0][1].set_title('Input ir107')
    ax[0][2].set_title('Input lght')
    ax[0][3].set_title('Truth')
    ax[0][4].set_title(f'Output\n{modelName} Loss')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.05,
                        wspace=0.35)

    if isinstance(y_preds,(list,)):
        y_pred=y_preds[0]
    else:
        y_pred=y_preds
    print(y_pred.shape)

    res = y_pred[0]
    for i in range(1, 49):
        res = np.insert(res, i, y_pred[i][:, :, 0], axis=2)

    pklname = f"{str(uuid.uuid4())}.pkl"
    with FS.open(f'gs://assignment1-data/res/pkl/{pklname}', 'wb') as pklRes:
        pickle.dump(res, pklRes)

    # name = f"/storage/{str(uuid.uuid4())}.png"
    imgname = f"{str(uuid.uuid4())}.png"
    imgRes = FS.open(f'gs://assignment1-data/res/img/{imgname}', 'wb')
    fig.savefig(imgRes, bbox_inches='tight')

    return imgname, pklname