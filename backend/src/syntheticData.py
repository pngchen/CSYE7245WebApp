import gcsfs
import h5py
import numpy as np
import pandas as pd


def lght_to_grid(data):
    """
    Converts SEVIR lightning data stored in Nx5 matrix to an LxLx49 tensor representing
    flash counts per pixel per frame

    Parameters
    ----------
    data  np.array
       SEVIR lightning event (Nx5 matrix)

    Returns
    -------
    np.array
       LxLx49 tensor containing pixel counts
    """
    FRAME_TIMES = np.arange(-120.0, 125.0, 5) * 60  # in seconds
    out_size = (48, 48, len(FRAME_TIMES))
    if data.shape[0] == 0:
        return np.zeros(out_size, dtype=np.float32)

    # filter out points outside the grid
    x, y = data[:, 3], data[:, 4]
    m = np.logical_and.reduce([x >= 0, x < out_size[0], y >= 0, y < out_size[1]])
    data = data[m, :]
    if data.shape[0] == 0:
        return np.zeros(out_size, dtype=np.float32)

    # Filter/separate times
    # compute z coodinate based on bin locaiton times
    t = data[:, 0]
    z = np.digitize(t, FRAME_TIMES) - 1
    z[z == -1] = 0  # special case:  frame 0 uses lght from frame 1

    x = data[:, 3].astype(np.int64)
    y = data[:, 4].astype(np.int64)

    k = np.ravel_multi_index(np.array([y, x, z]), out_size)
    n = np.bincount(k, minlength=np.prod(out_size))
    return np.reshape(n, out_size).astype(np.float32)

def get_data(idx):
    FS = gcsfs.GCSFileSystem(project="Assignment1",
                         token="hardy-portal-318606-3c8e02bd3a5d.json")

    data_file = FS.open("gs://assignment1-data/CATALOG_cleaned.csv", 'rb')
    catalog = pd.read_csv(data_file,low_memory=False)

    events = catalog.groupby('id')
    event_ids = list(events.groups.keys())
    event_ids = [i for i in event_ids if i.startswith("S")]
    event_ids.sort()

    y_test = {"vil":[]}
    x_test = {"ir107":[], "ir069":[], "lght":[]}
    dataSource = zip(events.get_group(event_ids[idx])["id"],
                     events.get_group(event_ids[idx])["file_name"],
                     events.get_group(event_ids[idx])["file_index"],
                     events.get_group(event_ids[idx])["img_type"])

    for id, fn, fi, it in dataSource:
        if it == "vil":
            print("Y: ", it)
            data_file = FS.open(f"gs://assignment1-data/data/{fn}", 'rb')
            data = h5py.File(data_file, 'r')

            temp = []
            for i in range(49):
                temp.append(data[it][fi][:, :, i:i + 1])
            temp = np.array(temp)

            y_test["vil"] = temp

        elif it == "lght":
            print("X: ", it)
            data_file = FS.open(f"gs://assignment1-data/data/{fn}", 'rb')
            data = h5py.File(data_file, 'r')
            lghtData = lght_to_grid(data[id][:])

            temp = []
            for i in range(49):
                temp.append(lghtData[:, :, i:i + 1])
            temp = np.array(temp)

            x_test[it] = temp

        elif it != "vis":
            print("X: ", it)
            data_file = FS.open(f"gs://assignment1-data/data/{fn}", 'rb')
            data = h5py.File(data_file, 'r')

            temp = []
            for i in range(49):
                temp.append(data[it][fi][:, :, i:i + 1])
            temp = np.array(temp)

            x_test[it] = temp

    return x_test, y_test