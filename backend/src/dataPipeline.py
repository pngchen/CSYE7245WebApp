import h5py
import numpy as np
import gcsfs
import pickle

def split_data(array, res):
    for i in range(25):
        temp = np.dsplit(array, np.array([i, i + 13, i + 25]))
        res['IN'].append(temp[1])
        res['OUT'].append(temp[2])

def run(pklname):
    # DATA_PATH = file

    FS = gcsfs.GCSFileSystem(project="Assignment1",
                            token="./hardy-portal-318606-3c8e02bd3a5d.json")

    # print(end)
    # with FS.open(f'gs://assignment1-data/data/vil/2018/{DATA_PATH}.h5', 'rb') as data_file:
    #     data = h5py.File(data_file, 'r')
    #     s = np.s_[0:end:1]
    #     vil = data['vil'][s]

    with FS.open(f'gs://assignment1-data/res/pkl/{pklname}', 'rb') as pklRes:
        vil = pickle.load(pklRes)

    res = {'IN':[],"OUT":[]}
    # for i in vil:
    #     split_data(i, res)
    split_data(vil, res)

    res["IN"] = np.array(res["IN"])
    res["OUT"] = np.array(res["OUT"])

    res["IN"] = (res["IN"].astype(np.float32) - 33.44) / 47.54
    res["OUT"] = (res["OUT"].astype(np.float32) - 33.44) / 47.54

    return res["IN"], res["OUT"]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run pipeline on the cloud')
    parser.add_argument('-f','--file', help='File Path', default='SEVIR_VIL_STORMEVENTS_2018_0101_0630')
    parser.add_argument('-e','--end', help='End', default=1)
    args = vars(parser.parse_args())

    IN, OUT = run(file=args['file'], end=int(args['end']))

    with h5py.File("data.h5", 'w') as data:
       data['IN'] = IN
       data['OUT'] = OUT

   # with h5py.File("data.h5", 'r') as data:
   #      IN = data["IN"]
   #      OUT = data["OUT"]
    print("The shape of IN: " + str(IN.shape))
    print("The shape of OUT: " + str(OUT.shape))