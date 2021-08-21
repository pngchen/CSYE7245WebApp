# backend/main.py

import sys
sys.path.append('./src/')
import uvicorn
import gcsfs
import tensorflow as tf
import h5py

import config
import nowcast
import synthetic
import dataPipeline
import syntheticData
from flask import escape


# app = FastAPI()

# @app.exception_handler(StarletteHTTPException)
# async def http_exception_handler(request, exc):
#     return PlainTextResponse(str(exc.detail), status_code=exc.status_code)

# @app.get("/")
def read_root(request):
    return {"message": "Welcome from the nowcast API"}

# @app.post("/synthetic/{modelName}/{idx}")
def get_syn(request):
    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'modelName' in request_json and 'idx' in request_json:
        modelName = request_json['modelName']
        idx = int(request_json['idx'])
    elif request_args and 'modelName' in request_args and 'idx' in request_args:
        modelName = request_args['modelName']
        idx = int(request_args['idx'])

    FS = gcsfs.GCSFileSystem(project="Assignment1",
                             token="hardy-portal-318606-3c8e02bd3a5d.json")
    model = config.synthetics[modelName]

    with FS.open(f'gs://assignment1-data/models/synrad/{model}.h5', 'rb') as model_file:
        model_gcs = h5py.File(model_file, 'r')
        model = tf.keras.models.load_model(model_gcs, compile=False, custom_objects={"tf": tf})

    x_test, y_test = syntheticData.get_data(idx)

    y_pred = synthetic.run_synrad(model, x_test)

    imgname, pklname = synthetic.main(modelName, x_test, y_test, y_pred)

    return {"imgname": imgname, "pklname": pklname}

# @app.post("/nowcast/{modelName}/{idx}")
def get_nowcast(request):
    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'modelName' in request_json and 'pklname' in request_json and 'idx' in request_json:
        modelName = request_json['modelName']
        pklname = request_json['pklname']
        idx = int(request_json['idx'])
    elif request_args and 'modelName' in request_args and 'pklname' in request_args and 'idx' in request_args:
        modelName = request_args['modelName']
        pklname = request_args['pklname']
        idx = int(request_args['idx'])

    FS = gcsfs.GCSFileSystem(project="Assignment1",
                             token="hardy-portal-318606-3c8e02bd3a5d.json")
    model = config.models[modelName]

    with FS.open(f'gs://assignment1-data/models/nowcast/{model}.h5', 'rb') as model_file:
        model_gcs = h5py.File(model_file, 'r')
        model = tf.keras.models.load_model(model_gcs, compile=False, custom_objects={"tf": tf})

    x_test, y_test = dataPipeline.run(pklname)

    name = nowcast.visualize_result(model, x_test, y_test, idx, modelName)

    return {"name": name}


# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8085)