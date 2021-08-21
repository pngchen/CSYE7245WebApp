# frontend/main.py

import gcsfs
import requests
import streamlit as st
from PIL import Image

models = {
    "MSE": "mse_model",
    "SC": "style_model",
    "MSE_SC": "mse_and_style",
    "cGAN_MAE": "gan_generator",
}

synthetics = {
    "GAN_MAE": "gan_mae_weights",
    "MSE_VGG": "mse_vgg_weights",
    "MSE": "mse_weights"
}

FS = gcsfs.GCSFileSystem(project="Assignment1",
                         token="hardy-portal-318606-3c8e02bd3a5d.json")

# https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Web App")

application = st.selectbox("Choose a application ", ["Synthetic", "Nowcast", "View Image"])

# global pklRes

if application == "Synthetic":
    idx = st.text_input('Index (0 - 235)', '')
    model = st.selectbox("Choose the model", [i for i in synthetics.keys()])

    if st.button("Synthetic"):
        if idx != '' and model is not None:
            res = requests.post(f"https://us-central1-hardy-portal-318606.cloudfunctions.net/get_syn?modelName={model}&idx={idx}", headers={"Connection": "close"})
            res_path = res.json()
            imgRes = FS.open(f'gs://assignment1-data/res/img/{res_path.get("imgname")}', 'rb')
            pklRes = res_path.get("pklname")
            image = Image.open(imgRes)
            st.write("Result of Synthetic")
            st.write(pklRes)
            st.write(res_path.get("imgname"))
            st.image(image, width=750)

if application == "Nowcast":
    haveVilFile = st.radio("Already have a VIL file?", ('Yes', 'No'))

    if haveVilFile == "No":
        idx_syn = st.text_input('Index (0 - 235)', '')
        model_syn = st.selectbox("Choose a Synthetic model", [i for i in synthetics.keys()])
    else:
        pklRes = st.text_input('VIL file name', '')

    idx = st.slider("Choose the Index", 0, 24)

    # displays the select widget for the styles
    model = st.selectbox("Choose a Nowcast model", [i for i in models.keys()])

    if st.button("Nowcast"):
        if haveVilFile == "Yes" and pklRes != '' and model is not None:
            res = requests.post(
                f"https://us-central1-hardy-portal-318606.cloudfunctions.net/get_nowcast?modelName={model}&pklname={pklRes}&idx={idx}",
                headers={"Connection": "close"})
            img_path = res.json()
            imgRes = FS.open(f'gs://assignment1-data/res/img/{img_path.get("name")}', 'rb')
            image = Image.open(imgRes)
            st.write("Result of Nowcast")
            st.write(img_path.get("name"))
            st.image(image, width=750)

        elif idx_syn != '' and model_syn is not None and model is not None:
            res_syn = requests.post(
                f"https://us-central1-hardy-portal-318606.cloudfunctions.net/get_syn?modelName={model_syn}&idx={idx_syn}",
                headers={"Connection": "close"})
            res_path = res_syn.json()
            imgRes = FS.open(f'gs://assignment1-data/res/img/{res_path.get("imgname")}', 'rb')
            pklRes = res_path.get("pklname")
            image = Image.open(imgRes)
            st.write("Result of Synthetic")
            st.write(pklRes)
            st.write(res_path.get("imgname"))
            st.image(image, width=750)

            res = requests.post(f"https://us-central1-hardy-portal-318606.cloudfunctions.net/get_nowcast?modelName={model}&pklname={pklRes}&idx={idx}", headers={"Connection": "close"})
            img_path = res.json()
            imgRes = FS.open(f'gs://assignment1-data/res/img/{img_path.get("name")}', 'rb')
            image = Image.open(imgRes)
            st.write("Result of Nowcast")
            st.write(img_path.get("name"))
            st.image(image, width=750)

if application == "View Image":
    imgName = st.text_input('Image file name', '')
    imgRes = FS.open(f'gs://assignment1-data/res/img/{imgName}', 'rb')
    image = Image.open(imgRes)
    st.image(image, width=750)