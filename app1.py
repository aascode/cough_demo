import io
import librosa as ls
import streamlit as st
import yaml
import pandas as pd
from minio import Minio
from scipy.io import wavfile
from matplotlib import pyplot as plt

import detection


def get_config():
    with open('config.yaml', 'rb') as f:
        cfg = yaml.safe_load(f)
    return cfg


gmm_dectector = detection.GMM_Detector()

account = get_config()["account"]

# create a cline minio
minio_client = Minio(
    endpoint="localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)


def update_file():
    st.subheader("Update Data Cough")
    file = st.file_uploader("Upload file", accept_multiple_files=True)
    button = st.button("Send")

    # Make 'data-cough' bucket if not exist.
    found = minio_client.bucket_exists("data-cough")
    if not found:
        minio_client.make_bucket("data-cough")
    else:
        print("Bucket 'data-cough' already exists")

    if button:
        if file is None:
            st.error("File is empty")
        else:
            for f in file:
                minio_client.put_object(
                    bucket_name="data-cough",
                    object_name=f.name,
                    data=io.BytesIO(f.read()),
                    length=f.size
                )
            st.success("Send file successful, Thank you!!!")


def detection_classification():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader("Detectiong and Classification Cough")
    status = st.radio("Select : ", ('Example', 'Upload your file'))
    if status == 'Example':
        task = st.selectbox("Test", ["Cough 1", "Cough 2", "Cough 3"])
        if task == "Cough 1":
            button = st.button("Start")
            if button:
                audio = st.audio("data_test/Cough1.wav", format="audio/wav")
                rate, audio = wavfile.read("data_test/Cough1.wav")
                data_shape = audio.shape
                if len(data_shape) == 2:
                    audio = audio[:, 0]
                fig, ax = plt.subplots(figsize=(20, 4))
                ax.plot(audio)
                st.pyplot()
                segments = gmm_dectector.predict("data_test/Cough1.wav")

                list_start = []
                list_end = []
                list_class = []

                for start, end in segments:
                    dur = end - start
                    if dur <= 1:
                        list_start.append(start)
                        list_end.append(end)
                        list_class.append("Cough")
                    else:
                        list_start.append(start)
                        list_end.append(end)
                        list_class.append("Whooping")

                data = {'Time start': list_start, 'Time end': list_end, 'Classification Cough': list_class}
                df = pd.DataFrame(data, columns=["Time start", "Time end", "Classification Cough"])
                df

        elif task == "Cough 2":
            button = st.button("Start")
            if button:
                audio = st.audio("data_test/Cough2.wav", format="audio/wav")
                rate, audio = wavfile.read("data_test/Cough2.wav")
                data_shape = audio.shape
                if len(data_shape) == 2:
                    audio = audio[:, 0]
                fig, ax = plt.subplots(figsize=(20, 4))
                ax.plot(audio)
                st.pyplot()
                segments = gmm_dectector.predict("data_test/Cough2.wav")

                list_start = []
                list_end = []
                list_class = []

                for start, end in segments:
                    dur = end - start
                    if dur <= 1:
                        list_start.append(start)
                        list_end.append(end)
                        list_class.append("Cough")
                    else:
                        list_start.append(start)
                        list_end.append(end)
                        list_class.append("Whooping")

                data = {'Time start': list_start, 'Time end': list_end, 'Classification Cough': list_class}
                df = pd.DataFrame(data, columns=["Time start", "Time end", "Classification Cough"])
                df

        elif task == "Cough 3":
            button = st.button("Start")
            if button:
                audio = st.audio("data_test/Cough3.wav", format="audio/wav")
                rate, audio = wavfile.read("data_test/Cough3.wav")
                data_shape = audio.shape
                if len(data_shape) == 2:
                    audio = audio[:, 0]
                fig, ax = plt.subplots(figsize=(20, 4))
                ax.plot(audio)
                st.pyplot()
                segments = gmm_dectector.predict("data_test/Cough3.wav")

                list_start = []
                list_end = []
                list_class = []

                for start, end in segments:
                    dur = end - start
                    if dur <= 1:
                        list_start.append(start)
                        list_end.append(end)
                        list_class.append("Cough")
                    else:
                        list_start.append(start)
                        list_end.append(end)
                        list_class.append("Whooping")

                data = {'Time start': list_start, 'Time end': list_end, 'Classification Cough': list_class}
                df = pd.DataFrame(data, columns=["Time start", "Time end", "Classification Cough"])
                df
    else:
        file = st.file_uploader("Upload file", accept_multiple_files=False)
        button = st.button("Start")

        if button:
            if file is None:
                st.error("File is empty")
            else:
                st.success("Send file successful, Thank you!!!")


st.title("Cough Monitoring System")
st.subheader("Login Section")
username = st.sidebar.text_input("User Name")
password = st.sidebar.text_input("Password", type='password')
if st.sidebar.checkbox("Login"):
    result = username in account and account[username] == password
    if result:
        st.success("Login Successful!!! \n Welcome, {}".format(username))
        task = st.selectbox("Service", ["Update Data Cough", "Detection and Classification Cough", "Cough Statistics"])
        if task == "Update Data Cough":
            update_file()

        elif task == "Detection and Classification Cough":
            detection_classification()

        elif task == "Cough Statistics":
            st.subheader("User Profiles")
    else:
        st.warning("Incorrect Username/Password")
