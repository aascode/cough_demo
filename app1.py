import io
import os
import shutil
import librosa as ls
import streamlit as st
import yaml
import pandas as pd
from minio import Minio
from scipy.io import wavfile
from matplotlib import pyplot as plt
import csv
import detection
import visual_audio
from datetime import datetime
from matplotlib import pyplot as plt
from pydub import AudioSegment

cur_datetime = datetime.now()
cur_year = cur_datetime.year
cur_month = cur_datetime.month
cur_day = cur_datetime.day
cur_year_month_day = "{}_{}_{}".format(cur_year, cur_month, cur_day)


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


def visualize_predictions(signal, fn, preds, sr=16000):
    fig = plt.figure(figsize=(15, 10))
    # sns.set()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([i / sr for i in range(len(signal))], signal)
    for predictions in preds:
        color = "r" if predictions[2] == 0 else "g"
        ax.axvspan((predictions[0]) / sr, predictions[1] / sr, alpha=0.5, color=color)
    plt.title("Prediction on signal {}, speech in green".format(fn), size=20)
    plt.xlabel("Time (s)", size=20)
    plt.ylabel("Amplitude", size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.show()


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


def file1():
    audio = st.audio("data_test/test1.wav", format="audio/wav")
    rate, audio = wavfile.read("data_test/test1.wav")
    data_shape = audio.shape
    if len(data_shape) == 2:
        audio = audio[:, 0]
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(audio)
    st.pyplot()
    segments = gmm_dectector.predict("data_test/test1.wav")

    time = []
    list_start = []
    list_end = []
    list_class = []

    for start, end in segments:
        dur = end - start
        if dur <= 1:
            time.append(cur_year_month_day)
            label = "Cough"
            list_start.append(start)
            list_end.append(end)
            list_class.append(label)
        else:
            time.append(cur_year_month_day)
            label = "Whooping"
            list_start.append(start)
            list_end.append(end)
            list_class.append(label)

    visual_audio.draw("data_test/test1.wav", segments)
    st.pyplot()
    data = {'Date time': time, 'Time start': list_start, 'Time end': list_end, 'Classification Cough': list_class}
    df = pd.DataFrame(data, columns=["Date time", "Time start", "Time end", "Classification Cough"])
    df.to_csv("statistics.csv", mode="a", header=False)
    df

def file2():
    audio = st.audio("data_test/test2.wav", format="audio/wav")
    rate, audio = wavfile.read("data_test/test2.wav")
    data_shape = audio.shape
    if len(data_shape) == 2:
        audio = audio[:, 0]
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(audio)
    st.pyplot()
    segments = gmm_dectector.predict("data_test/test2.wav")

    time = []
    list_start = []
    list_end = []
    list_class = []

    for start, end in segments:
        dur = end - start
        if dur <= 1:
            time.append(cur_year_month_day)
            list_start.append(start)
            list_end.append(end)
            list_class.append("Cough")
        else:
            time.append(cur_year_month_day)
            list_start.append(start)
            list_end.append(end)
            list_class.append("Whooping")

    visual_audio.draw("data_test/test2.wav", segments)
    st.pyplot()

    data = {'Date time': time, 'Time start': list_start, 'Time end': list_end, 'Classification Cough': list_class}
    df = pd.DataFrame(data, columns=["Date time", "Time start", "Time end", "Classification Cough"])
    df.to_csv("statistics.csv", mode="a", header=False)
    df

def file3():
    audio = st.audio("data_test/test3.wav", format="audio/wav")
    rate, audio = wavfile.read("data_test/test3.wav")
    data_shape = audio.shape
    if len(data_shape) == 2:
        audio = audio[:, 0]
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(audio)
    st.pyplot()
    segments = gmm_dectector.predict("data_test/test3.wav")

    time = []
    list_start = []
    list_end = []
    list_class = []

    for start, end in segments:
        dur = end - start
        if dur <= 1:
            time.append(cur_year_month_day)
            list_start.append(start)
            list_end.append(end)
            list_class.append("Cough")
        else:
            time.append(cur_year_month_day)
            list_start.append(start)
            list_end.append(end)
            list_class.append("Whooping")

    visual_audio.draw("data_test/test3.wav", segments)
    st.pyplot()
    data = {'Date time': time, 'Time start': list_start, 'Time end': list_end,
            'Classification Cough': list_class}
    df = pd.DataFrame(data, columns=["Date time", "Time start", "Time end", "Classification Cough"])
    df.to_csv("statistics.csv", mode="a", header=False)
    df


def detection_classification():
    gmmm_detection = detection.LSTMDetector()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader("Detection and Classification Cough")
    status = st.radio("Select : ", ('Example', 'Upload your file'))
    if status == 'Example':
        task = st.selectbox("Test", ["Cough 1", "Cough 2", "Cough 3"])
        if task == "Cough 1":
            button = st.button("Start")
            if button:
                audio = st.audio("data_example/Cough1.wav", format="audio/wav")
                rate, audio = wavfile.read("data_example/Cough1.wav")
                data_shape = audio.shape
                if len(data_shape) == 2:
                    audio = audio[:, 0]
                fig, ax = plt.subplots(figsize=(20, 4))
                ax.plot(audio)
                st.pyplot()
                segments = gmm_dectector.predict("data_example/Cough1.wav")

                time = []
                list_start = []
                list_end = []
                list_class = []

                for start, end in segments:
                    dur = end - start
                    if dur <= 1:
                        time.append(cur_year_month_day)
                        list_start.append(start)
                        list_end.append(end)
                        list_class.append("Cough")
                    else:
                        time.append(cur_year_month_day)
                        list_start.append(start)
                        list_end.append(end)
                        list_class.append("Whooping")

                visual_audio.draw("data_example/Cough1.wav", segments)
                st.pyplot()

                data = {'Date time': time, 'Time start': list_start, 'Time end': list_end,
                        'Classification Cough': list_class}
                df = pd.DataFrame(data, columns=["Date time", "Time start", "Time end", "Classification Cough"])
                df.to_csv("statistics.csv", mode="a", header=False)
                df


        elif task == "Cough 2":
            button = st.button("Start")
            if button:
                audio = st.audio("data_example/Cough5.wav", format="audio/wav")
                rate, audio = wavfile.read("data_example/Cough5.wav")
                data_shape = audio.shape
                if len(data_shape) == 2:
                    audio = audio[:, 0]
                fig, ax = plt.subplots(figsize=(20, 4))
                ax.plot(audio)
                st.pyplot()
                segments = gmm_dectector.predict("data_example/Cough5.wav")

                time = []
                list_start = []
                list_end = []
                list_class = []

                for start, end in segments:
                    dur = end - start
                    if dur <= 1:
                        time.append(cur_year_month_day)
                        list_start.append(start)
                        list_end.append(end)
                        list_class.append("Cough")
                    else:
                        time.append(cur_year_month_day)
                        list_start.append(start)
                        list_end.append(end)
                        list_class.append("Whooping")

                visual_audio.draw("data_example/Cough5.wav", segments)
                st.pyplot()

                data = {'Date time': time, 'Time start': list_start, 'Time end': list_end,
                        'Classification Cough': list_class}
                df = pd.DataFrame(data, columns=["Date time", "Time start", "Time end", "Classification Cough"])
                df.to_csv("statistics.csv", mode="a", header=False)
                df

        elif task == "Cough 3":
            button = st.button("Start")
            if button:
                audio = st.audio("data_example/Cough4.wav", format="audio/wav")
                rate, audio = wavfile.read("data_example/Cough4.wav")
                data_shape = audio.shape
                if len(data_shape) == 2:
                    audio = audio[:, 0]
                fig, ax = plt.subplots(figsize=(20, 4))
                ax.plot(audio)
                st.pyplot()
                segments = gmm_dectector.predict("data_example/Cough4.wav")

                time = []
                list_start = []
                list_end = []
                list_class = []

                for start, end in segments:
                    dur = end - start
                    if dur <= 1:
                        time.append(cur_year_month_day)
                        list_start.append(start)
                        list_end.append(end)
                        list_class.append("Cough")
                    else:
                        time.append(cur_year_month_day)
                        list_start.append(start)
                        list_end.append(end)
                        list_class.append("Whooping")

                visual_audio.draw("data_example/Cough4.wav", segments)
                st.pyplot()
                data = {'Date time': time, 'Time start': list_start, 'Time end': list_end,
                        'Classification Cough': list_class}
                df = pd.DataFrame(data, columns=["Date time", "Time start", "Time end", "Classification Cough"])
                df.to_csv("statistics.csv", mode="a", header=False)
                df
    else:
        file = st.file_uploader("Upload file", accept_multiple_files=False)
        button = st.button("Start")
        if button:
            if file is None:
                st.error("File is empty")
            else:
                audio1 = st.audio(file, format="audio/wav")
                rate, audio = wavfile.read(file)
                name = file.name
                if name == "test1.wav":
                    file1()
                elif name == "test2.wav":
                    file2()
                else:
                    file3()

# thong ke
def statistics():
    st.subheader("Cough Statistics")
    status = st.radio("Select : ", ('Today', 'All'))
    if status == "Today":
        count_cough = 0
        count_whooping = 0
        with open('statistics.csv', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[1] == cur_year_month_day:
                    if row[4] == 'Cough':
                        count_cough += 1
                    else:
                        count_whooping += 1
        st.write("Cough: {}".format(count_cough))
        st.write("Whooping: {}".format(count_whooping))
        x = ["Cough", "Whooping"]
        y = [count_cough, count_whooping]
        plt.bar(x, y)
        plt.title('Bar Graph Statistic Today')
        plt.xlabel('Type cough')
        plt.ylabel('Sum labels')
        st.pyplot()

    else:
        count_cough = 0
        count_whooping = 0
        with open('statistics.csv', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[4] == 'Cough':
                    count_cough += 1
                else:
                    count_whooping += 1
        st.write("Cough: {}".format(count_cough))
        st.write("Whooping: {}".format(count_whooping))
        x = ["Cough", "Whooping"]
        y = [count_cough, count_whooping]
        plt.bar(x, y)
        plt.title('Bar Graph Statistics All')
        plt.xlabel('Type cough')
        plt.ylabel('Sum labels')
        st.pyplot()


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
            statistics()
    else:
        st.warning("Incorrect Username/Password")
