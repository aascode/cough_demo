import os
import pickle
import torch
import torchaudio
from scipy.io import wavfile
from torchaudio.transforms import MFCC
import numpy as np

from generate_segment import generate_segment, remove_small


class Extractor:
    def __init__(self):
        sample_rate = 16000
        self.mfcc = MFCC(sample_rate=sample_rate, n_mfcc=40,
                         melkwargs={'win_length': int(0.025 * sample_rate),
                                    'hop_length': int(0.010 * sample_rate),
                                    'n_fft': int(0.025 * sample_rate)})

    def extract_feature(self, audio):
        shape = audio.size()
        audio = audio.reshape(-1, shape[-1])
        feature = self.mfcc(audio)
        feature = feature.reshape(shape[:-1] + feature.shape[-2:])[:, 0]
        feature = feature.transpose(1, 2)
        return feature


def batching(audio, chunk, batch_size=1):
    idx = 0
    num_sample = chunk * 16000 // 1000 * batch_size
    while idx + num_sample <= audio.shape[1]:
        yield audio[:, idx: idx + num_sample].reshape(batch_size, audio.shape[0], -1)
        idx += num_sample


class GMM_Detector:
    def __init__(self):
        self.chunk = 100
        self.extractor = Extractor()
        self.model_0 = pickle.load(open("model/non_cough.pkl", 'rb'))
        self.model_1 = pickle.load(open("model/cough.pkl", 'rb'))

    def predict(self, path):
        audio, sr = torchaudio.load(path)
        all_pred = []
        for batch in batching(audio, self.chunk):
            feature = self.extractor.extract_feature(batch)
            feature = feature.reshape(feature.shape[0], -1)
            score_0 = self.model_0.score_samples(feature)
            score_1 = self.model_1.score_samples(feature)
            for i in range(len(score_1)):
                if score_0[i] > score_1[i]:
                    all_pred.append(0)
                else:
                    all_pred.append(1)

        all_pred.append(0)
        segment = []
        start = -1
        end = -1
        for i, y in enumerate(all_pred):
            if y == 1 and (i == 0 or all_pred[i - 1] == 0):
                start = i

            if y == 0 and i > 0 and all_pred[i - 1] == 1:
                end = i

            if start != -1 and end != -1:
                segment.append([start * self.chunk / 1000, end * self.chunk / 1000])
                start = -1
                end = -1

        segment = remove_small(segment)
        return segment


    def predict_audio(self, file):
        sr, audio  = wavfile.read(file)
        all_pred = []
        for batch in batching(audio, self.chunk):
            feature = self.extractor.extract_feature(batch)
            feature = feature.reshape(feature.shape[0], -1)
            score_0 = self.model_0.score_samples(feature)
            score_1 = self.model_1.score_samples(feature)
            for i in range(len(score_1)):
                if score_0[i] > score_1[i]:
                    all_pred.append(0)
                else:
                    all_pred.append(1)

        all_pred.append(0)
        segment = []
        start = -1
        end = -1
        for i, y in enumerate(all_pred):
            if y == 1 and (i == 0 or all_pred[i - 1] == 0):
                start = i

            if y == 0 and i > 0 and all_pred[i - 1] == 1:
                end = i

            if start != -1 and end != -1:
                segment.append([start * self.chunk / 1000, end * self.chunk / 1000])
                start = -1
                end = -1

        segment = remove_small(segment)
        return segment


def padding(feature, batch_size, seq_size):
    feature = [x.tolist() for x in feature]
    data_len = len(feature)
    old_data_len = data_len
    assert len(feature) == data_len
    if data_len % seq_size != 0:
        padding_len = seq_size - (data_len % seq_size)
        feature = feature + [feature[-1]] * padding_len

    data_len = len(feature)
    feature = np.array(feature)
    data = []
    idx = 0
    while idx < data_len:
        if idx + batch_size * seq_size < data_len:
            data.append(feature[idx: idx + batch_size * seq_size])
            idx += batch_size * seq_size
        else:
            data.append(feature[idx:])
            idx = data_len

    return data, old_data_len


class LSTMDetector:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.seq_size = 20
        self.batch_size = 32
        sample_rate = 16000
        model_path = "model/model_lstm.pt"

        self.model = torch.load(model_path).to(self.device)
        self.model.eval()

        self.mfcc_ft = MFCC(sample_rate=sample_rate, n_mfcc=40,
                            melkwargs={'win_length': int(0.025 * sample_rate),
                                       'hop_length': int(0.010 * sample_rate),
                                       'n_fft': int(0.025 * sample_rate)}).to(self.device)

    def predict(self, path):
        audio, sr = torchaudio.load(path)
        audio = audio.to(self.device)
        feature = self.mfcc_ft(audio)[0].T
        feature = feature.cpu().detach().numpy()

        dataset, data_len = padding(feature, self.batch_size, self.seq_size)
        output = []
        for data in dataset:
            feature = data
            feature = torch.from_numpy(feature).float()
            feature = feature.to(self.device)
            with torch.no_grad():
                y_hat = self.model(feature)

            y_hat = torch.softmax(y_hat, dim=1)

            for y in y_hat:
                if y[0] >= 0.5:
                    output.append(0)
                else:
                    output.append(1)

        output = output[: data_len]
        segments, segments_raw = generate_segment(output)
        return segments_raw


if __name__ == '__main__':
    detector = LSTMDetector()
    print(detector.predict("/home/chiendb/Desktop/cough_classification/Data_Cough/audio/Cough1.wav"))
