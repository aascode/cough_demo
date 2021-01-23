import os
import random
import numpy as np
import torch
from torchaudio.transforms import MFCC
import config
import torchaudio
from torch.utils.data import Dataset
import librosa
import numpy as np


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


def read_data(data_dir):
    audio_dir = os.path.join(data_dir, "audio")
    label_dir = os.path.join(data_dir, "label")
    feature_silent = []
    feature_cough = []

    for filename in os.listdir(audio_dir):
        audio_path = os.path.join(audio_dir, filename)
        label_path = os.path.join(label_dir, filename.replace(".wav", ".txt"))
        # audio, sr = torchaudio.load(audio_path)
        audio, sr = librosa.load(audio_path, sr=16000)
        gfcc_feature = gfcc(
            sig=audio,
            fs=sr,
            hop_length=int(0.010 * sr),
            n_fft=int(0.025 * sr),
            num_ceps=13)

        feature_size = gfcc_feature.shape[0]

        mfcc_feature = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            hop_length=int(0.010 * sr),
            n_fft=int(0.025 * sr),
            n_mfcc=40).transpose(1, 0)[: feature_size]

        zcr_feature = librosa.feature.zero_crossing_rate(
            y=audio,
            hop_length=int(0.010 * sr),
            n_fft=int(0.025 * sr),).transpose(1, 0)[: feature_size]

        feature = np.concatenate((mfcc_feature, gfcc_feature, zcr_feature), axis=1)

        segment_silent = []
        segment_cough = []

        last = 0
        with open(label_path, "r") as f:
            for line in f.readlines():
                start, end, _ = line.split()
                start, end = int(float(start) * 1000) // 10, int(float(end) * 1000) // 10
                segment_silent.append((max(0, last), min(start, feature_size)))
                segment_cough.append((max(0, start), min(end, feature_size)))
                last = end

            segment_silent.append((max(last, 0), feature_size))

        for start, end in segment_silent:
            chunk_sample = config.CHUNK // 10
            if end - start >= chunk_sample:
                for x in range(start, end - chunk_sample, chunk_sample):
                    feature_silent.append(feature[x: x + chunk_sample])

        for start, end in segment_cough:
            chunk_sample = config.CHUNK // 10
            if end - start >= chunk_sample:
                for x in range(start, end - chunk_sample, chunk_sample):
                    feature_cough.append(feature[x: x + chunk_sample])

    feature_silent = np.stack(feature_silent, axis=0)
    feature_cough = np.stack(feature_cough, axis=0)

    feature = np.concatenate([feature_silent, feature_cough], axis=0)
    label = torch.tensor(([0] * feature_silent.shape[0] + [1] * feature_cough.shape[0]), dtype=torch.int64)
    return feature, label


class MyDataset(Dataset):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    def __getitem__(self, item):
        return self.feature[item], self.label[item]

    def __len__(self):
        return self.feature.shape[0]


if __name__ == '__main__':
    read_data(config.DATA_DIR)
