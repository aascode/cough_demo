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
    extractor = Extractor()
    audio_dir = os.path.join(data_dir, "audio")
    label_dir = os.path.join(data_dir, "label")
    audio_silent = []
    audio_cough = []

    for filename in os.listdir(audio_dir):
        audio_path = os.path.join(audio_dir, filename)
        label_path = os.path.join(label_dir, filename.replace(".wav", ".txt"))
        audio, sr = torchaudio.load(audio_path)
        audio_size = audio.shape[1]
        segment_silent = []
        segment_cough = []

        last = 0
        with open(label_path, "r") as f:
            for line in f.readlines():
                start, end, _ = line.split()
                start, end = int(float(start) * 16000), int(float(end) * 16000)
                segment_silent.append((max(0, last), min(start, audio_size)))
                segment_cough.append((max(0, start), min(end, audio_size)))
                last = end

            segment_silent.append((max(last, 0), audio_size))

        for start, end in segment_silent:
            chunk_sample = config.CHUNK * 16000 // 1000
            if end - start >= chunk_sample:
                if config.model == "LSTM":
                    for x in range(start, end - chunk_sample, 160):
                        audio_silent.append(audio[:, x: x + chunk_sample])
                else:
                    for i in range((end - chunk_sample - start) // chunk_sample):
                        x = random.randint(start, end - chunk_sample)
                        audio_silent.append(audio[:, x: x + chunk_sample])

        for start, end in segment_cough:
            chunk_sample = config.CHUNK * 16000 // 1000
            if end - start >= chunk_sample:
                if config.model == "LSTM":
                    for x in range(start, end - chunk_sample, 160):
                        audio_cough.append(audio[:, x: x + chunk_sample])
                else:
                    for i in range((end - chunk_sample - start) // chunk_sample):
                        x = random.randint(start, end - chunk_sample)
                        audio_cough.append(audio[:, x: x + chunk_sample])

    audio_silent = torch.stack(audio_silent, dim=0)
    audio_cough = torch.stack(audio_cough, dim=0)
    feature_silent = extractor.extract_feature(audio_silent)
    feature_cough = extractor.extract_feature(audio_cough)

    # feature_silent = feature_silent.reshape(-1, feature_silent.shape[2])
    # feature_cough = feature_cough.reshape(-1, feature_cough.shape[2])

    # feature_silent = torch.reshape(feature_silent, (feature_silent.shape[0], -1))
    # feature_cough = torch.reshape(feature_cough, (feature_cough.shape[0], -1))
    feature = torch.cat([feature_silent, feature_cough], dim=0)
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
