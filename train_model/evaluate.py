import os
import pickle
from preprocess import Extractor
import config
import torch
import torchaudio
import librosa



def batching(audio, batch_size=1):
    idx = 0
    num_sample = config.CHUNK * 16000 // 1000 * batch_size
    while idx + num_sample <= audio.shape[1]:
        yield audio[:, idx: idx + num_sample].reshape(batch_size, audio.shape[0], -1)
        idx += num_sample


# class Detector:
#     def __init__(self):
#         self.extractor = Extractor()
#         self.model_0 = pickle.load(open("resource/non_cough.pkl", 'rb'))
#         self.model_1 = pickle.load(open("resource/cough.pkl", 'rb'))
#
#     def detect(self, path):
#         audio, sr = torchaudio.load(path)
#         all_pred = []
#         for batch in batching(audio):
#             feature = self.extractor.extract_feature(batch)
#             feature = feature.reshape(feature.shape[0], -1)
#             score_0 = self.model_0.score_samples(feature)
#             score_1 = self.model_1.score_samples(feature)
#             for i in range(len(score_1)):
#                 if score_0[i] > score_1[i]:
#                     all_pred.append(0)
#                 else:
#                     all_pred.append(1)
#
#         all_pred.append(0)
#         segment = []
#         start = -1
#         end = -1
#         for i, y in enumerate(all_pred):
#             if y == 1 and (i == 0 or all_pred[i - 1] == 0):
#                 start = i
#
#             if y == 0 and i > 0 and all_pred[i - 1] == 1:
#                 end = i
#
#             if start != -1 and end != -1:
#                 segment.append((start * config.CHUNK / 1000, end * config.CHUNK / 1000))
#                 start = -1
#                 end = -1
#
#         return segment


def merge_segment(segment, min_cough_length=0, max_silent_length=400):
    result = []
    last = -100000000000
    for start, end in segment:
        if end - start <= min_cough_length:
            continue

        if start - last > max_silent_length:
            result.append([start, end])
        else:
            result[-1][1] = end

        last = end

    return result


class Detector:
    def __init__(self):
        self.extractor = Extractor()
        self.model = torch.load(os.path.join(config.MODEL_DIR, "model_15.pt"), map_location="cpu")  # da bo sung them
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def detect(self, path):
        audio, sr = torchaudio.load(path)
        all_pred = []
        for batch in batching(audio):
            feature = self.extractor.extract_feature(batch).to(self.device)
            y_hat = self.model(feature)
            y_hat = torch.nn.functional.softmax(y_hat, dim=1)
            y_pred = torch.argmax(y_hat, dim=1)
            for y in y_hat:
                if y[1] >= 0.5:
                    all_pred.append(1)
                else:
                    all_pred.append(0)
            # all_pred += y_pred.cpu().numpy().tolist()

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
                segment.append((start * config.CHUNK, end * config.CHUNK))
                start = -1
                end = -1

        return segment


def main():
    detector = Detector()
    test_dir = os.path.join(config.TEST_DIR)
    for filename in os.listdir(test_dir):
        if not filename.endswith(".wav"):
            continue

        path = os.path.join(test_dir, filename)
        segment = detector.detect(path)
        segment = merge_segment(segment)
        with open(os.path.join(test_dir, filename.replace(".wav", ".txt")), "w") as f:
            for start, end in segment:
                f.write("{}\t{}\n".format(start / 1000, end / 1000))


if __name__ == '__main__':
    #main()
    detector = Detector()
    segment = detector.detect("Data_Cough/audio/Cough1.wav")
    print(segment)
    segment = merge_segment(segment)
    print(segment)
    for i in segment:
        print(i)
        print(i[0])
        print(i[1])


