import matplotlib.pyplot as plt
import librosa
import numpy as np
import requests


class Vis:
    
    @staticmethod
    def _norm_raw(raw):
        return raw / np.max(np.abs(raw),axis=0)
    
    
    @staticmethod
    def _time_axis(raw, labels):
        '''
        Private function.
        Generates time axis for a raw signal and its labels.
        '''
        time = np.linspace(0, len(raw) / 16000, num=len(raw))
        time_labels = np.linspace(0, len(raw) / 16000, num=len(labels))
        return time, time_labels
    
    
    @staticmethod
    def _plot_waveform(frames, labels, title = 'Sample'):
        '''
        Private function.
        Plot a raw signal as waveform and its corresponding labels.
        '''
        raw = Vis._norm_raw(frames.flatten())
        time, time_labels = Vis._time_axis(raw, labels)
        
        plt.figure(1, figsize=(16, 3))
        plt.title(title)
        plt.plot(time, raw)
        plt.plot(time_labels, labels - 0.5)
        plt.show()
        
    
    @staticmethod
    def plot_sample(frames, labels, title = 'Sample', show_distribution = True):
        '''
        Plot a sample with its original labels
        (before noise is applied to sample).
        '''
        Vis._plot_waveform(frames, labels, title)

        # Print label distribution if enabled.
        if show_distribution:
            voice = (labels.tolist().count(1) * 100) / len(labels)
            silence = (labels.tolist().count(0) * 100) / len(labels)
            print('{0:.0f} % voice {1:.0f} % silence'.format(voice, silence))
          
    
    @staticmethod
    def plot_sample_webrtc(frames, sensitivity = 0):
        '''
        Plot a sample labeled with WebRTC VAD
        (after noise is applied to sample).
        Sensitivity is an integer from 0 to 2,
        with 0 being the most sensitive.
        '''
        vad = webrtcvad.Vad(sensitivity)
        labels = np.array([1 if vad.is_speech(f.tobytes(), sample_rate=SAMPLE_RATE) else 0 for f in frames])
        Vis._plot_waveform(frames, labels, title = 'Sample (WebRTC)')    
        
    
    @staticmethod
    def plot_features(mfcc = None, delta = None):
        '''
        Plots the MFCC and delta-features
        for a given sample.
        '''
        if mfcc is not None:
            plt.figure(1, figsize=(16, 3))
            plt.plot(mfcc)
            plt.title('MFCC ({0} features)'.format(mfcc.shape[1]))
            plt.show()
        
        if delta is not None:
            plt.figure(1, figsize=(16, 3))
            plt.plot(delta)
            plt.title('Deltas ({0} features)'.format(mfcc.shape[1]))
            plt.show()


def draw(path, segments):
    audio, sr = librosa.load(path, sr=16000)
    label = [0] * len(audio)
    for start, end in segments:
        for i in range(int(start * 16000), int(end * 16000)):
            label[i] = 1

    label = np.array(label)

    Vis.plot_sample(audio, label, show_distribution=False)

def main():
    url = "http://localhost:5002/api/detect"
    path = "/home/lehoa/PycharmProjects/cough_classification/Data_Cough/audio/Cough3.wav"
    payload = {"token": "12345678", "alg": "lstm", "audio_path": path}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        response = response.json()
        if response['msg'] == 'Success':
            segments = response['result']

            draw(path, segments)


if __name__ == '__main__':
    SAMPLE_RATE = 16000
    main()