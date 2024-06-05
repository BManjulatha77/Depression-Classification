import librosa
import numpy
from numpy import *
import tools
from scipy.stats import entropy

def density(signal):
    fs = 1000.0  # 1 kHz sampling frequency
    F1 = 10  # First signal component at 10 Hz
    F2 = 60  # Second signal component at 60 Hz
    T = 10  # 10s signal length
    N0 = -10  # Noise level (dB)
    t = np.r_[0:T:(1 / fs)]  # Sample times

    # # Two Sine signal components at frequencies F1 and F2.
    # signal = np.sin(2 * F1 * np.pi * t) + np.sin(2 * F2 * np.pi * t)
    #
    # # White noise with power N0
    # signal += np.random.randn(len(signal)) * 10 ** (N0 / 20.0)
    import scipy.signal

    # f contains the frequency components
    # S is the PSD
    (f, S) = scipy.signal.periodogram(signal, fs, scaling='density')
    return np.mean(S)


import numpy as np
from typing import Union
# from . import tools
import scipy.stats


def calculate_entropy(spectrum: Union[list, np.ndarray],
                      max_mz: float = None,
                      noise_removal: float = 0.01,
                      ms2_da: float = 0.05, ms2_ppm: float = None):
    """
    The spectrum will be cleaned with the procedures below. Then, the spectral entropy will be returned.
    1. Remove ions have m/z higher than a given m/z (defined as max_mz).
    2. Centroid peaks by merging peaks within a given m/z (defined as ms2_da or ms2_ppm).
    3. Remove ions have intensity lower than max intensity * fixed value (defined as noise_removal)
    :param spectrum: The input spectrum, need to be in 2-D list or 2-D numpy array
    :param max_mz: The ions with m/z higher than max_mz will be removed.
    :param noise_removal: The ions with intensity lower than max ion's intensity * noise_removal will be removed.
    :param ms2_da: The MS/MS tolerance in Da.
    :param ms2_ppm: The MS/MS tolerance in ppm.
    If both ms2_da and ms2_ppm is given, ms2_da will be used.
    """
    spectrum = tools.clean_spectrum(spectrum, max_mz=max_mz, noise_removal=noise_removal, ms2_da=ms2_da, ms2_ppm=ms2_ppm)
    return scipy.stats.entropy(spectrum[:, 1])


def Rolloff(audio_data, sr):
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
    # audio_path = aud
    # audio=  np.load('Data_Audio.npy', allow_pickle=True)
    # for j in range(12):
    #     y, sr = librosa.load(audio)
    #     # Calculate the spectral roll-off
    #     roll_off = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    #     rolloff = np.mean(roll_off)
    return spectral_rolloff[0]



def centroid(self):
    """
    Compute the spectral centroid.
    Characterizes the "center of gravity" of the spectrum.
    Approximately related to timbral "brightness"
    """
    binNumber = 0

    numerator = 0
    denominator = 0

    for bin in self:
        # Compute center frequency
        f = (self.sampleRate / 2.0) / len(self)
        f = f * binNumber

        numerator = numerator + (f * abs(bin))
        denominator = denominator + abs(bin)

        binNumber = binNumber + 1

    return (numerator * 1.0) / denominator


def rms(self):
    """
    Compute the root-mean-squared amplitude
    """
    sum = 0
    for i in range(0, len(self)):
        sum = sum + self[i] ** 2

    sum = sum / (1.0 * len(self))

    return (sum)  # math.sqrt(sum)


def rolloff(self, samplerate):
    """
    Determine the spectral rolloff, i.e. the frequency below which 85% of the spectrum's energy
    is located
    """
    absSpectrum = abs(self)
    spectralSum = numpy.sum(absSpectrum)

    rolloffSum = 0
    rolloffIndex = 0
    for i in range(0, len(self)):
        rolloffSum = rolloffSum + absSpectrum[i]
        if (rolloffSum > (0.85 * spectralSum)).all():
            rolloffIndex = i
            break

    # Convert the index into a frequency
    frequency = rolloffIndex * (samplerate / 2.0) / len(self)
    return frequency


def zcr(self):
    """
    Compute the Zero-crossing rate (ZCR)
    """
    zcr = 0
    for i in range(1, len(self)):
        if ((self[i - 1] * self[i]) < 0).all():
            zcr = zcr + 1

    return zcr / (1.0 * len(self))

def extract_entropy_feature(y, frame_size=2048, hop_size=512):


    # Calculate short-term energy
    energy = librosa.feature.rms(y=y, frame_length=frame_size, hop_length=hop_size)[0]

    # Calculate entropy from the energy values
    entropies = []
    for i in range(len(energy) ):
        frame_energy = energy[i]
        normalized_energy = frame_energy / np.sum(frame_energy)  # Normalize energy values
        entropies.append(entropy(normalized_energy))

    return entropies[0]