import numpy as np
import os
import pandas as pd
import wfdb
from nltk.tokenize import sent_tokenize
import librosa
from scipy import signal
import Objective_Function
from BERT import Model_BERT
from ESMA import ESMA
from HGSO import HGSO
from IOOA import IOOA
from LBOA import LBOA
from Model_HCARDNet import Model_HCARDNet
from OOA import OOA
from MY_emd import emd
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
import soundfile as sf
from Global_Vars import Global_Vars
from scipy.stats import skew, kurtosis
from sklearn import preprocessing
import random as rn
from numpy import matlib
import cv2 as cv
from Model_CNN import Model_CNN
from Model_DNN import Model_DNN
from Model_LSTM import Model_LSTM
from Spectral_Features import  rms, zcr, Rolloff, density, extract_entropy_feature
from Spectral_Flux import spectralFlux
from THDN import THDN
from Plot_Results import Plot_Results, Confusion_matrix, plot_Fitness
# Removing puctuations
def rem_punct(my_str):
    # define punctuation
    punctuations = '''!()-[]{};:'"\'',""<>./?@#$%^&*_~â€™â€˜'''
    # remove punctuation from the string
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char + " "
    # display the unpunctuated string
    return no_punct
def spectral_centroid(x, samplerate=44100):  # samplerate=44100
    magnitudes = np.abs(np.fft.rfft(x))  # magnitudes of positive frequencies
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0 / samplerate)[:length + 1])  # positive frequencies
    return np.sum(magnitudes * freqs[:len(magnitudes)]) / np.sum(magnitudes)  # return weighted mean

def Read_Image(Filename):
    image = cv.imread(Filename)
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = np.uint8(image)
    image = cv.resize(image, (512, 512))
    return image

def Wave_Features(ecg_signal, fs):
    Feat = []

    # Define a function to detect peaks in the signal
    def detect_peaks(signal):
        peaks, _ = wfdb.processing.find_peaks(signal, height=0.5, distance=int(fs * 0.6))
        return peaks

    # Detect R-peaks (QRS complex)
    r_peaks = detect_peaks(ecg_signal)
    # Extract P, T, and U waves around the R-peaks
    p_waves = ecg_signal[len(r_peaks) - int(0.2 * fs):len(r_peaks)]
    t_waves = ecg_signal[len(r_peaks):len(r_peaks) + int(0.4 * fs)]
    u_waves = ecg_signal[len(r_peaks) + int(0.1 * fs):len(r_peaks) + int(0.2 * fs)]
    peaks, _ = signal.find_peaks(ecg_signal, height=np.mean(ecg_signal), distance=round(fs * 0.200))
    QRS = peaks[peaks <= 10 * fs]
    waves = np.concatenate(([p_waves, t_waves, u_waves, QRS]), axis=None)
    return waves
def Read_Video(Filename):
    Images = []
    capture = cv.VideoCapture(Filename)
    currentframe = 0
    while True:
        ret, frame = capture.read()
        if ret:
            if len(frame.shape) == 3:
                frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            frame = np.uint8(frame)
            frame = cv.resize(frame, (512, 512))
            Images.append(frame)
            currentframe += 1
        else:
            break
    capture.release()
    return Images
def Read_Dataset(Directory):
    Data = []
    Target = []
    listFiles = os.listdir(Directory)
    for i in range(10):  # due to insufficient memory to save
        filename = Directory + listFiles[i]
        data = np.load(filename, allow_pickle=True)
        for j in range(len(data)):
            for k in range(data[j].shape[0]):
                print(i, j, k)
                subdata = data[j][k, :, :]
                Data.append(subdata[:14, :].reshape(-1))
                Target.append(0)
                Data.append(subdata[14:20, :].reshape(-1))
                Target.append(1)
                Data.append(subdata[20:29, :].reshape(-1))
                Target.append(2)
                Data.append(subdata[29:, :].reshape(-1))
                Target.append(3)
    Min = np.min([len(i) for i in Data])
    Data = [i[:Min] for i in Data]
    return Data, Target

# Read Dataset
an = 0
if an == 1:
    Data, Target = Read_Dataset('./Dataset/Dataset_2/')
    np.save('EEG_data.npy', Data)
    np.save('EEG_tar.npy', Target)

# read the Text Dataset
an = 0
if an == 1:
    File = './Dataset/Dataset_1/Suicide_Detection.csv'
    Data = pd.read_csv(File, encoding='unicode_escape')
    data_set = np.asarray(Data)
    data = data_set[:, :3]
    tar = data[:, -1]
    targ = np.zeros((len(tar))).astype('int')
    for i in range(len(tar)):
        print(i)
        if tar[i] == 'suicide':
            targ[i] = 1
        elif tar[i] == 'non-suicide':
            targ[i] = 0
    np.save('Text_data.npy', data[:, 1].reshape(-1, 1))
    np.save('Text_tar.npy', targ.reshape(-1, 1))

# read the audio Dataset
an = 0
if an == 1:
    File = './Dataset/Dataset_3/Audio/Audio_Song'
    Path = os.listdir(File)
    Target = []
    Audio_file = []
    for j in range(len(Path)):
        files = File + '/' + Path[j]
        files_path = os.listdir(files)
        for k in range(len(files_path)):
            print(j, k)
            Aud_files = files + '/' + files_path[k]
            SPLIT_name = os.path.basename(Aud_files)
            split_n = SPLIT_name.split('-')
            Emotion = split_n[2]
            data_1, samplerate_1 = sf.read(Aud_files)
            Target.append(int(Emotion[1]))
            if len(data_1.shape) == 1:
                Audio_file.append(np.asarray(data_1[:1000]))
            else:
                Audio_file.append(np.asarray(data_1[:1000, 1]))
    Target = np.asarray(Target)
    Uni = np.unique(Target)
    uni = np.asarray(Uni)
    tar = np.zeros((Target.shape[0], len(uni))).astype('int')
    for j in range(len(uni)):
        ind = np.where(Target == uni[j])
        tar[ind, j] = 1
    np.save('Audio_data.npy', Audio_file)
    np.save('Audio_tar.npy', tar)

# read the video Dataset
an = 0
if an == 1:
    File = './Dataset/Dataset_3/Video/Video_Song'
    Path = os.listdir(File)
    Video_Data = []
    Target = []
    for j in range(len(Path)):
        files = File + '/' + Path[j]
        files_path = os.listdir(files)
        video_file = []
        for k in range(len(files_path)):
            print(j, k)
            Aud_files = files + '/' + files_path[k]
            SPLIT_name = os.path.basename(Aud_files)
            split_n = SPLIT_name.split('-')
            Emotion = split_n[2]
            Image = Read_Video(Aud_files)
            Target.append(int(Emotion[1]))
            Video_Data.append(Image[:5])
    Target = np.asarray(Target)
    Uni = np.unique(Target)
    uni = np.asarray(Uni)
    tar = np.zeros((Target.shape[0], len(uni))).astype('int')
    for j in range(len(uni)):
        ind = np.where(Target == uni[j])
        tar[ind, j] = 1
    np.save('Video_data.npy', Video_Data)
    np.save('Video_tar.npy', tar)

# minimum dataset
an = 0
if an == 1:
    dataset = np.load('Text_data.npy', allow_pickle=True)
    audio = np.load('Audio_data.npy', allow_pickle=True)
    video = np.load('Video_data.npy', allow_pickle=True)
    dataeeg = np.load('EEG_data.npy', allow_pickle=True)
    datatar = np.load('Text_tar.npy', allow_pickle=True)
    audio_tar = np.load('Audio_tar.npy', allow_pickle=True)
    video_tar = np.load('Video_tar.npy', allow_pickle=True)
    eeg_tar = np.load('EEG_tar.npy', allow_pickle=True)
    Min = min([len(dataset), len(audio), len(video), len(dataeeg)])
    np.save('Data_Text.npy', dataset[:Min])
    np.save('Data_Audio.npy', audio[:Min])
    np.save('Data_Video.npy', video[:Min])
    np.save('Data_EEG.npy', dataeeg[:Min])
    np.save('Tar_Text.npy', datatar[:Min])
    np.save('Tar_Audio.npy', audio_tar[:Min])
    np.save('Tar_Video.npy', video_tar[:Min])
    np.save('Tar_EEG.npy', dataeeg[:Min])

# Text feature
an = 0
if an == 1:
    ps = PorterStemmer()
    Trans = []
    feat = []
    translateds = []
    Data = np.load('Data_Text.npy', allow_pickle=True)[:, 0]  # Load Datas
    Feat = []
    for i in range(len(Data)):
        print(i)
        D = Data[i]
        if type(D) == float:
            D = 'No Statement'
        ps = PorterStemmer()
        # punc = rem_punct(D)
        text_tokens = word_tokenize(D)  # convert in to tokens
        stem = []
        for w in text_tokens:  # Stemming
            stem_tokens = ps.stem(w)
            stem.append(stem_tokens)
        words = [word for word in stem if
                 not word in stopwords.words()]  # tokens without stop words
        # Punctuation Removal
        prep = rem_punct(words)
        dat = []
        v = []
        for m in sent_tokenize(str(prep)):
            temp1 = []
            # tokenize the sentence into words
            for n in word_tokenize(m):
                temp1.append(n.lower())
            dat.append(temp1[0])
        Bert = Model_BERT(dat)
        feat.append(Bert[0])
    np.save('Feat_1.npy', feat)

# Wave feature
an = 0
if an == 1:
    EEG_Signal = np.load('Data_EEG.npy', allow_pickle=True)
    EEG_feat = []
    fs = 300
    for i in range(len(EEG_Signal)):
        print(i, len(EEG_Signal))
        # EEG_Wave_features = extract_Wave_features(EEG_Signal[i], fs)
        EEG_Wave_features = Wave_Features(EEG_Signal[i], fs)
        EEG_feat.append(EEG_Wave_features[:140])
    np.save('Feat_2.npy', np.asarray(EEG_feat))

# EEG Feature
an = 0
if an == 1:
    eeg_signal = np.load('Data_EEG.npy', allow_pickle=True)
    IMFS = []
    for i in range(len(eeg_signal)):  # len(eeg_signal)
        print(i, len(eeg_signal))
        imfs = emd(eeg_signal[i], nIMF=1)
        IMFS.append(imfs[0])
    EMD_FEAT = np.asarray(IMFS)

    # Extract linear features from IMFs
    linear_features = []
    for imf in EMD_FEAT:
        mean = np.mean(imf)
        std = np.std(imf)
        skewness = skew(imf)
        kurt = kurtosis(imf)
        linear_features.append([mean, std, skewness, kurt])
    linear_features = np.array(linear_features)
    # Extract non-linear features
    non_linear_features = []
    for imf in EMD_FEAT:
        # Normalize the IMFs before computing non-linear features
        normalized_imf = preprocessing.scale(imf)
        # # Calculate Hjorth parameters
        hj_activity = np.var(normalized_imf)
        hj_mobility = np.sqrt(np.var(np.diff(normalized_imf)) / hj_activity)
        hj_complexity = np.sqrt(np.var(np.diff(np.diff(normalized_imf))) / np.var(np.diff(normalized_imf)))
        non_linear_features.append([hj_activity, hj_mobility, hj_complexity])
    non_linear_features = np.array(non_linear_features)
    Features = np.concatenate((linear_features, non_linear_features), axis=1)
    np.save('Feat_3.npy', Features)

# Spectral Feature Extraction from Audio Data
an = 0
if an == 1:
    Audios = np.load('Data_Audio.npy', allow_pickle=True)
    sf, f, dur = 100, 1, 4
    spectral = []
    for i in range(len(Audios)):
        print(i)
        Aud = Audios[i]
        cetroid = spectral_centroid(Aud)
        zero_crossings = librosa.zero_crossings(Aud, pad=True)
        zero_crossing = float(sum(zero_crossings))
        RMS = rms(Aud)  #
        ZCR = zcr(Aud)  #
        mfccs = librosa.feature.mfcc(y=Aud, sr=44100, n_mfcc=1)[0, 0]
        Density = density(Aud)
        Flux = spectralFlux(Aud)
        entropy = extract_entropy_feature(Aud)
        Thdn = THDN((Aud[:]), 44100)
        peaks = max(Aud)
        peak_amp = np.mean(peaks)
        rolloff = Rolloff(Aud, 44100)
        spec = [cetroid, Density, Flux, zero_crossing, entropy, peak_amp, Thdn, RMS, ZCR, rolloff, mfccs]
        spectral.append(spec)
    np.save('Feat_4.npy', spectral)

### Optimization for Weighted fused features
an = 0
if an == 1:
    Feat_1 = np.load('Feat_1.npy', allow_pickle=True)  # Load the feat1
    Feat_2 = np.load('Feat_2.npy', allow_pickle=True)  # Load the feat2
    Feat_3 = np.load('Feat_3.npy', allow_pickle=True)  # Load the feat3
    Feat_4 = np.load('Feat_4.npy', allow_pickle=True)  # Load the feat4
    Best_sol = []
    Global_Vars.Feat_1 = Feat_1
    Global_Vars.Feat_2 = Feat_2
    Global_Vars.Feat_3 = Feat_3
    Global_Vars.Feat_4 = Feat_4
    Npop = 10
    Chlen = 2 * 20
    xmin = matlib.repmat(
        np.concatenate(([np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), 0.01 * np.ones(20)]), axis=None), Npop,
        1)
    xmax = matlib.repmat(np.concatenate(([Feat_1.shape[1] - 1 * np.ones(5), Feat_2.shape[1] - 1 * np.ones(5),
                                          Feat_3.shape[1] - 1 * np.ones(5), Feat_4.shape[1] - 1 * np.ones(5),
                                          0.99 * np.ones(20)]), axis=None), Npop, 1)
    initsol = np.zeros(xmin.shape)
    for i in range(xmin.shape[0]):
        for j in range(xmin.shape[1]):
            initsol[i, j] = rn.uniform(xmin[i, j], xmax[i, j])
    fname = Objective_Function.Objfun
    max_iter = 50

    print('ESMA....')
    [bestfit1, fitness1, bestsol1, Time1] = ESMA(initsol, fname, xmin, xmax, max_iter)  # ESMA

    print('HGSO....')
    [bestfit2, fitness2, bestsol2, Time2] = HGSO(initsol, fname, xmin, xmax, max_iter)  # HGSO

    print('LBOA....')
    [bestfit3, fitness3, bestsol3, Time3] = LBOA(initsol, fname, xmin, xmax, max_iter)  # LBOA

    print('OOA....')
    [bestfit4, fitness4, bestsol4, Time4] = OOA(initsol, fname, xmin, xmax, max_iter)  # OOA

    print('IOOA....')
    [bestfit5, fitness5, bestsol5, Time5] = IOOA(initsol, fname, xmin, xmax, max_iter)  # improved OOA

    BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]

    np.save('BEST_Sol.npy', BestSol)  # Save the Bestsol

### feature Concatenation for Feature Fusion
an = 0
if an == 1:
    Feat_1 = np.load('Feat_1.npy', allow_pickle=True)  # Load the Feat 1
    Feat_2 = np.load('Feat_2.npy', allow_pickle=True)  # Load the Feat 1
    Feat_3 = np.load('Feat_3.npy', allow_pickle=True)  # Load the Feat 1
    Feat_4 = np.load('Feat_4.npy', allow_pickle=True)  # Load the Feat 1
    bests = np.load('BEST_Sol.npy', allow_pickle=True)[4, :]  # Load the Feat 4
    selected_feat1 = Feat_1[:, np.round(bests[:5]).astype('int')]
    selected_feat2 = Feat_2[:, np.round(bests[5:10]).astype('int')]
    selected_feat3 = Feat_3[:, np.round(bests[10:15]).astype('int')]
    selected_feat4 = Feat_4[:, np.round(bests[15:20]).astype('int')]
    Feature = np.concatenate((selected_feat1, selected_feat2, selected_feat3, selected_feat4), axis=1)
    np.save('Feature.npy', Feature)  # Save the Feature


### Optimization for Classification
an = 0
if an == 1:
    Feat = np.load('Feature.npy', allow_pickle=True)
    Video = np.load('Data_Video.npy', allow_pickle=True)
    Target = np.load('Tar_Video.npy',allow_pickle=True)
    Best_sol = []
    Global_Vars.Feat = Feat
    Global_Vars.Video = Video
    Global_Vars.Target = Target

    Npop = 10
    Chlen = 3
    xmin = matlib.repmat(([5,5,100]), Npop,1)
    xmax = matlib.repmat(([255,50,500]), Npop, 1)
    initsol = np.zeros(xmin.shape)
    for i in range(xmin.shape[0]):
        for j in range(xmin.shape[1]):
            initsol[i, j] = rn.uniform(xmin[i, j], xmax[i, j])
    fname = Objective_Function.Objfun_Cls
    max_iter = 50

    print('ESMA....')
    [bestfit1, fitness1, bestsol1, Time1] = ESMA(initsol, fname, xmin, xmax, max_iter)  # ESMA

    print('HGSO....')
    [bestfit2, fitness2, bestsol2, Time2] = HGSO(initsol, fname, xmin, xmax, max_iter)  # HGSO

    print('LBOA....')
    [bestfit3, fitness3, bestsol3, Time3] = LBOA(initsol, fname, xmin, xmax, max_iter)  # LBOA

    print('OOA....')
    [bestfit4, fitness4, bestsol4, Time4] = OOA(initsol, fname, xmin, xmax, max_iter)  # OOA

    print('IOOA....')
    [bestfit5, fitness5, bestsol5, Time5] = IOOA(initsol, fname, xmin, xmax, max_iter)  # improved OOA

    fitness = [fitness1, fitness2, fitness3, fitness4, fitness5]
    BestSol = [bestsol1,bestsol2,bestsol3,bestsol4,bestsol5]

    np.save('Fitness.npy', fitness)  # Save the Fitness
    np.save('Bestsol_Cls.npy', BestSol)  # Save the Bestsol

# Classification
an = 0
if an == 1:
    EVAL_all = []
    Feature = np.load('Feature.npy', allow_pickle=True)  # Load the Feat
    Target = np.load('Target.npy', allow_pickle=True)  # Load the Targets
    Video = np.load('Data_Video.npy',allow_pickle=True)
    BestSol = np.load('Bestsol_Cls.npy', allow_pickle=True)
    EVAL = []
    Feat = Feature
    Activation = [1,2,3,4,5]
    for learn in range(len(Activation)):
        Activation_function = round(Feat.shape[0] * 0.75)
        Train_Data = Feat[:Activation_function, :]
        Train_Target = Target[:Activation_function, :]
        Test_Data = Feat[Activation_function:, :]
        Test_Target = Target[Activation_function:, :]
        Eval = np.zeros((10, 14))
        for j in range(BestSol.shape[0]):
            print(learn, j)
            sol = np.round(BestSol[j, :]).astype(np.int16)
            Eval[j, :], pred = Model_HCARDNet(Feature,Video,Target,sol)
        Eval[5, :], pred_1 = Model_CNN(Feature,Video,Target)
        Eval[6, :], pred_2 = Model_LSTM(Feature,Video,Target)
        Eval[7, :], pred_3 = Model_DNN(Feature,Video,Target)
        Eval[8, :], pred_4 = Model_HCARDNet(Feature,Video,Target)
        Eval[9, :], pred_5 = Eval[4, :]
        EVAL.append(Eval)
    np.save('Eval_all.npy', EVAL)  # Save the Eval_all

Plot_Results()
Confusion_matrix()
plot_Fitness()
