import os

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

no_of_dataset = 1


def Image_Results_for_Audio():
    for n in range(1):
        Images = np.load('Sample_audio.npy', allow_pickle=True)
        Image = [6,7,8,9,10]
        for i in range(len(Image)):
            tar = Images[Image[i]]
            fig = plt.figure(figsize=(6, 6))
            # fig.canvas.set_window_title(classs[i])
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.plot(tar, 'b-')
            ax1.set_title('Original Image', fontsize=12)
            fig.tight_layout()
            path1 = "./Sample images/Image_results/Dataset-orig-Audio-%s.png" % str(i + 1)
            plt.savefig(path1)
            plt.show()
        # for i in range(len(Image)):
        #     plt.title('Original Image')
        #     plt.plot(Images[Image[i]])
        #     plt.xlabel('Time (ms)')
        #     plt.ylabel('Amplitude')
        #     # image = cv.resize(Images[Image[i]], [268, 268])
        #     plt.show()
        #     path1 = "./Sample images/Image_results/Dataset-orig-Audio_%s_%s_image.png" % (Images[Image[n]], i + 1)
        #     plt.savefig(path1)
            # cv.imwrite('./Sample images/Image_results/Dataset-orig-Audio-' + str(n + 1) + '-' + str(i + 1) + '.png',Images[Image[i]])
            # cv.imwrite(os.path.join('./Sample images/Image_results/','Dataset-orig-Audio-' + str(n + 1) + '-' + str(i + 1) + '.png'), Images[Image[i]])



def Image_Results_for_EEG():
    for n in range(1):
        Images = np.load('EEG_data.npy', allow_pickle=True)
        Image = [1, 2, 3, 4, 5]
        for i in range(len(Image)):
            tar = Images[Image[i]]
            fig = plt.figure(figsize=(6, 6))
            # fig.canvas.set_window_title(classs[i])
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.plot(tar, 'b-')
            ax1.set_title('Original Image', fontsize=12)
            fig.tight_layout()
            path1 = "./Sample images/Image_results/Dataset-orig-EEG-%s.png" % str(i + 1)
            plt.savefig(path1)
            plt.show()
        # for i in range(len(Image)):
        #     plt.title('Original Image')
        #     plt.plot(Images[Image[i]])
        #     plt.xlabel('Time (ms)')
        #     plt.ylabel('Amplitude')
        #     plt.show()
        #     cv.imwrite('./Sample images/Image_results/Dataset-orig-EEG-' + str(n + 1) + '-' + str(i + 1) + '.png',Images[Image[i]])


def Image_Results_for_Video():
    for n in range(5):  # For 5 videos
        cls = ['Video_Song_1', 'Video_Song_2', 'Video_Song_3', 'Video_Song_4', 'Video_Song_5']
        Video = np.load('Sample_Video.npy', allow_pickle=True)[n]
        for i in range(1):
            print(n)
            Original = Video
            Orig_1 = Original[i]
            Orig_2 = Original[i + 1]
            Orig_3 = Original[i + 2]
            Orig_4 = Original[i + 3]
            Orig_5 = Original[i + 4]
            Orig_6 = Original[i + 5]
            plt.suptitle('Sample frames from ' + cls[n] + ' ', fontsize=25)
            plt.subplot(2, 3, 1).axis('off')
            plt.imshow(Orig_1)
            plt.subplot(2, 3, 2).axis('off')
            plt.imshow(Orig_2)
            plt.subplot(2, 3, 3).axis('off')
            plt.imshow(Orig_3)
            plt.subplot(2, 3, 4).axis('off')
            plt.imshow(Orig_4)
            plt.subplot(2, 3, 5).axis('off')
            plt.imshow(Orig_5)
            plt.subplot(2, 3, 6).axis('off')
            plt.imshow(Orig_6)
            path1 = "./Sample images/Image_results/Sample image of _%s_%s_image.png" % (cls[n], i + 1)
            plt.savefig(path1)
            plt.show()
            cv.imwrite('./Sample images/Image_results/Dataset-Orig_1-Video_' + str(n + 1) + '-' + str(i + 1) + '.png',
                       Orig_1)
            cv.imwrite('./Sample images/Image_results/Dataset-Orig_2-Video_' + str(n + 1) + '-' + str(i + 1) + '.png',
                       Orig_2)
            cv.imwrite('./Sample images/Image_results/Dataset-Orig_3-Video_' + str(n + 1) + '-' + str(i + 1) + '.png',
                       Orig_3)
            cv.imwrite('./Sample images/Image_results/Dataset-Orig_4-Video_' + str(n + 1) + '-' + str(i + 1) + '.png',
                       Orig_4)
            cv.imwrite('./Sample images/Image_results/Dataset-Orig_5-Video_' + str(n + 1) + '-' + str(i + 1) + '.png',
                       Orig_5)
            cv.imwrite('./Sample images/Image_results/Dataset-Orig_6-Video_' + str(n + 1) + '-' + str(i + 1) + '.png',
                       Orig_6)


if __name__ == '__main__':
    Image_Results_for_EEG()
    Image_Results_for_Audio()
    Image_Results_for_Video()
