from scipy.io import wavfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
#import tensorflow as tf
#from tensorflow import keras
from os import path
from pydub import AudioSegment


from pydub import AudioSegment

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


#dataset_train = keras.preprocessing.timeseries_dataset_from_array()
def processMP3(filename, save):
    sound = AudioSegment.from_mp3(filename)
    filename = filename[:len(filename)-4]
    filename = filename[4:]
    filename=filename+".wav"
    print(filename)
    sound.export(filename, format="wav")

    print(str(filename)+ " converted to .wav format")


def querysamplerate(filename):
    samplerate, data = wavfile.read(filename)
    return(samplerate)

#p
def processWAV(filename, save):
    print("Converting "+str(filename)+" to array file now...")
    samplerate, data = wavfile.read(filename)

    c = data
    print("Zipped, saving...")
    if(save==True):
        np.save("array_"+str(filename), c)
        print("Successfully saved left and right wav arrays for track: " + str(filename))


filenames = ["hotshit.wav", "lawandorder.wav", "semionem.wav", "banger.wav", "war.wav", "faneto.wav",
"faneto.wav", "foreigns.wav", "24.wav", "hotshit.wav", "paper.wav", "semionem.wav", "banger.wav", "faneto.wav", "semionem.wav", "semionem.wav", "semionem.wav", "semionem.wav"]

c = []
normalize=False
if(normalize==True):
    for x in range(0, len(filenames)):
        sound = AudioSegment.from_file(filenames[x], "wav")
        normalized_sound = match_target_amplitude(sound, -20.0)
        normalized_sound.export(filenames[x][:len(filenames[x])-4], format="wav")
normmatrix = 0
avg_length = 0
short_length= float('inf')
for x in range(0, len(filenames)):
    samplerate, data = wavfile.read(filenames[x])

    c.append(data)
    #compute average/short length

    if(len(data)<short_length):
        short_length = len(data)
    avg_length = avg_length + len(data)

avg_length = int(avg_length/len(filenames))
theta = np.zeros((short_length,2))
print(short_length)


for q in range(0, len(c)):
    for x in range(0, short_length):
        for y in range(0, 2):
            print(str(c[3   ][x]) + filenames[3])
            theta[x][y]+=c[q][x][y]
theta = theta / len(filenames)
wavfile.write("better_semionem.wav", querysamplerate("semionem.wav"), theta)




#c1 = [np.load("array_hotshit.wav.npy"), np.load("array_lawandorder.wav.npy"), np.load("array_paper.wav.npy")]
#c2 = [np.load("array_semionem.wav.npy"), np.load("array_24.wav.npy"), np.load("array_banger.wav.npy")]
#c1 = [np.load("semionem.wav.npy"), np.load("lawandorder.wav.npy"), np.load("array_paper.wav.npy")]

#print(c1.shape)
#print(c2.shape)
# x1: index, x2: r val, x3....x20:prev val y: l value
# x1: index, x2: l val, x3....x20:prev val, y: r value


#processMP3("mp3/24.mp3", True)
