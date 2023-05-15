import os
import librosa as librosa
from tkinter import Tk, filedialog
import soundfile as sf
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import time
from scipy.signal import fftconvolve

total_iterations = 100
progress_bar = tqdm(total=total_iterations)

# Open file dialog to select a file
root = Tk()
root.withdraw()
audio_file_path = filedialog.askopenfilenames(title="Select Audio sample file(s)")

# Load the HRTF data from the .mat file
master_HRTF = sio.loadmat('ReferenceHRTF.mat')

# print((master_HRTF[]))
source_positions = master_HRTF['sourcePosition']

# as the "sourcePosition" is a numpy array and it has 1550 position that actually defines the multiple points.
# The Azimuth is the first indices and the elevation is the second one , it has seven indices but as per the requirement
# We are working on the first two.

# Load the input audio file
# input_file = 'Assets/nightingale_near_street-23902.mp3'
input_files = list(audio_file_path)
# print(input_files)

# Initialisation of source list and audio_data for processing.
Source_name = []
audio_data = []
audio_count = 0

print("\nThe Audio files selected by the users are :")
for audio_source in input_files:
    Source_name.insert(audio_count, os.path.basename(audio_source))
    temp_audio, sample_rate = librosa.load(audio_source, sr=None)
    # creating a temp audio so that we can store it later in a list.
    audio_data.insert(audio_count, temp_audio)
    print(Source_name[audio_count])
    audio_count = audio_count + 1
    # Sample rate none kar diya haikyuki deafunt data ka SR pata hi nhia hai hame.
    # Define the spatial location of the sound source in the below HRTF dataset
    # the data set is for the last value which 1550
audio_data_unpadded = audio_data
# creating another variable for the unpadded data, for the concatination process.

# Calculate the maximum length of the audios, actually we were not able to merge the multiple audios due to different lengts
max_length = max([len(audio) for audio in audio_data])
# Filling the shorter audios with zeros, so that we can merge them later.
for i in range(len(audio_data)):
    if len(audio_data[i]) < max_length:
        padding = np.zeros(max_length - len(audio_data[i]))
        audio_data[i] = np.concatenate((audio_data[i], padding))

azimuth_deg = []
# horizontal angle in degrees
elevation_deg = []
# vertical angle in degrees
input_iteration = 0
while input_iteration < audio_count:
    temp_azimuth_degree = float(input("\nEnter the Azimuth Degree for {0}: ".format(Source_name[input_iteration])))
    temp_elevation_degree = float(input("Enter the Elevation Degree for {0}: ".format(Source_name[input_iteration])))
    input_iteration = input_iteration + 1
    azimuth_deg.append(temp_azimuth_degree)
    elevation_deg.append(temp_elevation_degree)

# # Find the closest HRTF measurements to the desired azimuth and elevation
index = []
# creating index for the list for storing the multiple indexes together.
for azimuth, elevation in zip(azimuth_deg, elevation_deg):
    input_counter = 0
    azimuth_index = np.abs(source_positions[:, 0] - azimuth).argmin()
    elevation_index = np.abs(source_positions[:, 1] - elevation).argmin()
    temp_index = np.where((source_positions[:, 0] == source_positions[azimuth_index, 0]) &
                          (source_positions[:, 1] == source_positions[elevation_index, 1]))[0][0]
    index.append(temp_index)
    print("Index of the HRTF dataset for the {0} source is : {1}".format(Source_name[input_counter], temp_index))
    input_counter = input_counter + 1

# Extract the HRTF impulse responses for the left and right ears
HRTF_left = master_HRTF['hrtfData']
HRTF_right = master_HRTF['hrtfData']

# checking if the left and right hrtf has 256 values as per the hrtf dataset from the australian website.
convolve_iteration = 0
specialized_audio_list = []
while convolve_iteration < audio_count:
    left_hrtf_at_location = HRTF_left[:, index[convolve_iteration], 0]
    right_hrtf_at_location = HRTF_right[:, index[convolve_iteration], 1]
    # storing the 255 values of the left and right HRIR in the variable.
    # Apply the HRTF to the input audio signal to generate binaural audio.
    left_channel = fftconvolve(audio_data[convolve_iteration], left_hrtf_at_location)
    right_channel = fftconvolve(audio_data[convolve_iteration], right_hrtf_at_location)
    # Save the binaural audio as a .wav file
    specialized_audio = np.column_stack((left_channel, right_channel))
    specialized_audio_list.append(specialized_audio)
    convolve_iteration = convolve_iteration + 1


for i in range(total_iterations):
    # Do some work here
    time.sleep(0.05)
    # Update the progress bar
    progress_bar.update(1)

# Now merging the file into one , hey bhagwaan chal jana :-)
# pehla audio sample here.
merged_audio_signal = specialized_audio_list[0]
for count in range(1, len(specialized_audio_list)):
    # print(count)
    # print(audio_count)
    merged_audio_signal += specialized_audio_list[count]

concatinated_audio_signal = specialized_audio_list[0]
for count in range(1, len(specialized_audio_list)):
    concatinated_audio_signal = np.concatenate((concatinated_audio_signal, specialized_audio_list[count]), axis=0)

output_file_name = Source_name[0] + "_Final_Output_Overlapped_Sandeep_Bhandari.wav"
sf.write(output_file_name, merged_audio_signal, 48000, 'PCM_24')

output_concatinated_file_name = Source_name[0] + "_Final_Output_Concatinated_Sandeep_Bhandari.wav"
sf.write(output_concatinated_file_name, concatinated_audio_signal, 48000, 'PCM_24')

print("\n The convolved audio files has been merged in one file first "
      "file's name in it & saved in the current directory : \n\n===>", output_file_name)

print("\n  The convolved audio files has been concatinated in one file first "
      "file's name in it & saved in the current directory : \n\n===>", output_concatinated_file_name)
