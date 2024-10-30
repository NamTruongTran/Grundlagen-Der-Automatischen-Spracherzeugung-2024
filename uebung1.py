#!/usr/bin/env python3

import numpy as np
from scipy.io import wavfile
import recognizer.feature_extraction as mf
import matplotlib.pyplot as plt


def compute_features():
    audio_file = "data/TEST-MAN-AH-3O33951A.wav"

    # TODO read in audio_file and call make_frames()
    sampling_rate, audio_data = wavfile.read(audio_file)
    window_size = 0.4  # 400 ms Fensterlänge
    hop_size = 0.25  # 250 ms Verschiebung

    frames = mf.make_frames(audio_data, sampling_rate, window_size, hop_size)

    num_plots = frames[:4]

    window_size_samples = frames.shape[1]  # Gibt Anzahl der Samples pro Fenster wieder
    zeit_pro_frame = window_size * (window_size_samples / sampling_rate)
    time_axis = np.linspace(0, zeit_pro_frame, window_size_samples)

    # Subplots erstellen
    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    fig.suptitle("Erste vier Fenster des Audiosignals mit Hamming-Fenster")

    for i in range(4):
        axs[i].plot(time_axis, num_plots[i])
        axs[i].set_xlabel("Zeit in Sekunden")
        axs[i].set_ylabel("Amplitude")
        axs[i].set_title(f"Frame {i+1}")

        axs[i].set_xlim([0, 0.2])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compute_features()
