
import scipy 
import numpy as np
import matplotlib.pyplot as plt
import recognizer.feature_extraction as fe


def compute_features():
    # Berechnung des Mel-Spektrums
    audio_file = "data/TEST-MAN-AH-3O33951A.wav"
    window_size = 0.025  # 25 ms
    hop_size = 0.01  # 10 ms
    n_filters = 24
    mel_spectrum = fe.compute_features(audio_file, window_size, hop_size, feature_type="FBANK", n_filters=n_filters)

    # Plot des Mel-Spektrums
    plt.figure(figsize=(10, 6))
    plt.imshow(
        mel_spectrum.T,
        aspect="auto",
        origin="lower",
        extent=[0, mel_spectrum.shape[0] * hop_size, 0, n_filters],
        cmap="viridis",
    )
    plt.colorbar(label="Amplitude (log)")
    plt.xlabel("Zeit in Sekunden")
    plt.ylabel("Mel-Filter-Index")
    plt.title("Mel-Spektrum der Audiodatei")
    plt.show()

def plot_mel_filters():
    """Plot der Mel-Dreiecksfilterbank."""
    sampling_rate = 16000
    window_size = 0.025
    n_filters = 24

    # Mel-Filterbank berechnen
    mel_filters = fe.get_mel_filters(sampling_rate, window_size, n_filters)

    # Plot der Filterbank
    plt.figure(figsize=(10, 6))
    for i in range(n_filters):
        plt.plot(mel_filters[i])

    plt.title("Mel-Dreiecksfilterbank")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    ################
    # SPEKTRALANALYSE
    ################
    compute_features()    
    ################
    # DREIECKSFILTER
    ################
    plot_mel_filters()
    
    ##############
    # MEL-SPEKTRUM
    ##############
    
    
    
