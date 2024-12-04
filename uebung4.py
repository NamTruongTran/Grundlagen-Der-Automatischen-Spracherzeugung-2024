
import scipy 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import recognizer.feature_extraction as fe

def compute_features():
    #window_size = 0.025  # 25 ms
    hop_size = 0.01  # 10 ms
    n_filters = 24
    audio_file = "data/TEST-MAN-AH-3O33951A.wav"
    mel_spectrum = fe.compute_features(audio_file,feature_type="MFCC_D_DD", num_ceps=13)

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
    plt.ylabel('MFCC + Delta + Delta-Delta Index')
    plt.title('MFCCs mit zeitlichen Ableitungen')
    plt.show()

if __name__ == "__main__":
    ################
    # SPEKTRALANALYSE
    ################
    compute_features()

    ################
    # DREIECKSFILTER
    ################

    
    ##############
    # MEL-SPEKTRUM
    ##############
    
    
    
