import numpy as np
import matplotlib.pyplot as plt
import recognizer.feature_extraction as fe
from scipy.io import wavfile
import recognizer.tools as tools


def compute_features():
    audio_file = "data/TEST-MAN-AH-3O33951A.wav"
    sampling_rate, audio_data = wavfile.read(audio_file)

    # TODO call compute_features() and plot spectrogram

    # Berechne das Spektrum der Audiodatei
    window_size = 0.025  # 25 ms Fensterlänge
    hop_size = 0.01  # 10 ms Verschiebung
    abs_spectrum = fe.compute_features(audio_file, window_size, hop_size)

    ### Berechne den logarithmischen Wert des Spektrums in dB
    # np.maximum : Schutz vor Berechnung von log(0) oder log(negativ)
    spectrum_db = 20 * np.log10(np.maximum(abs_spectrum, 1e-10))

    ### Berechnung der Dauer eines Frames in Sekunden
    # Um die Zeitachse für die x-Achse des Spektrogramms zu berechnen, ist der zeitliche Abstand zwischen den einzelnen Frames entscheidend
    hop_size_samples = tools.sec_to_samples(hop_size, sampling_rate)
    zeit_sample_pro_hop = hop_size_samples / sampling_rate
    num_frames = abs_spectrum.shape[0]

    # Erstelle die Zeitachse in Sekunden
    # np.linspace(start, stop, num)
    #  stop : Gesamtdauer des Spektrogramms
    #  num  : Jeder Frame bekommt einen Wert auf der Zeitachse
    time_axis = np.linspace(0, zeit_sample_pro_hop * num_frames, num_frames)

    ### Plotten des Spektrogramms
    # auto : skaliert Bild passen für den Raum
    # lowe : Urspung unten links
    # cmap="virdis" : Farbschema, viridis ist Farbskala von dunkel (niedrige Werte) bis hell (Hohe Werte)
    # extent : Skalierung der Achsen [x_min, x_max, y_min, y_max]
    #               [-1]         : Greift auf letzten Wert zu (Skalierung auf gesamte Dauer)
    #          sampling_rate / 2 : entspricht Nyquist-Frequenz (Höchste Frequenz die in abgetasteten Signal dargestellt werden kann)
    plt.figure(figsize=(10, 6))
    plt.imshow(
        spectrum_db.T,
        aspect="auto",
        origin="lower",
        extent=[0, time_axis[-1], 0, sampling_rate / 2],
        cmap="viridis",
    )
    plt.colorbar(label="Amplitude (dB)")
    plt.xlabel("Zeit in Sekunden")
    plt.ylabel("Frequenz (Hz)")
    plt.title("Spektrogramm der Audiodatei")
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


def plot_mel_spectrum():
    # Berechnung des Mel-Spektrums
    audio_file = "data/TEST-MAN-AH-3O33951A.wav"
    sampling_rate, audio_data = wavfile.read(audio_file)
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

def plot_mfcc():
    audio_file = "data/TEST-MAN-AH-3O33951A.wav"
    sampling_rate, audio_data = wavfile.read(audio_file)
    features = fe.compute_features(audio_file, feature_type='MFCC_D_DD', num_ceps=13)

    plt.figure(figsize=(10, 6))
    plt.imshow(features.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Amplitude')
    plt.xlabel('Zeit (Frames)')
    plt.ylabel('MFCC + Delta + Delta-Delta Index')
    plt.title('MFCCs mit zeitlichen Ableitungen')
    plt.show()

if __name__ == "__main__":
    ################
    # SPEKTRALANALYSE
    ################
    #compute_features()
    plot_mel_spectrum()

    ################
    # DREIECKSFILTER
    ################
    #plot_mel_filters() 

    ##############
    # MEL-SPEKTRUM
    ##############
    #plot_mfcc()
