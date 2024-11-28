import numpy as np
import recognizer.tools as tools
from scipy.io import wavfile

def make_frames(audio_data, sampling_rate, window_size, hop_size):
    # TODO implement this method

    # Berechne die Anzahl der Samples für die Fenster- und Hop-Größe
    window_size_samples = tools.dft_window_size(window_size, sampling_rate)
    hop_size_samples = tools.sec_to_samples(hop_size, sampling_rate)
    
    # Berechne die Anzahl der Rahmen, die benötigt werden
    num_frames = tools.get_num_frames(len(audio_data), window_size_samples, hop_size_samples)
    
    # Erzeuge ein Hamming-Fenster der berechneten Fenstergröße
    hamming_window = np.hamming(window_size_samples)
    
    # Initialisiere das Array für die Rahmen
    frames = np.zeros((num_frames, window_size_samples), dtype=float)
    
    # Fülle das Array mit den Fensterrahmen
    for i in range(num_frames):
        # Bestimme den Start- und Endindex für den aktuellen Rahmen
        start_index = i * hop_size_samples
        end_index = start_index + window_size_samples
        frame = audio_data[start_index:end_index]
        
        # Extrahiere den aktuellen Rahmen und prüfe, ob Zero-Padding erforderlich ist
        if len(frame) < window_size_samples:
            # Füge Zero-Padding hinzu, falls das Ende des Audiosignals erreicht ist
            frame = np.pad(frame, (0, window_size_samples - len(frame)))
 
        # Multipliziere den Rahmen mit dem Hamming-Fenster
        frames[i, :] = frame * hamming_window

        # ohne Hamming-window
        # frames[i, :] = frame * hamming_window
    
    return frames
    pass

# numpy.fft.rfft()
# die aus einem Array frames mit Signalrahmen
# wie er von make frames() zur¨uckgegeben wird, den nicht-redundanten Teil des Betragsspektrums als zweidimensionales Array im Datentyp float zur¨uck gibt.

def compute_absolute_spectrum(frames):
    # Betragsspektrum : gibt die Amplitude (Stärke) der jeweiligen Frequenzkomponente an
    # Jede Zeile ist ein Frame : repräsentiert das Betragsspektrum des Frames (enthält Frequenzinformationen: Frequenzkomponenten sind in der Reihenfolge von der niedrigsten bis zur höchsten Frequenz angeordnet)
    #   axis=1      : Jedes Frame bzw. Zeile in Frames wird mit FFT analysiert
    # np.fft.rfft() : Zerlegung in Frequenzkomponenten gibt ein Array von Komplexen Zahlen und automatisch den nicht redudanten Teil zurück
    #   np.abs      : Berechnet den Betrag der Komplexen Zahlen (Amplitude)
    spectrum = np.fft.rfft(frames, axis=1)
    abs_spectrum = np.abs(spectrum)

    return abs_spectrum

def apply_mel_filters(abs_spectrum, filterbank):
    """
    Wendet eine Dreiecksfilterbank auf ein Betragsspektrum an.
    """
    return np.dot(filterbank, abs_spectrum)


def compute_features(audio_file, window_size=25e-3, hop_size=10e-3, feature_type='STFT', fbank_fmax=8000, num_ceps=13 , n_filters=24, fbank_fmin=0):
    # Audiodatei einlesen
    sampling_rate, audio_data = wavfile.read(audio_file)
    
    # Normalisieren der Audiodaten auf den Bereich -1 bis 1
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Frames erstellen mit der Funktion make_frames
    frames = make_frames(audio_data, sampling_rate, window_size, hop_size)
    
    # Berechnung des Betragsspektrums mit der Funktion compute_absolute_spectrum
    absolute_spectrum = compute_absolute_spectrum(frames)
    
    return absolute_spectrum

def get_mel_filters(sampling_rate, window_size_sec, n_filters, f_min=0, f_max=8000):

    # in FFT groesse
    N = int(sampling_rate * window_size_sec)  
    f_min_mel = tools.hz_to_mel(f_min)
    f_max_mel = tools.hz_to_mel(f_max)

    # Stuetzstellen der Punkte
    mel_points = np.linspace(f_min_mel, f_max_mel, n_filters + 2) # Stützstellen werden im Mel-Bereich äquidistant verteilt
    hz_points = tools.mel_to_hz(mel_points) # wieder in Hz umgewandelt
    
    # Frequenzstützstellen (in Hz) mit der Formel aus der PDF berechnen
    f_hz = (hz_points * sampling_rate / 2) * (2 / N)  # Frequenzen auf FFT-Bereich skalieren

    # Bin-Indizes berechnen
    f = np.floor((f_hz / (sampling_rate / 2)) * (N // 2)).astype(int)

    # Indizes von Frequenzpunkten
    f = np.floor((hz_points / (sampling_rate / 2)) * (N // 2)).astype(int)

    # Sicherstellen, dass punkte in grenzen sind
    f = np.clip(f, 0, N // 2)

    # Debug 
    print(f"f_min: {f_min}, f_max: {f_max}")
    print(f"Mel points: {mel_points}")
    print(f"Bin points: {f}")

    # Filterbank-Matrix initialisieren
    filters = np.zeros((n_filters, N // 2 ))

    for m in range(1, n_filters + 1):
        f_left = f[m - 1]  # f[m-1]
        f_center = f[m]    # f[m]
        f_right = f[m + 1]  # f[m+1]

        # negative laengen uerspringen
        if f_left == f_center or f_center == f_right:
            print(f"Skipping filter {m}: f_left={f_left}, f_center={f_center}, f_right={f_right}")
            continue

        # links
        if f_left < f_center:
            filters[m - 1, f_left:f_center] = (np.arange(f_left, f_center) - f_left) / (f_center - f_left)

        # rechts
        if f_center < f_right:
            filters[m - 1, f_center:f_right] = (f_right - np.arange(f_center, f_right)) / (f_right - f_center)

    # Normalize each filter
    for m in range(n_filters):
        if np.sum(filters[m]) > 0:
            filters[m] /= np.sum(filters[m])

    return filters


def get_mel_filterss(sampling_rate, window_size_sec, n_filters, f_min=0, f_max=8000):
    """
    Computes the Mel filterbank with triangular filters.
    """
    N = int(sampling_rate * window_size_sec)  # FFT size
    f_min_mel = tools.hz_to_mel(f_min)
    f_max_mel = tools.hz_to_mel(min(f_max, sampling_rate / 2))  # Clamp f_max to Nyquist frequency

    # Mel points
    mel_points = np.linspace(f_min_mel, f_max_mel, n_filters + 2)
    hz_points = tools.mel_to_hz(mel_points)

    # Bin points
    bin_points = np.floor((hz_points / sampling_rate) * (N // 2 + 1)).astype(int)

    # Ensure bin points are within valid range
    bin_points = np.clip(bin_points, 0, N // 2)

    # Debug output
    print(f"f_min: {f_min}, f_max: {f_max}")
    print(f"Mel points: {mel_points}")
    print(f"Bin points: {bin_points}")

    # Initialize filterbank
    filters = np.zeros((n_filters, N // 2 + 1))

    for m in range(1, n_filters + 1):
        f_left, f_center, f_right = bin_points[m - 1], bin_points[m], bin_points[m + 1]

        # Skip degenerate filters
        if f_left == f_center or f_center == f_right:
            print(f"Skipping filter {m}: f_left={f_left}, f_center={f_center}, f_right={f_right}")
            continue

        # Left slope
        if f_left < f_center:
            filters[m - 1, f_left:f_center] = (np.arange(f_left, f_center) - f_left) / (f_center - f_left)

        # Right slope
        if f_center < f_right:
            filters[m - 1, f_center:f_right] = (f_right - np.arange(f_center, f_right)) / (f_right - f_center)

    # Normalize each filter
    for m in range(n_filters):
        if np.sum(filters[m]) > 0:
            filters[m] /= np.sum(filters[m])

    return filters

