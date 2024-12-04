import numpy as np
import recognizer.tools as tools
from scipy.io import wavfile
from scipy.fftpack import dct

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

def compute_features(audio_file, window_size=25e-3, hop_size=10e-3, feature_type='FBANK', fbank_fmax=8000, num_ceps=13 , n_filters=24, fbank_fmin=0):
    # Audiodatei einlesen
    sampling_rate, audio_data = wavfile.read(audio_file)
    
    # Normalisieren der Audiodaten auf den Bereich -1 bis 1
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Frames erstellen mit der Funktion make_frames
    frames = make_frames(audio_data, sampling_rate, window_size, hop_size)
    
    # Berechnung des Betragsspektrums mit der Funktion compute_absolute_spectrum
    absolute_spectrum = compute_absolute_spectrum(frames)

     # Feature-Berechnung basierend auf feature_type
    if feature_type == 'FBANK':
        # Mel-Filterbank berechnen
        filterbank = get_mel_filters(sampling_rate, window_size, n_filters, fbank_fmin, fbank_fmax)

        # Mel-Filterbank anwenden
        mel_spectrum = apply_mel_filters(absolute_spectrum, filterbank)
        epsi = np.full(mel_spectrum.shape, np.nextafter(0, 1))
        mel_spectrum = np.where(mel_spectrum == 0, epsi, mel_spectrum)
        log_mel_spectrum=np.log(mel_spectrum)
        return log_mel_spectrum

    
    # Log-Mel-Spektrum berechnen
    if feature_type.startswith('MFCC'):
        filterbank = get_mel_filters(sampling_rate, window_size, n_filters, fbank_fmin, fbank_fmax)
        mel_spectrum = apply_mel_filters(absolute_spectrum, filterbank)
        log_mel_spectrum = np.log(np.maximum(mel_spectrum, 1e-10))
        cep = compute_cepstrum(absolute_spectrum, num_ceps)
        
        # MFCCs berechnen
        #mfcc = compute_cepstrum(log_mel_spectrum, num_ceps)
        
        if feature_type == 'MFCC':
            return cep
        elif feature_type == 'MFCC_D':
            delta = get_delta(cep)
            return append_delta(cep, delta)
        elif feature_type == 'MFCC_D_DD':
            delta = get_delta(cep)
            delta_delta = get_delta(delta)
            return append_delta(append_delta(cep, delta), delta_delta)

    return absolute_spectrum


def get_mel_filters(sampling_rate, window_size_sec, n_filters, f_min=0, f_max=8000):

    # FFT-Größe
    N = tools.dft_window_size(window_size_sec, sampling_rate)

    # Mel-Skala berechnen
    f_min_mel = tools.hz_to_mel(f_min)
    f_max_mel = tools.hz_to_mel(f_max)
    

    # Mel-Frequenzstützstellen berechnen
    mel_points = np.linspace(f_min_mel, f_max_mel, n_filters + 2)
    #hz_points = np.array([tools.mel_to_hz(i) for i in mel_points])
    hz_points = tools.mel_to_hz(mel_points) 


    # Frequenzen in Bin-Indizes umwandeln
    f = np.round(hz_points / (sampling_rate/N)).astype(int)
    #f = np.round((hz_points/((sampling_rate) / N))).astype(int)

    # Initialisieren der Filterbank
    filters = np.zeros((n_filters, int(N/2) + 1)) 

    # Filter berechnen
    for m in range(1, n_filters + 1):

        if m + 1 >= len(f):
            print(f"Überspringen von Filter {m}: Index außerhalb der Grenzen")
            continue

        f_left, f_center, f_right = f[m - 1], f[m], f[m + 1]
     
        # Linke Flanke
        for k in range(f_left, f_center):
            if f_left <= k < f_center:
                filters[m - 1, k] = (
                    2 * (k - f_left) / ((f_right - f_left) * (f_center - f_left))
                )
                    
        # Rechte Flanke
        for k in range(f_center, f_right):
            if f_center <= k <= f_right:
                filters[m - 1, k] = (
                    2 * (f_right - k) / ((f_right - f_left) * (f_right - f_center))
                )
         

    return filters


def apply_mel_filters(abs_spectrum, filterbank):
    
    S_mel = np.dot(abs_spectrum, filterbank.T)
    return S_mel


def compute_cepstrum(mel_spectrum, num_ceps):
    # Numerische Probleme vermeiden
    mel_spectrum = np.maximum(mel_spectrum, np.finfo(float).eps)
    
    # Logarithmus des Mel-Spektrums berechnen
    log_mel_spectrum = np.log(mel_spectrum)
    
    # Diskrete Kosinustransformation (DCT)
    cepstrum = dct(log_mel_spectrum, type=2, axis=1, norm='ortho')
    
    # Gibt nur die ersten num_ceps Koeffizienten zurueck
    return cepstrum[:, :num_ceps]

def get_delta(x):
    # berechnet erste zeitlich Ableitung des Merkmalsvektors
    delta = np.zeros_like(x)
    for t in range(x.shape[0]):
        if t == 0:
            delta[t] = x[t + 1] - x[t]
        elif t == x.shape[0] - 1:
            delta[t] = x[t] - x[t - 1]
        else:
            delta[t] = 0.5 * (x[t + 1] - x[t - 1])
    return delta

def append_delta(x, delta):
    # konkateniert Merkmalsvektor mit erster zeitlich Ableitung
    return np.hstack((x, delta))