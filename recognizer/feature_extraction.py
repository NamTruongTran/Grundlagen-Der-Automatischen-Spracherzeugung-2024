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
            frame = np.pad(frame, (window_size_samples - len(frame)))
 
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