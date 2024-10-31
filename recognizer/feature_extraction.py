import numpy as np
from .tools import sec_to_samples, next_pow2, dft_window_size, get_num_frames


def make_frames(audio_data, sampling_rate, window_size, hop_size):
    # TODO implement this method

    # Verschiebung in Abtastpunkte umrechnen
    hop_size_samples = sec_to_samples(hop_size, sampling_rate)

    # Fenster in Abtastpunkte umrechnen
    # Aufrunden auf nächsthöhere Zweierpotenz -> Besser für die Berechnung der Fourier-Transformation (Bei zweierpotenzen ist die FT schneller)
    window_size_samples_power = dft_window_size(window_size, sampling_rate)

    # Anzahl der Fenster berechnen
    audio_data_len = len(audio_data)
    num_frames = get_num_frames(
        audio_data_len, window_size_samples_power, hop_size_samples
    )

    # Hamming-Fenster für das glätten der Ränder erstellen (Amplitude am Ende und Anfang des Fensters auf Null bringen -> Reduziert Störung benachtbarten Frequenzen)
    hamm_window = np.hamming(window_size_samples_power)

    # Leeres 2-dim Array
    # Anzahl der Zeilen   : Gesamt Anzahl der Fenster
    # Anzahl der Spalten  : Anzahl der Samples pro Fenster
    frames = np.zeros((num_frames, window_size_samples_power), dtype=float)

    # Fensterung
    for i in range(num_frames):
        # Start des Fensters im Audio Signal
        start = i * hop_size_samples

        # Ende des Fensters im Audio Signal
        end = start + window_size_samples_power

        # Abschnitt des Audio Signals in das Frame kopieren
        frame = audio_data[start:end]

        # Zero-Padding
        # Falls das Frame nicht vollständig ist, wird es auf die Länge window_size_samples_power gebracht indem pad dazu addiert
        # (0, window_size_samples - len(frame)) : Am ende werden "window_size_samples - len(frame)" Nullen zu dem Fenster hinzugefügt
        if len(frame) < window_size_samples_power:
            frame = np.pad(frame, (0, window_size_samples_power - len(frame)))

        ## Hamming-Fenster auf i-te Fenster anwenden
        frames[i, :] = frame * hamm_window

        ## Ohne Hamming-Fenster
        # frames[i, :] = frame

    return frames
