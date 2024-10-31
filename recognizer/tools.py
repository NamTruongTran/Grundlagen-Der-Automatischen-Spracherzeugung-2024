import math


def sec_to_samples(x, sampling_rate):
    # TODO implement this method
    #      x              : Zeitintervall in Sek
    # sampling_rate       : Abtastfrequenz (Abtastung des Signals pro Sek)
    # num_samples         : Anzahl der Abtastpunkte innerhalb von der Zeit x
    num_samples = x * sampling_rate
    num_samples_round = round(num_samples)
    num_samples_int = int(num_samples_round)
    return num_samples_int


def next_pow2(x):
    # TODO implement this method
    # Logarithmus zur Basis 2 verwenden um die Potenz n zu bestimmen, damit 2^n >= x
    # math.ceil : Aufrunden der Zahl, weil wenn n eine Dezi Zahl ist dann ist 2^n < x
    two_next_pow = math.log2(x)
    two_next_pow_round = math.ceil(two_next_pow)
    two_next_pow_int = int(two_next_pow_round)
    return two_next_pow_int


def dft_window_size(x, sampling_rate):
    # TODO implement this method

    # num_samples: Anzahl der Abtastpunkte im Zeitintervall x
    num_samples = sec_to_samples(x, sampling_rate)

    # Nächste Zweierpotenz finden damit gilt : 2^n >= num_samples
    w_size = 2 ** next_pow2(num_samples)

    return w_size


def get_num_frames(signal_length_samples, window_size_samples, hop_size_samples):
    # TODO implement this method
    # Signal ist in Fenster unterteilt: Anzahl der Fenster für ein Audiosignal berechnen
    # signal_length_samples : Gesamtanzahl der Abtastpunkte des Signals
    # window_size_samples   : Anzahl der Abtastpunkte pro Fenster
    # hop_size_samples      : Verschiebung von einem Fenster zum nächsten in Abtastpunkte
    #      math.ceil        : Aufrunden falls das letzte Frame nicht mehr passt
    # Formel gibt an wie oft gehopt wird bzw. wie viele Fenster im GesamtSignal
    overlap = window_size_samples - hop_size_samples
    num_frames = (signal_length_samples - overlap) / hop_size_samples
    num_frames_round = math.ceil(num_frames)

    return num_frames_round
