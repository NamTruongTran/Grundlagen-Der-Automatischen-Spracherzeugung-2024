import math
import numpy as np

def sec_to_samples(x, sampling_rate):

    return int(x * sampling_rate)
    pass

def next_pow2(x):

    return math.ceil(math.log2(abs(x)))
    pass

# next pow2 rundet immer auf die naechst hoehere potenz
# bsp: fuer 300 => 2^8 ist 256
# man will aber eine potenz finden die groesser 300 ist
# hoehere ist 2^9 = 500...

# gibt die 2^9 zurueck, da die dft besser mit den zweierpotenzen arbeitet, statt einer ganzen zahl
def dft_window_size(x, sampling_rate):

    samples = sec_to_samples(x, sampling_rate)
    b = 2 ** next_pow2(samples)
    return b 
    pass

# window_size gibt die abtastrate an zb 512 samples/abtastraten
# ein frame hat dann 512 samples
# hop_size ist die einheit, in der das ganze signal nach und nach in frames abgetastet wird
# man nimmt hop_size, damit sogenannte overlaps entstehen, die dann praeziser sind, da mehrmals die gleichen abschnitte/overlaps 
# vom signal genommen werden
def get_num_frames(signal_length_samples, window_size_samples, hop_size_samples):

    overlap = window_size_samples - hop_size_samples

    num_frames = (signal_length_samples - overlap) / hop_size_samples

    return math.ceil(num_frames) # mit math bib wird zu math.ceil(num_frames)
    pass

def hz_to_mel(x):
    # berechnet den Wert der Mel-Skala für den entsprechenden Frequenzwert x
    return 2595 * np.log10(1 + x/700 )

def mel_to_hz(x):
    # Konvertiert eine Frequenz x (in Mel) zurück in Hz
    return 700 * (10**(x / 2595) - 1 )