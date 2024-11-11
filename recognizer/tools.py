import math

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

    return next_pow2(num_frames) # mit math bib wird zu math.ceil(num_frames)
    pass