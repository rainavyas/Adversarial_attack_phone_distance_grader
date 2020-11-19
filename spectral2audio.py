'''
Convert spectral energy coefficients
to a frame of values in the raw audio domain
'''
import math
import numpy as np
from bisect import bisect_left
from scipy.io.wavfile import write
import argparse
import torch

SAMPLING_FREQ = 16000
FRAME_SIZE = 400 # 25ms length frame
DFT_SIZE = FRAME_SIZE
SPECTRAL_SIZE = 24
HIGH_FREQ = SAMPLING_FREQ/2
LOW_FREQ = 0

#spectral_vals_if_no_model = [1.0000e+00, 7.6181e-02, 1.0000e+00, 1.0000e+00, 5.0850e-01, 1.3706e-03, 2.1112e-02, 8.6159e-02, 5.4362e-02, 1.0000e+00, 1.0000e+00, 3.0498e-02, 4.3693e-02, 1.0173e-02, 1.0008e-02, 1.9726e-01, 3.0916e-02, 8.6139e-01, 6.3313e-04, 9.9886e-01, 1.9224e-01, 1.2433e-01, 1.4528e-02, 9.9355e-01]
spectral_vals_if_no_model = [1000]*24

def f2mel(f):
    '''
    Converts a frequency to the Mel frequency scale
    '''
    mel = 1125*math.log(1+(f/700))
    return mel

def mel2f(mel):
    '''
    Converts a Mel frequency to the normal definition of frequency
    '''
    f = 700 * (math.exp(mel/1125) - 1)
    return f

def get_closest(theNum, theList):
    '''
    Assumes the list is ordered
    Returns closest value in theList to theNum
    If two numbers equally close, returns the smaller one
    '''
    pos = bisect_left(theList, theNum)
    if pos == 0:
        return theList[0]
    if pos == len(theList):
        return theList[-1]
    before = theList[pos - 1]
    after = theList[pos]
    if after - theNum < theNum - before:
       return after
    else:
       return before


def spectral2audio(spectral_vals, output_file):

    # Transform low and high frequencies to mel frequencies
    mel_h = f2mel(HIGH_FREQ)
    mel_l = f2mel(LOW_FREQ)

    # Get the mel filter positions on mel scale
    filters_mel = np.linspace(mel_l, mel_h, SPECTRAL_SIZE + 2)
    filters_mel = filters_mel[1:-1].tolist()

    # Convert to normal frequency positions
    filters_f = [mel2f(mel) for mel in filters_mel]

    # Create list of the frequencies for dft
    dft_f_pos = np.linspace(LOW_FREQ, SAMPLING_FREQ, DFT_SIZE).tolist()
    # Keep only the first half for now (the posotive associated freq)-> second half (negative associated freq) will be used for symmetry to ensure x(t) is real valued
    dft_f_pos = dft_f_pos[:int(FRAME_SIZE/2)+1]

    # Create dictionary of closest frequency to relevant spectral values
    f2spectral_val = {}
    for m, filter in enumerate(filters_f):
        f = get_closest(filter, dft_f_pos)
        f2spectral_val[f] = spectral_vals[m]

    # Iterate through DFT positioned frequencies and assign spectral energies
    # to the closest frequncies closest to the mel filter positions
    powers = []
    for dft_f in dft_f_pos:
        if dft_f in f2spectral_val:
            power = f2spectral_val[dft_f]
        else:
            power = 0
        powers.append(power)
     

    # Convert power signal to a Fourier signal
    fourier_vals = [math.sqrt(FRAME_SIZE*p) for p in powers]

    # The DFT values have only been defined from frequencies 0Hz to 8kHz
    # We need it defined from -8kHz to 8kHz, which in the DFT requires
    # values to be defined from 0kHz to 16kHz (periodicity of DTFT)
    # For real valued in time systems the spectrum should be symmetric.
    fourier_vals_reverse = list(reversed(fourier_vals[1:]))
    full_dft_vals = fourier_vals + fourier_vals_reverse
    full_dft_vals = np.array(full_dft_vals)

    # Perform the inverse DFT (use a FFT)
    frame = np.fft.ifft(full_dft_vals).real
    print("Frame values, sampled at Hz", SAMPLING_FREQ)
    print(frame)

    # The current frame is very short, so it is useful for hearing purposes to loop it
    num_loops = 4000
    frames_repeated = np.tile(frame, num_loops)

    # Convert the signal to an actual audio file
    write(output_file, SAMPLING_FREQ, frames_repeated.astype(np.dtype('i2')))

# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('--model_path', default=None, type=str, help='Specify trained attack model path')
commandLineParser.add_argument('--output_file', type=str, help='Specify wav file path to save to')

args = commandLineParser.parse_args()
model_path = args.model_path
output_file = args.output_file

# Get spectral values
if model_path == None:
    spectral_vals = spectral_vals_if_no_model
else:
    attack_model = torch.load(model_path)
    spectral_vals = attack_model.get_noise().tolist()

# Transform to audio domain
spectral2audio(spectral_vals, output_file)
