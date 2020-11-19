'''
Convert two separate wav files with element-wise addition
'''

import numpy as np
from scipy.io.wavfile import read, write
import argparse

commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('--first', type=str, help='Specify first wav file location')
commandLineParser.add_argument('--second', type=str, help='Specify second wav file location')

args = commandLineParser.parse_args()
first = args.first
second = args.second

f1, x = read(first)
f2, y = read(second)

print(f1)
print(f2)

l1 = len(x)
l2 = len(y)

if l1>l2:
    x = x[:l2]
else:
    y = y[:l1]

z = x + y
out = first[:-4]+'_and_'+second
write(out, f1, z)
