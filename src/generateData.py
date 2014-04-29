#! /usr/bin/python

import sys
import numpy as np
from scipy.signal import find_peaks_cwt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter
    
def avg(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a =  np.convolve(values, weights)[:len(values)]
    a[:window] = a[window]
    return a

if __name__ == "__main__":
    file = open(sys.argv[1], 'r')
    data = []
    for line in file.readlines():
        row = line.split(',')
        data.append(float(row[1]))
    file.close()
    data = np.array(data)
    
    # spell out the args that were passed to the Matlab function
    N=6
    Fc=60
    Fs=1600
    # provide them to firwin
    h = firwin( numtaps=N, cutoff=40, nyq=Fs/2)

    y = lfilter( h, 1.0, data)
#    plt.plot(xrange(0,len(data)), data, 'b', xrange(0, len(y)), y, 'k')
    
    widths = np.arange(4, 24)
    max_distances = np.array(len(widths)).fill(20)
    delta = 200
    peakind = np.array([])
    for peak in find_peaks_cwt(vector=y, widths=widths, min_snr=1.75):
        range = np.arange(peak-delta, peak+delta)
        range = range[range >= 0]
        range = range[range < len(data)]
        if len(range) == len(np.arange(peak-delta, peak+delta)):
            max = np.argmax(data[range])
            if max != delta-1 or max != 0:
                peakind = np.int_(np.append(peakind, max+peak - delta))
        else:
            peakind = np.int_(np.append(peakind, peak))
    
    minind = np.array([])
    for minima in find_peaks_cwt(vector=[x * -1 for x in y], widths=widths, min_snr=1.75):
        range = np.arange(minima-delta, minima+delta)
        range = range[range >= 0]
        range = range[range < len(data)]
        if len(range) == len(np.arange(peak-delta, peak+delta)):
            min = np.argmin(data[range])
            if min != delta-1 or min != 0:
                minind = np.int_(np.append(minind, min+minima-delta))
        else:
            minind = np.int_(np.append(minind, minima))
    
    x = np.sort(np.append(minind, peakind))
    y = data[x]
    
    buy = np.array([])
#    for p in xrange(0,len(data)):
    for p in x:
        if p in peakind:
            buy = np.append(buy, -1)
        elif p in minind:
            buy = np.append(buy, 1)
        else:
            buy = np.append(buy, 0)
#    buy = buy[9:]
    buy = np.append(buy, 0)
    bsig = interp1d(np.append(x, len(data)),buy)
    
    expNorm = 40
    
    norm = avg(data, expNorm)
    
    out = None
    header = None
    for size in xrange(9, 25):
        a = np.divide(avg(data, size), norm)
        a = a[9:]
        if out is None:
            out = np.array([a])
            header = np.array([size])
        else:
            header = np.append(header, size)
            out = np.append(out, [a], axis=0)
        print out.shape, header.shape
    
    header = np.append(header, 'buy')
    out = np.append(out, [bsig(xrange(9, len(data)))], axis=0)
#    out = np.append(out, [buy], axis=0)
    out = np.transpose(out)
    print out.shape, header.shape
    
    np.savetxt(sys.argv[2], out, delimiter=',', fmt="%.10f")
    
#    plt.plot(xrange(0,len(data)), data, 'b', xrange(9, len(data)), buy, 'k', minind, data[minind], 'r+', peakind, data[peakind], 'kx')
    plt.show()
    
    