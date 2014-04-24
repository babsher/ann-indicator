import sys
#! /usr/bin/python

import os
import numpy as np
from numpy import absolute, arange
from scipy.signal import find_peaks_cwt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
    
def avg(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a =  np.convolve(values, weights)[:len(values)]
    a[:window] = a[window]
    return a

if __name__ == "__main__":
    file = open('../btceUSD-hourly.csv', 'r')
    data = []
    for line in file.readlines():
        row = line.split(',')
        data.append(float(row[1]))
    file.close()
    data = np.array(data)
    widths = np.arange(1, 24)
    max_distances = np.array(len(widths)).fill(20)
    delta = 25
    peakind = np.array([])
    for peak in find_peaks_cwt(vector=data, widths=widths, min_snr=1.75):
        range = np.arange(peak-delta, peak+delta)
        range = range[range >= 0]
        range = range[range < len(data)]
        if len(range) == len(np.arange(peak-delta, peak+delta)):
            max = np.argmax(data[range])
            peakind = np.int_(np.append(peakind, max+peak - delta))
        else:
            peakind = np.int_(np.append(peakind, peak))
    
    minind = np.array([])
    for minima in find_peaks_cwt(vector=[x * -1 for x in data], widths=widths, min_snr=1.75):
        range = np.arange(minima-delta, minima+delta)
        range = range[range >= 0]
        range = range[range < len(data)]
        if len(range) == len(np.arange(peak-delta, peak+delta)):
            min = np.argmin(data[range])
            minind = np.int_(np.append(minind, min+minima-delta))
        else:
            minind = np.int_(np.append(minind, minima))
    
    x = np.sort(np.append(minind, peakind))
    y = data[x]
    
    buy = np.array([])
    for p in x:
        if p in peakind:
            buy = np.append(buy, 0)
        else:
            buy = np.append(buy, 1)
    buy = np.append(buy, 0)
    x = np.append(x, len(data))
    bsig = interp1d(x,buy)
    
    expNorm = 26
    
    norm = avg(data, expNorm)
    
    out = None
    header = None
    for x in xrange(9, 25):
        a = np.divide(avg(data, x), norm)
        a = a[9:]
        if out is None:
            out = np.array([a])
            header = np.array([x])
        else:
            header = np.append(header, x)
            out = np.append(out, [a], axis=0)
        print out.shape, header.shape
    
    header = np.append(header, 'buy')
    out = np.append(out, [bsig(xrange(9, len(data)))], axis=0)
    print out.shape, header.shape
    
    np.savetxt(sys.argv[1], out, delimiter=',')
    
#    f1 = interp1d(x,y)
#    newx = np.linspace(4, x[len(x)-1], len(data))
#    plt.plot(np.arange(0,len(data)), data, 'b', newx, f1(newx), 'y', newx, bsig(newx), 'k')
#    plt.show()
    
    