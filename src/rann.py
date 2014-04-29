#!/usr/bin/python
import numpy as np
import sys
from pybrain.datasets import SupervisedDataSet
from pybrain.datasets.sequential import SequentialDataSet
from pybrain.supervised import RPropMinusTrainer, BackpropTrainer
from pybrain.tools.validation import CrossValidator
from pybrain.structure import RecurrentNetwork, LinearLayer, TanhLayer, FullConnection, SigmoidLayer, IdentityConnection, MDLSTMLayer, BiasUnit

def evalRnnOnSeqDataset(net, DS, verbose = False, silent = False):
    """ evaluate the network on all the sequences of a dataset. """
    r = 0.
    samples = 0.
    for seq in DS:
        net.reset()
        for i, t in seq:
            res = net.activate(i)
            if verbose:
                print t, res
            r += sum((t-res)**2)
            samples += 1
        if verbose:
            print '-'*20
    r /= samples
    if not silent:
        print 'MSE:', r
    return r

num = 15
hist = 9

net = RecurrentNetwork('simpleMDLstmNet')
i = LinearLayer(num, name = 'i')
dim = num
h = MDLSTMLayer(dim, peepholes = True, name = 'MDlstm')
o = TanhLayer(1, name = 'o')
b = BiasUnit('bias')
net.addModule(b)
net.addOutputModule(o)
net.addInputModule(i)
net.addModule(h)
net.addConnection(FullConnection(i, h, outSliceTo = 4*dim, name = 'f1'))
net.addConnection(FullConnection(b, h, outSliceTo = 4*dim, name = 'f2'))
net.addRecurrentConnection(FullConnection(h, h, inSliceTo = dim, outSliceTo = 4*dim, name = 'r1'))
net.addRecurrentConnection(IdentityConnection(h, h, inSliceFrom = dim, outSliceFrom = 4*dim, name = 'rstate'))
net.addConnection(FullConnection(h, o, inSliceTo = dim, name = 'f3'))
net.sortModules()

print net

ds = SequentialDataSet(15, 1)
ds.newSequence()

input = open(sys.argv[1], 'r')
for line in input.readlines():
    row = np.array(line.split(','))
    ds.addSample([float(x) for x in row[:15]], float(row[16]))
print ds

test = SequentialDataSet(15, 1)
test.newSequence()
input = open(sys.argv[2], 'r')
for line in input.readlines():
    row = np.array(line.split(','))
    test.addSample([float(x) for x in row[:15]], float(row[16]))
print test

net.reset()
trainer = RPropMinusTrainer( net, dataset=ds, verbose=True)
trainer.trainEpochs(1000)
evalRnnOnSeqDataset(net, test, verbose = True)