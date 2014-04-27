#!/usr/bin/python
import numpy as np
from pybrain.datasets import SupervisedDataSet
from pybrain.datasets.sequential import SequentialDataSet
from pybrain.supervised import RPropMinusTrainer, BackpropTrainer
from pybrain.tools.validation import CrossValidator
from pybrain.structure import RecurrentNetwork, LinearLayer, TanhLayer, BiasUnit, FullConnection, SigmoidLayer, IdentityConnection

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

net = RecurrentNetwork()
net.addInputModule(LinearLayer(num, name = 'i'))

for i in xrange(0,hist):
    net.addModule(LinearLayer(num, name='r{}'.format(i)))

net.addModule(SigmoidLayer(num*hist, name = 'h1'))
net.addModule(SigmoidLayer(num, name = 'h2'))
net.addOutputModule(TanhLayer(1, name = 'o'))

net.addRecurrentConnection(IdentityConnection(net['i'], net['r0']))
for i in xrange(1,hist):
    net.addRecurrentConnection(IdentityConnection(net['r{}'.format(i-1)], net['r{}'.format(i)]))
    net.addConnection(FullConnection(net['r{}'.format(i)], net['h1']))

net.addConnection(FullConnection(net['i'], net['h1']))
net.addConnection(FullConnection(net['h1'], net['h2']))
net.addConnection(FullConnection(net['h2'], net['o']))
net.sortModules()

print net

ds = SequentialDataSet(15, 1)
ds.newSequence()

input = open('../btceUSD-labled.csv', 'r')
for line in input.readlines():
    row = np.array(line.split(','))
#    print float(row[16])
#    print [float(x) for x in row[:15]]
#    print len(row)
    ds.addSample([float(x) for x in row[:15]], float(row[16]))
print ds

#trainer = RPropMinusTrainer( net, dataset=ds, verbose=True )
#trainer = BackpropTrainer(net, learningrate = 0.5, momentum = 0.3, verbose = True, weightdecay=0, lrdecay=0.000001)

# carry out the training
#for i in xrange(2):
#    trainer.trainEpochs( 5 )

net.reset()
trainer = RPropMinusTrainer( net, dataset=ds, verbose=True, etaplus=1.7)
trainer.trainEpochs(1000)
evalRnnOnSeqDataset(net, ds, verbose = True)