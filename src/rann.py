#!/usr/bin/python
import numpy as np
from pybrain.datasets import SupervisedDataSet
from pybrain.datasets.sequential import SequentialDataSet
from pybrain.supervised import RPropMinusTrainer, BackpropTrainer
from pybrain.tools.validation import CrossValidator
from pybrain.structure import RecurrentNetwork, LinearLayer, TanhLayer, BiasUnit, FullConnection, SigmoidLayer, IdentityConnection

num = 15
hist = 3

net = RecurrentNetwork()
net.addInputModule(LinearLayer(num, name = 'i'))

for i in xrange(0,hist):
    net.addModule(LinearLayer(num, name='r{}'.format(i)))

net.addModule(SigmoidLayer(num*hist, name = 'h1'))
net.addModule(SigmoidLayer(num, name = 'h2'))
net.addOutputModule(SigmoidLayer(1, name = 'o'))

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
trainer = BackpropTrainer(net, learningrate = 0.3, momentum = 0.3, verbose = True, weightdecay=.001, lrdecay=0)

# carry out the training
#for i in xrange(2):
#    trainer.trainEpochs( 5 )

trainer.trainOnDataset(ds, 1000)
trainer.testOnData(verbose= True)