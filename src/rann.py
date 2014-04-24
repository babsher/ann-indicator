
from pybrain.structure import RecurrentNetwork, LinearLayer, TanhLayer, BiasUnit, FullConnection, SigmoidLayer, IdentityConnection

num = 2
hist = 3

net = RecurrentNetwork()
net.addInputModule(LinearLayer(num, name = 'i'))

for i in xrange(0,hist):
    net.addModule(LinearLayer(num, name='r{}'.format(i)))

net.addModule(SigmoidLayer(num*hist, name = 'h1'))
#net.addModule(SigmoidLayer(num, name = 'h2'))
net.addOutputModule(TanhLayer(1, name = 'o'))

net.addRecurrentConnection(IdentityConnection(net['i'], net['r1']))
for i in xrange(1,hist):
    net.addRecurrentConnection(IdentityConnection(net['r{}'.format(i-1)], net['r{}'.format(i)]))
    net.addConnection(FullConnection(net['r{}'.format(i)], net['h1']))

#net.addConection(FullConnection(net['h1'], net['h2']))
net.addConnection(FullConnection(net['h1'], net['o']))
net.sortModules()

print net