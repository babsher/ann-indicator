#!/usr/bin/python
from datetime import datetime,timedelta
import time

f = open('../btceUSD.csv','r')
out = open('out.csv','w')

def nextHour(date):
    d = datetime(year=date.year, month=date.month, day=date.day, hour=date.hour)
    return d + timedelta(hours=1)
    
lastHour = None
lastPrice = None
lastDelta = None
volume = 0
for line in f.readlines():
    data = line.split(',')
    date = datetime.utcfromtimestamp(float(data[0]))
    #print 'Last values {0}, {1}, {2}'.format(lastHour, lastPrice, lastDelta)
    #print '{0},{1},{2}'.format(date, data[1], float(data[2]))
    if lastHour == None:
        lastHour = nextHour(date)
        lastPrice = data[1]
        lastDelta = lastHour - date
        volume = volume + float(data[2])
    else:
        delta = lastHour - date
        if delta.total_seconds() < 0: # next hour
            output = '{0},{1},{2}'.format(time.mktime(lastHour.timetuple()), lastPrice, volume)
            #print 'writing: {}'.format(output)
            out.write('{}\n'.format(output))
            lastHour = nextHour(date)
            lastPrice = data[1]
            lastDelta = lastHour - date
            volume = 0
        elif delta < lastDelta:
            lastDelta = delta
            lastPrice = data[1]
        #print '{0} | {1}'.format(delta, lastDelta)
        volume = volume + float(data[2])