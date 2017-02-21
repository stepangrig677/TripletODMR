#cd c:\Users\kamch\Dropbox\Science\PhD\Orsay
#python TaskTrip.py

import pandas as pd
import pylab as pl
import numpy as np
import array
data = np.loadtxt("testupto30up.txt", comments='%')#, usecols=(0,1,3),unpack=True)
field = np.zeros(29)
freq = data[:5000,0]
Intensity = np.zeros((29,5000))
for i in range(29):
	y[i] = np.mean(data[i*5000:(i+1)*5000,1])
	z[i,:] = data[i*5000:(i+1)*5000,3]

#x1,y1 = np.meshgrid(x,y)
pl.figure()
pl.pcolor(freq, field, Intensity)
pl.xlabel(" Frequency")
pl.ylabel(" B (G)")
pl.show()

df = pd.DataFrame(np.zeros(45,45,29), items = [bla-,bla], major_axis = Phi, minor_axis = )

lrw35@cam.ac.uk
#
