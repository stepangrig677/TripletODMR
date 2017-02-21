import pylab as pl
import numpy as np

data = np.loadtxt("testupto30up.txt", comments='%')
freq = data[:5000,0]
B = np.zeros(29)
I = np.zeros((29,5000))
for i in range(29):
	B[i] = np.mean(data[i*5000:(i+1)*5000,1])
	I[i,:] = data[i*5000:(i+1)*5000,3]

pl.figure()
pl.pcolor(freq, B, I)
pl.show()
