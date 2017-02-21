#cd c:\Users\kamch\Dropbox\Science\PhD\Orsay\Measurements\MagField200Gauss\DC2
#python TaskTrip.py
import pandas as pd
import matplotlib as mpl
import numpy as np
import array
from vector import vector, plot_peaks
import pylab as pl


elem = np.array([])
with open("testupto30up.txt") as file:
     for row in file:
        if ('%') not in row:
            elem = np.append(elem,row.split())
#print(elem)
#первая колонка: частота,
#вторая: магнитное поле,
#четвёртая: интенсивность
#точек 2500 в каждом спектре/ 5000, в последнем 4854
#спектров 124 штуки / 30
a = int(elem.shape[0])/7
print(elem.size)
b = 5000 #number of points in each spectrum
c = 30 #number os pectrums
elem.shape = (int(a),7)
vectors = np.zeros((5000,30))
i = 0
for j in range(int(a)):
    k = int(j) - int(i) * b
    vectors[k, i] += float(elem[j, 3])
    print(j,i)
    if float(elem[j, 0]) == 2.500000E+9:
        i += 1
print(vectors.shape)
print(vectors)
#13/12/2016 12:04:32
