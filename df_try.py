import pandas as pd
import numpy as np
import math

#hpi = math.pi/2
a = 91*math.pi/180
b = a/45
c = 402/30
d = 401+c
Phi = np.arange(0,a,b)
Theta = np.arange(0,a,b)
Magnetic = np.arange(0,d,c)
iterables = [Phi, Theta]#, Magnetic]
e = len(Phi)*len(Theta)
f = len(Magnetic)
index = pd.MultiIndex.from_product(iterables, names=['Phi1', 'Theta1']#, 'Field'])
daf = pd.DataFrame(np.random.randn( f, e), index = Magnetic, columns = index)
#s = pd.Series(np.random.randn(62775), index=index)

print(s)
#print(len(Phi),len(Theta),len(Magnetic))
"""
Предыдущий вариант, который работал:
iterables = [Phi, Theta, Magnetic]
index = pd.MultiIndex.from_product(iterables, names=['Phi1', 'Theta1', 'Field'])
s = pd.Series(np.random.randn(62775), index=index)
"""