#1) Считать данные
#2) Построить 3д график
#3) Положить данные в pandas.dataframe
#4) Построить тероретический график при заданном Н
#5) Построить сверху эмпирический график при заданном Н
#6) Решить простейшую задачу в scipy.minimize
#7) Найти значения Фи и Тэта для заданного поля
#8) Оценить, насколько эти значения описывают данные при других Н
#9) Захуярить в все в класс с итератором

# Поехали:
#1)

import os.path
import pandas as pd

def CreateClearFile():
    f = open('testupto30_clear.csv', 'w')
    f.write('frequency,magnetic,intensity\n')
    with open('testupto30up.txt', 'r') as fp:
       for line in fp:
           if line[0]!='%':
               b=line.strip().split('\t')
               if len(b)>3:
                   c = [b[0], b[1], b[3]]
                   f.write(','.join([str(x) for x in c])+'\n')
       fp.close()
    f.close()

if not (os.path.exists('testupto30_clear.csv')):
    CreateClearFile()

df = pd.read_csv('testupto30_clear.csv')
print(df)



#2)

import math
import matplotlib.pyplot as plt
import scipy as sp


#a = sp.random.random((16, 16))
#plt.imshow(a, cmap='hot', interpolation='nearest')
#plt.show()

#5)
'''
import math
from scipy.optimize import minimize_scalar

def alpha(x):
    print('x'+str(x))
    return x*x+3

result = minimize_scalar(alpha, bounds=[-10., 10.], method = 'bounded')

print(result)

print('\n')
import scipy.optimize as op

def F(x):
    print('x: ' + str(x[0])+ ', y:' + str(x[1]))
    return x[0]*x[0]+x[1]*(x[1]-3)

op.fmin(F, [-10., 100.])

'''
