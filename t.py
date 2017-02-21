from PIL import Image, ImageDraw
from decimal import Decimal
from gradient import z_gradient

'''
сокращаем число на общий коофицент
input:  num число
        koof коофицент (правим ручками до удобоваримого вывода)
return: если число меньше 0,01, то умножаем на коофицент (1,7Е-8 -> 0.017)
        иначе делим на коофицент (3Е+8 -> 300.0)
'''
def optim(num, koof = 1E+6):
  if float(num) < float(0.01):
    return float(num) * float(koof)
  else:
    return float(num) / float(koof)


'''
двигаем занчения влево до минимального значения равного нулю
input: arr - массив чисел [[x,y,z], [x1,y1,z1],...,[xn,yn,zn]]
return: массив чисел x, массив чисел y, массив чисел z, длина массива len
'''
def optim_arr(arr):
  maxa = tuple(map(max, zip(*arr))) # строим массив из максимальных значений [max_x, max_y, max_z]
  print('max list', maxa)
  mina = tuple(map(min, zip(*arr)))# строим массив из минимальных значений [min_x, min_y, min_z]
  print('min list', mina)
  x = ([n - mina[0] for n in tuple(zip(*arr))[0]]) # обходим массив по значениям x, каждый элемент уменьшаем на минимальное значение (двигаем влево)
  y = ([n - mina[1] for n in tuple(zip(*arr))[1]]) # обходим массив по значениям y, каждый элемент уменьшаем на минимальное значение (двигаем влево)
  z = ([n - mina[2] for n in tuple(zip(*arr))[2]]) # обходим массив по значениям z, каждый элемент уменьшаем на минимальное значение (двигаем влево)
  print('min values', min(x),min(y),min(z)) # должен выводить нули
  print('max values', max(x),max(y),max(z))
  return {'x':x,'y':y,'z':z, 'len': len(arr)}
  
def extract_data(filename):
    infile = open(filename, 'r')
    coords = []
    for line in infile:
      if ('%') not in line:
        words = line.split()
        
        if len(words) != 0:
          w = [optim( words[0] ), optim(words[1]), optim(words[3])]
        coords.append(w)
    infile.close()
    return optim_arr(coords)


values = extract_data('testupto30up.txt')


grad = z_gradient(values['z'], '#000000', '#ff000') # строим градиендную маску, см gradient.py:42

img = Image.new('RGB',(int(max(values['x'])),int(max(values['y']))), (255, 255, 255)) # создаем изображение с белым фоном размером количества элементов по ис и игрек
#img = Image.new('RGB',(30000, 30000), (255, 255, 255)) # создаем изображение с белым фоном размером количества элементов по ис и игрек
draw = ImageDraw.Draw(img) 


i=0
while i < values['len']:
  x = values['x'][i]
  y = values['y'][i]
  z = values['z'][i]
  for gr in grad:
    if z > gr['min'] and z < gr['max']: # если значение z входит в промежуток градиентной маски, то присваиваем ему цвет
      color = gr['color']
  print(x,y,color)
 # draw.point((x,y), fill=color) # рисуем пиксель с координтой икс игрек и цветом от зет
  draw.rectangle((x,y,x+1,y+100), fill=color, outline=color)
  i=i+1
del draw
img.save("test.png", "PNG") # сохраниям изображени
