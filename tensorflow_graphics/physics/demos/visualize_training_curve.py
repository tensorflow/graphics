import matplotlib.pyplot as plt

file_name = 'rolling_0.3.log'
fin = open(file_name, 'r')

x, y = [], []
for line in fin:
    xx, yy = line[:-1].split(' ')
    x.append(int(xx))
    y.append(float(yy))

plt.plot(x, y)
plt.show()
