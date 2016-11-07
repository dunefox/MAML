import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 0])
y = np.array([0, 0, 1])
plt.scatter(x, y, c='red')
x = np.array([1])
y = np.array([1])
plt.scatter(x, y, c='blue')

W = -1
b = 1.5
x = np.linspace(0, 1.5)
plt.plot(x, W*x+b, c='gray')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.show()
