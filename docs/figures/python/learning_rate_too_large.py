import matplotlib.pyplot as plt
import numpy as np

xi = 2
yi = -1
wc = 5
eta = .6

def L(w):
    return 1/2 * (1 - w * xi)**2

def dw_L(w):
    return -(1 - w * xi) * xi

def update(w):
    return w - eta * dw_L(w)

ws = [ wc ]
for _ in range(4):
    wn = update(wc)
    ws.append(wn)
    wc = wn

ws = np.array(ws)

w = np.linspace(-20, 20, 1000)
plt.plot(w, L(w))
plt.xlabel('w')
plt.ylabel('L(w)')

x = ws
y = L(ws)
plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1, width=0.005)

plt.show()
